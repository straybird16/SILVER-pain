from __future__ import annotations

import argparse
import glob
from dataclasses import replace
from pathlib import Path

import pandas as pd

from config.preprocessing import (
    custom_merge_config_older_adults,
    default_merge_config_older_adults,
)
from silver_pain.merge import join_self_report_to_physio
from silver_pain.preprocessing import SCIPY_OK, process_all_subjects, process_subject


def parse_subjects(raw_values: list[str] | None) -> list[str] | None:
    if not raw_values:
        return None

    subjects: set[str] = set()
    for raw in raw_values:
        for part in raw.split(","):
            subject = part.strip()
            if subject:
                subjects.add(subject.zfill(3))

    return sorted(subjects) or None


def load_config(preset: str):
    presets = {
        "default": default_merge_config_older_adults,
        "custom": custom_merge_config_older_adults,
    }
    return replace(presets[preset])


def warn_if_scipy_required(cfg) -> None:
    if (cfg.map_method == "interp" or cfg.hr_interp_kind in ("quadratic", "cubic")) and not SCIPY_OK:
        print("[WARN] SciPy not available: quadratic/cubic will fall back to linear; bandpass disabled.")


def find_embraceplus_v6_dir(physio_root: Path, subject_id: str) -> Path:
    pattern = str(physio_root / f"{subject_id}-*" / "raw_data" / "v6")
    hits = sorted(Path(path) for path in glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"Could not find EmbracePlus v6 folder for {subject_id}: {pattern}")
    if len(hits) > 1:
        print(f"[WARN] Multiple matches for {subject_id}; using: {hits[0]}")
    return hits[0]


def discover_subjects(physio_root: Path) -> list[str]:
    discovered = sorted(
        {
            path.name.split("-")[0]
            for path in physio_root.iterdir()
            if path.is_dir() and "-" in path.name and (path / "raw_data" / "v6").is_dir()
        }
    )
    if discovered:
        return discovered

    fallback = [f"{index:03d}" for index in range(1, 8)]
    existing = []
    for subject_id in fallback:
        pattern = str(physio_root / f"{subject_id}-*" / "raw_data" / "v6")
        if glob.glob(pattern):
            existing.append(subject_id)
    if existing:
        return existing
    raise FileNotFoundError(f"No EmbracePlus subject folders found in {physio_root}")


def extract_subjects(physio_root: Path, extraction_output_dir: Path, subjects: list[str]) -> None:
    from silver_pain.extraction import parse_folder_to_channel_dfs, write_channel_csvs

    extraction_output_dir.mkdir(parents=True, exist_ok=True)
    for subject_id in subjects:
        raw_v6_dir = find_embraceplus_v6_dir(physio_root, subject_id)
        out_dir = extraction_output_dir / subject_id
        print(f"[SUBJECT {subject_id}] raw={raw_v6_dir}")
        channel_dfs = parse_folder_to_channel_dfs(str(raw_v6_dir))
        write_channel_csvs(channel_dfs, out_dir=str(out_dir), prefix=subject_id)


def ensure_extraction_subjects(extraction_output_dir: Path, subjects: list[str]) -> None:
    missing = [subject_id for subject_id in subjects if not (extraction_output_dir / subject_id).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Missing extracted subject folders in {extraction_output_dir}: {', '.join(missing)}"
        )


def merge_subjects(extraction_output_dir: Path, combined_dir: Path, cfg, subjects: list[str] | None) -> None:
    combined_dir.mkdir(parents=True, exist_ok=True)
    if subjects is None:
        process_all_subjects(str(extraction_output_dir), str(combined_dir), cfg)
        return

    for subject_id in subjects:
        subject_dir = extraction_output_dir / subject_id
        if not subject_dir.is_dir():
            raise FileNotFoundError(f"Missing extracted subject folder: {subject_dir}")
        out_path = combined_dir / f"{subject_id}_merged_64hz.csv"
        print(f"[SUBJECT {subject_id}] -> {out_path}")
        process_subject(str(subject_dir), subject_id, str(out_path), cfg)


def join_self_reports(
    combined_dir: Path,
    self_report_dir: Path,
    out_dir: Path,
    merged_glob: str,
    time_window_csv: Path | None,
    pain_tz: str,
    max_snap_s: float | None,
    subjects: list[str],
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_paths = sorted(combined_dir.glob(merged_glob))
    allowed = set(subjects)
    merged_paths = [path for path in merged_paths if path.name.split("_")[0] in allowed]
    if not merged_paths:
        raise FileNotFoundError(f"No merged files found in {combined_dir} with glob {merged_glob}")

    written_paths: list[Path] = []
    for phys_path in merged_paths:
        subject_id = phys_path.name.split("_")[0]
        candidates = [
            self_report_dir / f"{subject_id}.csv",
            self_report_dir / f"{subject_id}_self_report.csv",
            self_report_dir / f"{subject_id}_selfreport.csv",
        ]
        sr_path = next((path for path in candidates if path.exists()), None)
        if sr_path is None:
            fallback = sorted(self_report_dir.glob(f"*{subject_id}*.csv"))
            sr_path = fallback[0] if fallback else None
        if sr_path is None:
            print(f"[WARN] no self report for subject {subject_id}; skipping")
            continue

        out_path = out_dir / f"{subject_id}_merged_64hz_with_self_report.csv"
        join_self_report_to_physio(
            str(phys_path),
            str(sr_path),
            str(out_path),
            keep_action=False,
            subject_id=subject_id,
            time_window_csv=str(time_window_csv) if time_window_csv else None,
            tz_local=pain_tz,
            max_snap_s=max_snap_s,
        )
        print(f"[OK] {subject_id}: {phys_path} + {sr_path} -> {out_path}")
        written_paths.append(out_path)

    if not written_paths:
        raise FileNotFoundError(f"No self-report outputs were produced in {out_dir}")
    return written_paths


def export_cohort(paths: list[Path], cohort_csv: Path, cohort_feather: Path) -> None:
    frames = []
    for file_path in paths:
        subject_df = pd.read_csv(file_path)
        subject_df["subject"] = file_path.stem.split("_")[0]
        frames.append(subject_df)

    if not frames:
        raise FileNotFoundError("No CSV files available for cohort export.")

    df = pd.concat(frames, ignore_index=True)
    cohort_feather.parent.mkdir(parents=True, exist_ok=True)
    cohort_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_feather(cohort_feather)
    df.to_csv(cohort_csv, index=False)
    print(f"[OK] Wrote cohort files: {cohort_csv} and {cohort_feather}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process SILVER older adults data from raw files to cohort exports.")
    parser.add_argument("--root-data-folder", default="./data")
    parser.add_argument("--cohort-folder")
    parser.add_argument("--root-original")
    parser.add_argument("--physio-root")
    parser.add_argument("--self-report-folder")
    parser.add_argument("--extraction-output-dir")
    parser.add_argument("--processed-data-output-dir")
    parser.add_argument("--combined-dir")
    parser.add_argument("--with-self-report-dir")
    parser.add_argument("--time-window-csv")
    parser.add_argument("--config-preset", choices=("default", "custom"), default="default")
    parser.add_argument("--subjects", action="append", help="Repeat or pass comma-separated subject IDs, e.g. 001,003")
    parser.add_argument("--merged-glob", default="*_merged_*hz.csv")
    parser.add_argument("--cohort-csv")
    parser.add_argument("--cohort-feather")
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--skip-self-report", action="store_true")
    parser.add_argument("--skip-cohort-export", action="store_true")
    parser.add_argument("--pain-tz")
    parser.add_argument("--pain-max-snap-s", type=float)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    root_data_folder = Path(args.root_data_folder)
    cohort_folder = Path(args.cohort_folder) if args.cohort_folder else root_data_folder / "older_adults"
    root_original = Path(args.root_original) if args.root_original else cohort_folder / "original_files"
    physio_root = Path(args.physio_root) if args.physio_root else root_original / "physiological_signal"
    self_report_folder = Path(args.self_report_folder) if args.self_report_folder else root_original / "self_report"
    extraction_output_dir = (
        Path(args.extraction_output_dir)
        if args.extraction_output_dir
        else cohort_folder / "extraction_from_original_files"
    )
    processed_data_output_dir = (
        Path(args.processed_data_output_dir)
        if args.processed_data_output_dir
        else cohort_folder / "processed_data"
    )
    combined_dir = Path(args.combined_dir) if args.combined_dir else processed_data_output_dir / "combined"
    with_self_report_dir = (
        Path(args.with_self_report_dir)
        if args.with_self_report_dir
        else processed_data_output_dir / "with_self_report"
    )
    time_window_csv = Path(args.time_window_csv) if args.time_window_csv else cohort_folder / "experiment_time.csv"
    cohort_csv = Path(args.cohort_csv) if args.cohort_csv else cohort_folder / "older_adults.csv"
    cohort_feather = Path(args.cohort_feather) if args.cohort_feather else cohort_folder / "older_adults.feather"

    if not physio_root.is_dir():
        raise FileNotFoundError(f"Missing physiological signal folder: {physio_root}")
    if not self_report_folder.is_dir():
        raise FileNotFoundError(f"Missing self-report folder: {self_report_folder}")

    subjects = parse_subjects(args.subjects)
    if subjects is None:
        subjects = discover_subjects(physio_root)

    cfg = load_config(args.config_preset)
    if args.pain_tz:
        cfg.pain_tz = args.pain_tz
    if args.pain_max_snap_s is not None:
        cfg.pain_max_snap_s = args.pain_max_snap_s

    warn_if_scipy_required(cfg)

    extraction_output_dir.mkdir(parents=True, exist_ok=True)
    combined_dir.mkdir(parents=True, exist_ok=True)
    with_self_report_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_extraction:
        extract_subjects(physio_root, extraction_output_dir, subjects)
    else:
        ensure_extraction_subjects(extraction_output_dir, subjects)

    if not args.skip_merge:
        merge_subjects(extraction_output_dir, combined_dir, cfg, subjects if args.subjects else None)

    labeled_paths: list[Path]
    if args.skip_self_report:
        allowed = set(subjects)
        labeled_paths = [
            path
            for path in sorted(with_self_report_dir.glob("*.csv"))
            if path.name.split("_")[0] in allowed
        ]
        if not labeled_paths:
            raise FileNotFoundError(f"No labeled CSV files found in {with_self_report_dir}")
    else:
        labeled_paths = join_self_reports(
            combined_dir=combined_dir,
            self_report_dir=self_report_folder,
            out_dir=with_self_report_dir,
            merged_glob=args.merged_glob,
            time_window_csv=time_window_csv if time_window_csv.exists() else None,
            pain_tz=cfg.pain_tz,
            max_snap_s=cfg.pain_max_snap_s,
            subjects=subjects,
        )

    if not args.skip_cohort_export:
        export_cohort(labeled_paths, cohort_csv, cohort_feather)


if __name__ == "__main__":
    main()
