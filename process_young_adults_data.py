from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from config.preprocessing import (
    custom_merge_config_young_adults,
    default_merge_config_young_adults,
)
from silver_pain.merge import join_self_report_to_physio
from silver_pain.preprocessing import (
    SCIPY_OK,
    build_empatica_session_merged_df,
    process_empatica_e4_physio_all_subjects,
)


def parse_subjects(raw_values: list[str] | None) -> list[str] | None:
    if not raw_values:
        return None

    subjects: set[str] = set()
    for raw in raw_values:
        for part in raw.split(","):
            subject = part.strip()
            if subject:
                subjects.add(subject)

    return sorted(subjects) or None


def load_config(preset: str):
    presets = {
        "default": default_merge_config_young_adults,
        "custom": custom_merge_config_young_adults,
    }
    return replace(presets[preset])


def warn_if_scipy_required(cfg) -> None:
    if (cfg.map_method == "interp" or cfg.hr_map_interp_kind in ("quadratic", "cubic")) and not SCIPY_OK:
        print(
            "[WARN] SciPy not available: quadratic/cubic interpolation will fall back to linear; "
            "bandpass disabled."
        )


def gather_subject_dirs(root_original: Path, subjects: list[str] | None) -> list[Path]:
    if subjects is None:
        subject_dirs = sorted(path for path in root_original.iterdir() if path.is_dir())
    else:
        subject_dirs = []
        for subject in subjects:
            subject_dir = root_original / subject
            if not subject_dir.is_dir():
                raise FileNotFoundError(f"Missing subject folder: {subject_dir}")
            subject_dirs.append(subject_dir)
    if not subject_dirs:
        raise FileNotFoundError(f"No subject folders found in {root_original}")
    return subject_dirs


def merge_selected_subjects(root_original: Path, out_physio: Path, cfg, subjects: list[str] | None) -> None:
    out_physio.mkdir(parents=True, exist_ok=True)
    if subjects is None:
        process_empatica_e4_physio_all_subjects(str(root_original), str(out_physio), cfg)
        return

    for subject_dir in gather_subject_dirs(root_original, subjects):
        subject_id = subject_dir.name
        merged = build_empatica_session_merged_df(str(subject_dir), subject_id, cfg)
        out_path = out_physio / f"{subject_id}_merged_64hz.csv"
        merged.to_csv(out_path, index=False)
        print(f"[OK] {subject_id} -> {out_path}")


def join_self_reports(
    merged_dir: Path,
    subjects_root: Path,
    out_dir: Path,
    report_filename: str,
    merged_glob: str,
    out_name_template: str,
    time_window_csv: Path | None,
    pain_tz: str,
    max_snap_s: float | None,
    subjects: list[str] | None,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_paths = sorted(merged_dir.glob(merged_glob))
    if subjects is not None:
        allowed = set(subjects)
        merged_paths = [path for path in merged_paths if path.name.split("_")[0] in allowed]

    if not merged_paths:
        raise FileNotFoundError(f"No merged files found in {merged_dir} with glob {merged_glob}")

    written_paths: list[Path] = []
    for phys_path in merged_paths:
        subject_id = phys_path.name.split("_")[0]
        sr_path = subjects_root / subject_id / report_filename
        if not sr_path.exists():
            subject_dir = subjects_root / subject_id
            if subject_dir.is_dir():
                files = {path.name.lower(): path for path in subject_dir.iterdir() if path.is_file()}
                sr_path = files.get(report_filename.lower(), sr_path)

        if not sr_path.exists():
            print(f"[WARN] Missing self report for subject {subject_id}: {sr_path}; skipping")
            continue

        out_path = out_dir / out_name_template.format(subject_id=subject_id)
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
    parser = argparse.ArgumentParser(description="Process SILVER young adults data from raw files to cohort exports.")
    parser.add_argument("--root-data-folder", default="./data")
    parser.add_argument("--cohort-folder")
    parser.add_argument("--root-original")
    parser.add_argument("--out-physio")
    parser.add_argument("--out-with-self-report")
    parser.add_argument("--time-window-csv")
    parser.add_argument("--config-preset", choices=("default", "custom"), default="default")
    parser.add_argument("--report-filename", default="pain_data.csv")
    parser.add_argument("--merged-glob", default="*_merged_64hz.csv")
    parser.add_argument("--out-name-template", default="{subject_id}_merged_64hz_with_pain.csv")
    parser.add_argument("--cohort-csv")
    parser.add_argument("--cohort-feather")
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--skip-self-report", action="store_true")
    parser.add_argument("--skip-cohort-export", action="store_true")
    parser.add_argument("--subjects", action="append", help="Repeat or pass comma-separated subject IDs, e.g. 101,102")
    parser.add_argument("--pain-tz")
    parser.add_argument("--pain-max-snap-s", type=float)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    root_data_folder = Path(args.root_data_folder)
    cohort_folder = Path(args.cohort_folder) if args.cohort_folder else root_data_folder / "young_adults"
    root_original = Path(args.root_original) if args.root_original else cohort_folder / "original_files"
    out_physio = Path(args.out_physio) if args.out_physio else cohort_folder / "processed_data" / "combined"
    out_with_self_report = (
        Path(args.out_with_self_report)
        if args.out_with_self_report
        else cohort_folder / "processed_data" / "with_self_report"
    )
    time_window_csv = Path(args.time_window_csv) if args.time_window_csv else cohort_folder / "experiment_time.csv"
    cohort_csv = Path(args.cohort_csv) if args.cohort_csv else cohort_folder / "young_adults.csv"
    cohort_feather = Path(args.cohort_feather) if args.cohort_feather else cohort_folder / "young_adults.feather"
    subjects = parse_subjects(args.subjects)

    if not root_original.is_dir():
        raise FileNotFoundError(f"Missing folder: {root_original}")

    cfg = load_config(args.config_preset)
    if args.pain_tz:
        cfg.pain_tz = args.pain_tz
    if args.pain_max_snap_s is not None:
        cfg.pain_max_snap_s = args.pain_max_snap_s

    warn_if_scipy_required(cfg)

    out_physio.mkdir(parents=True, exist_ok=True)
    out_with_self_report.mkdir(parents=True, exist_ok=True)

    if not args.skip_merge:
        merge_selected_subjects(root_original, out_physio, cfg, subjects)

    labeled_paths: list[Path]
    if args.skip_self_report:
        labeled_paths = sorted(out_with_self_report.glob("*.csv"))
        if subjects is not None:
            allowed = set(subjects)
            labeled_paths = [path for path in labeled_paths if path.name.split("_")[0] in allowed]
        if not labeled_paths:
            raise FileNotFoundError(f"No labeled CSV files found in {out_with_self_report}")
    else:
        labeled_paths = join_self_reports(
            merged_dir=out_physio,
            subjects_root=root_original,
            out_dir=out_with_self_report,
            report_filename=args.report_filename,
            merged_glob=args.merged_glob,
            out_name_template=args.out_name_template,
            time_window_csv=time_window_csv if time_window_csv.exists() else None,
            pain_tz=cfg.pain_tz,
            max_snap_s=cfg.pain_max_snap_s,
            subjects=subjects,
        )

    if not args.skip_cohort_export:
        export_cohort(labeled_paths, cohort_csv, cohort_feather)


if __name__ == "__main__":
    main()
