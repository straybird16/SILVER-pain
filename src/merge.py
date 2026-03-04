"""
src.merge

Utilities to merge self-report pain annotations with merged physiological time series.

Core behavior:
- Parse self-report timestamps (either local wall-clock time or numeric epoch time)
- Convert self-report timestamps to UTC
- Snap each self-report row to the nearest physiological timestamp
  (optionally enforcing a maximum snapping tolerance)

This module is intentionally conservative about timestamp handling and relies on
pandas datetime alignment to avoid ns/us/ms unit mismatch bugs.
"""


import os
import glob
import numpy as np
import pandas as pd
from typing import Optional


def _infer_epoch_unit(values: pd.Series) -> str:
    """Infer epoch unit for numeric timestamps.

    Heuristic based on magnitude:
      - ns: ~1e18
      - us: ~1e15
      - ms: ~1e12
      - s : ~1e9 or smaller

    Returns one of: {"s", "ms", "us", "ns"}.
    """
    v = pd.to_numeric(values, errors="coerce")
    if v.notna().sum() == 0:
        return "ns"
    m = float(np.nanmedian(np.abs(v.to_numpy(dtype=np.float64))))
    if m >= 1e17:
        return "ns"
    if m >= 1e14:
        return "us"
    if m >= 1e11:
        return "ms"
    return "s"


def _coerce_timestamp_utc(ts: pd.Series, tz_local: str) -> pd.Series:
    """Parse self-report timestamps and return tz-aware UTC datetimes."""
    t = pd.to_datetime(ts, errors="coerce")
    if hasattr(t.dt, "tz") and t.dt.tz is not None:
        return t.dt.tz_convert("UTC")
    return (
        t.dt.tz_localize(tz_local, ambiguous="infer", nonexistent="shift_forward")
        .dt.tz_convert("UTC")
    )

def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize common column-name variants to a canonical schema:
      - timestamp (local time)
      - PainLevel
      - Trial (optional)
      - Action (optional)
    """
    if df is None or df.empty:
        return df

    cols = {c: str(c).strip() for c in df.columns}
    lower = {c: cols[c].lower() for c in df.columns}

    # timestamp column
    ts_candidates = {"timestamp", "time", "datetime", "date_time"}
    ts_col = next((c for c, lc in lower.items() if lc in ts_candidates), None)

    # pain column
    pain_candidates = {"painlevel", "pain_level", "pain", "nrs", "rating"}
    pain_col = next((c for c, lc in lower.items() if lc in pain_candidates), None)

    # trial / action
    trial_col = next((c for c, lc in lower.items() if lc == "trial"), None)
    action_col = next((c for c, lc in lower.items() if lc == "action"), None)

    out = df.copy()
    if ts_col and ts_col != "timestamp":
        out = out.rename(columns={ts_col: "timestamp"})
    if pain_col and pain_col != "PainLevel":
        out = out.rename(columns={pain_col: "PainLevel"})
    if trial_col and trial_col != "Trial":
        out = out.rename(columns={trial_col: "Trial"})
    if action_col and action_col != "Action":
        out = out.rename(columns={action_col: "Action"})

    return out

def load_self_report(self_report_csv: str, tz_local: str = "America/New_York") -> pd.DataFrame:
    """Load a self-report CSV and normalize timestamps to UTC.

    The function accepts either:
      1) local wall-clock timestamps (e.g., 'YYYY-MM-DD HH:MM:SS.sss' in America/New_York), or
      2) numeric epoch timestamps (seconds/ms/us/ns; unit inferred), interpreted as UTC.

    Accepted file formats
    ---------------------
    1. Headered CSV including timestamp and pain columns (name variants accepted).
    2. Headerless CSV with first two columns interpreted as timestamp and pain level.

    Returns
    -------
    pandas.DataFrame
        Original columns plus `timestamp_utc` and normalized `PainLevel`.
    """
    if not os.path.isfile(self_report_csv):
        raise FileNotFoundError(self_report_csv)

    # First attempt: headered CSV.
    sr = pd.read_csv(self_report_csv)

    if "timestamp" not in {str(c).strip().lower() for c in sr.columns} and "timestamp" not in sr.columns:
        # Likely headerless pain_data.csv (young cohort).
        sr = pd.read_csv(self_report_csv, header=None)
        # Expect at least 2 columns: timestamp, pain
        if sr.shape[1] < 2:
            raise ValueError(f"Unrecognized self-report format: {self_report_csv}")
        # Keep first three columns if present.
        cols = ["timestamp", "PainLevel", "subject_id"]
        sr = sr.iloc[:, : min(3, sr.shape[1])].copy()
        sr.columns = cols[: sr.shape[1]]

    sr = _canonicalize_columns(sr)

    if "timestamp" not in sr.columns or "PainLevel" not in sr.columns:
        raise ValueError(f"Self-report CSV is missing required columns: {self_report_csv}")

    sr["PainLevel"] = pd.to_numeric(sr["PainLevel"], errors="coerce")
    sr["timestamp_utc"] = _coerce_timestamp_utc(sr["timestamp"], tz_local=tz_local)
    #sr["timestamp_ns"] = sr["timestamp_utc"].astype("int64")

    if "Trial" in sr.columns:
        trial_num = pd.to_numeric(sr["Trial"], errors="coerce")
        sr["Trial"] = trial_num.ffill().bfill().round().astype("Int32")

    return sr


def _norm_subject_id(x) -> str:
    s = str(x).strip()
    s2 = s.lstrip("0")
    return s2 if s2 != "" else "0"


def load_subject_time_windows(time_window_csv: str, tz_local: str = "America/New_York") -> pd.DataFrame:
    """
    Parse per-subject start/end windows and convert to UTC.
    """
    w = pd.read_csv(time_window_csv, dtype={"subject": str})

    required = {"subject", "start", "end"}
    if not required.issubset(set(w.columns)):
        return pd.DataFrame(columns=["subject_norm", "start_utc", "end_utc"])

    start_local = pd.to_datetime(w["start"], errors="coerce")
    end_local   = pd.to_datetime(w["end"],   errors="coerce")

    start_utc = (start_local
                 .dt.tz_localize(tz_local, ambiguous="infer", nonexistent="shift_forward")
                 .dt.tz_convert("UTC"))
    end_utc = (end_local
               .dt.tz_localize(tz_local, ambiguous="infer", nonexistent="shift_forward")
               .dt.tz_convert("UTC"))

    out = pd.DataFrame({
        "subject_norm": w["subject"].map(_norm_subject_id),
        "start_utc": start_utc,
        "end_utc": end_utc,
    })

    # keep only valid
    out = out[out["start_utc"].notna() & out["end_utc"].notna()].copy()
    if out.empty:
        return out

    # enforce start <= end
    s = out["start_utc"].to_numpy()
    e = out["end_utc"].to_numpy()
    out["start_utc"] = np.minimum(s, e)
    out["end_utc"]   = np.maximum(s, e)

    return out[["subject_norm", "start_utc", "end_utc"]]


def _parse_physio_timestamps_utc(df: pd.DataFrame, physio_csv: str) -> pd.Series:
    """Convert `timestamp_ns` column to a UTC datetime series with unit inference."""
    if "timestamp_ns" not in df.columns:
        raise ValueError(f"{physio_csv} missing timestamp_ns")

    ts_num = pd.to_numeric(df["timestamp_ns"], errors="coerce")
    if ts_num.notna().sum() > 0 and ts_num.notna().mean() > 0.8:
        unit = _infer_epoch_unit(ts_num)
        return pd.to_datetime(ts_num, unit=unit, utc=True, errors="coerce")

    return pd.to_datetime(df["timestamp_ns"], utc=True, errors="coerce")


def _filter_by_windows(
    df: pd.DataFrame,
    ts_utc: pd.Series,
    windows: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Filter a DataFrame/time-series pair by one or more UTC windows."""
    if windows is None or windows.empty:
        return df, ts_utc

    mask = np.zeros(len(df), dtype=bool)
    for start_utc, end_utc in windows[["start_utc", "end_utc"]].itertuples(index=False, name=None):
        mask |= (ts_utc >= start_utc) & (ts_utc <= end_utc)
    return df.loc[mask].copy().reset_index(drop=True), ts_utc.loc[mask].reset_index(drop=True)


def _ensure_alignment_columns(df: pd.DataFrame, keep_action: bool) -> pd.DataFrame:
    """Ensure output frame has required label columns with expected dtypes."""
    if "PainLevel" not in df.columns:
        df["PainLevel"] = np.nan
    if "Trial" not in df.columns:
        df["Trial"] = pd.Series(pd.array([pd.NA] * len(df), dtype="Int32"), index=df.index)
    if keep_action and "Action" not in df.columns:
        df["Action"] = np.nan
    return df


def _write_csv(df: pd.DataFrame, out_csv: str) -> None:
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(out_csv, index=False)


def join_self_report_to_physio(
    physio_csv: str,
    self_report_csv: str,
    out_csv: str,
    keep_action: bool = False,
    subject_id: str | None = None,
    time_window_csv: str | None = None,
    tz_local: str = "America/New_York",
    max_snap_s: Optional[float] = None,
) -> pd.DataFrame:
    """Join one self-report file to one merged physiology file by nearest UTC time.

    Parameters
    ----------
    physio_csv
        Merged physiology CSV with `timestamp_ns`.
    self_report_csv
        Self-report CSV, either headered or headerless.
    out_csv
        Output CSV path.
    keep_action
        Preserve self-report `Action` when available.
    subject_id
        Subject identifier for time-window filtering.
    time_window_csv
        Optional CSV containing subject-specific start/end windows.
    tz_local
        Time zone for naive self-report wall-clock timestamps.
    max_snap_s
        Optional maximum nearest-neighbor snapping tolerance in seconds.
    """
    df = pd.read_csv(physio_csv)
    df_ts_utc = _parse_physio_timestamps_utc(df, physio_csv)

    df["_ts_utc"] = df_ts_utc
    df = df.sort_values("_ts_utc", kind="mergesort").reset_index(drop=True)
    df_ts_utc = df.pop("_ts_utc")

    windows_for_subject = None
    if time_window_csv and subject_id is not None:
        win = load_subject_time_windows(time_window_csv)
        if not win.empty:
            sid_norm = _norm_subject_id(subject_id)
            windows_for_subject = win[win["subject_norm"] == sid_norm].copy()
            df, df_ts_utc = _filter_by_windows(df, df_ts_utc, windows_for_subject)

    if df.empty:
        df = _ensure_alignment_columns(df, keep_action)
        if "Trial" in df.columns:
            df["Trial"] = pd.Series(dtype="Int32")
        _write_csv(df, out_csv)
        return df

    sr = load_self_report(self_report_csv, tz_local)

    if windows_for_subject is not None and not sr.empty:
        sr, _ = _filter_by_windows(sr, sr["timestamp_utc"], windows_for_subject)

    if not sr.empty:
        sr = sr[sr["timestamp_utc"].notna()].copy()

    if sr.empty:
        df = _ensure_alignment_columns(df, keep_action)
        df["Trial"] = pd.Series(dtype="Int32")
        _write_csv(df, out_csv)
        return df

    grid = pd.DatetimeIndex(df_ts_utc)
    targets = pd.DatetimeIndex(sr["timestamp_utc"])
    tolerance = pd.Timedelta(seconds=max_snap_s) if max_snap_s is not None else None
    nearest_idx = grid.get_indexer(targets, method="nearest", tolerance=tolerance)
    valid = nearest_idx >= 0
    nearest_idx = nearest_idx[valid]
    sr_aligned = sr.iloc[np.flatnonzero(valid)].copy()

    df = _ensure_alignment_columns(df, keep_action)

    df.loc[nearest_idx, "PainLevel"] = sr_aligned["PainLevel"].to_numpy()
    if "Trial" in sr_aligned.columns:
        df.loc[nearest_idx, "Trial"] = sr_aligned["Trial"].to_numpy()
    if keep_action and "Action" in sr_aligned.columns:
        df.loc[nearest_idx, "Action"] = sr_aligned["Action"].to_numpy()

    df["Trial"] = pd.to_numeric(df["Trial"], errors="coerce").ffill().bfill()
    df["Trial"] = df["Trial"].round().astype("Int32")

    _write_csv(df, out_csv)
    return df


def batch_join(
    merged_dir: str,
    self_report_dir: str,
    out_dir: str,
    merged_glob: str = "*_merged_64hz.csv",
    time_window_csv: str | None = None,
    max_snap_s: Optional[float] = None,
):
    os.makedirs(out_dir, exist_ok=True)

    merged_paths = sorted(glob.glob(os.path.join(merged_dir, merged_glob)))
    if not merged_paths:
        raise FileNotFoundError(f"No merged files found in {merged_dir} with glob {merged_glob}")

    for phys_path in merged_paths:
        base = os.path.basename(phys_path)
        subject_id = base.split("_")[0]

        candidates = [
            os.path.join(self_report_dir, f"{subject_id}.csv"),
            os.path.join(self_report_dir, f"{subject_id}_self_report.csv"),
            os.path.join(self_report_dir, f"{subject_id}_selfreport.csv"),
        ]
        sr_path = next((p for p in candidates if os.path.exists(p)), None)
        if sr_path is None:
            fallback = sorted(glob.glob(os.path.join(self_report_dir, f"*{subject_id}*.csv")))
            sr_path = fallback[0] if fallback else None

        if sr_path is None:
            print(f"[WARN] no self report for subject {subject_id}; skipping")
            continue

        out_path = os.path.join(out_dir, f"{subject_id}_merged_64hz_with_self_report.csv")
        print(f"[OK] {subject_id}: {phys_path} + {sr_path} -> {out_path}")

        join_self_report_to_physio(
            phys_path,
            sr_path,
            out_path,
            keep_action=False,
            subject_id=subject_id,
            time_window_csv=time_window_csv,
            max_snap_s=max_snap_s,
        )



def batch_join_subject_folders(
    merged_dir: str,
    subjects_root: str,
    out_dir: str,
    *,
    report_filename: str = "pain_data.csv",
    merged_glob: str = "*_merged_64hz.csv",
    out_name_template: str = "{subject_id}_merged_64hz_with_pain.csv",
    time_window_csv: Optional[str] = None,
    tz_local: str = "America/New_York",
    max_snap_s: Optional[float] = None,
) -> None:
    """
    Batch join for directory layouts where each subject folder contains the self-report.

    This matches the young cohort layout:
      subjects_root/101/BVP.csv, ... , pain_data.csv
      merged_dir/101_merged_64hz.csv
    """
    os.makedirs(out_dir, exist_ok=True)

    merged_paths = sorted(glob.glob(os.path.join(merged_dir, merged_glob)))
    if not merged_paths:
        raise FileNotFoundError(f"No merged files found in {merged_dir} with glob {merged_glob}")

    for phys_path in merged_paths:
        base = os.path.basename(phys_path)
        subject_id = base.split("_")[0]

        sr_path = os.path.join(subjects_root, subject_id, report_filename)
        if not os.path.exists(sr_path):
            # try case-insensitive fallback inside the subject folder
            subj_dir = os.path.join(subjects_root, subject_id)
            if os.path.isdir(subj_dir):
                files = {f.lower(): f for f in os.listdir(subj_dir)}
                key = report_filename.lower()
                if key in files:
                    sr_path = os.path.join(subj_dir, files[key])

        if not os.path.exists(sr_path):
            print(f"[WARN] Missing self report for subject {subject_id}: {sr_path}; skipping")
            continue

        out_path = os.path.join(out_dir, out_name_template.format(subject_id=subject_id))
        print(f"[OK] {subject_id}: {phys_path} + {sr_path} -> {out_path}")

        join_self_report_to_physio(
            phys_path,
            sr_path,
            out_path,
            keep_action=False,
            subject_id=subject_id,
            time_window_csv=time_window_csv,
            tz_local=tz_local,
            max_snap_s=max_snap_s,
        )
