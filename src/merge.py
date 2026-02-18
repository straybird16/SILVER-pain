"""
    Functions that merge differnt channels in original files or extraction from original files to a single .csv file per subject
    
"""


import os
import glob
import numpy as np
import pandas as pd
from typing import Iterable, Optional, Tuple


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


def _to_utc_datetime(values: pd.Series, tz_local: str = "America/New_York") -> pd.Series:
    """Convert a timestamp series to tz-aware UTC datetimes using pandas-native conversion.

    Supported inputs:
      - datetime-like strings (interpreted as local time in tz_local unless they carry tz info)
      - numeric epoch timestamps (unit inferred via _infer_epoch_unit; interpreted as UTC)

    Returns a tz-aware datetime Series in UTC.
    """
    if np.issubdtype(values.dtype, np.datetime64):
        return pd.to_datetime(values, errors="coerce", utc=True)

    # Detect numeric epoch-like input (including strings of digits).
    v_num = pd.to_numeric(values, errors="coerce")
    if v_num.notna().sum() > 0 and v_num.notna().mean() > 0.8:
        unit = _infer_epoch_unit(v_num)
        return pd.to_datetime(v_num, unit=unit, utc=True, errors="coerce")

    # Otherwise parse as local wall-clock time and convert to UTC.
    t_local = pd.to_datetime(values, errors="coerce")
    return (t_local
            .dt.tz_localize(tz_local, ambiguous="infer", nonexistent="shift_forward")
            .dt.tz_convert("UTC"))



def _nearest_index(target_ns: np.ndarray, grid_ns: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(grid_ns, target_ns, side="left")
    idx0 = np.clip(idx - 1, 0, grid_ns.size - 1)
    idx1 = np.clip(idx,     0, grid_ns.size - 1)

    d0 = np.abs(target_ns - grid_ns[idx0])
    d1 = np.abs(target_ns - grid_ns[idx1])
    return np.where(d1 < d0, idx1, idx0)

def _coerce_timestamp_utc(ts: pd.Series, tz_local: str) -> pd.Series:
    """
    Parse timestamps and return a tz-aware UTC pandas Series.

    If parsed timestamps are tz-aware, they are converted to UTC.
    If parsed timestamps are naive, they are localized to `tz_local` then converted to UTC.
    """
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

    Output columns:
      - timestamp_utc: tz-aware UTC datetime
      - other original columns are preserved
    """
    """ sr = pd.read_csv(self_report_csv)

    if "timestamp" not in sr.columns:
        raise ValueError(f"{self_report_csv} missing required column: 'timestamp'")

    sr["timestamp_utc"] = _to_utc_datetime(sr["timestamp"])

    if "Trial" in sr.columns:
        trial_num = pd.to_numeric(sr["Trial"], errors="coerce")
        sr["Trial"] = trial_num.ffill().bfill().astype(int) """
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
    Reads a CSV like:
      subject,start,end
      001,YYYY-MM-DD HH:MM:SS.***,YYYY-MM-DD HH:MM:SS.***
    or:
      subject,start,end
      101,MM/DD/YYYY  HH:MM:SS PM,MM/DD/YYYY  HH:MM:SS PM 
    (BY DEFAULT MONTH FIRST)

    Behavior:
      - Parse start/end with pd.to_datetime(errors="coerce") (handles both formats in your examples)
      - Interpret as tz_local (naive local times) then convert to UTC
      - Drop rows where start/end are invalid datetimes
      - Ensure start <= end
      
    Returns tz-aware UTC datetimes.
    Output columns: subject_norm, start_utc, end_utc
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
    df = pd.read_csv(physio_csv)
    if "timestamp_ns" not in df.columns:
        raise ValueError(f"{physio_csv} missing timestamp_ns")

    # Parse physio timestamps as tz-aware UTC using pandas-native conversion.
    # The 'timestamp_ns' column is expected to be an epoch timestamp (ns/us/ms/s),
    # but we infer the unit to avoid silent unit-mismatch bugs.
    ts_num = pd.to_numeric(df["timestamp_ns"], errors="coerce")
    if ts_num.notna().sum() > 0 and ts_num.notna().mean() > 0.8:
        unit = _infer_epoch_unit(ts_num)
        df_ts_utc = pd.to_datetime(ts_num, unit=unit, utc=True, errors="coerce")
    else:
        # Fallback: treat as datetime-like strings. We assume these are already UTC.
        df_ts_utc = pd.to_datetime(df["timestamp_ns"], utc=True, errors="coerce")

    # Sort by time for stable downstream operations (window filtering + nearest snapping).
    df["_ts_utc"] = df_ts_utc
    df = df.sort_values("_ts_utc", kind="mergesort").reset_index(drop=True)
    df_ts_utc = df.pop("_ts_utc")

    win_sub = None
    if time_window_csv and subject_id is not None:
        win = load_subject_time_windows(time_window_csv)
        if not win.empty:
            sid_norm = _norm_subject_id(subject_id)
            win_sub = win[win["subject_norm"] == sid_norm].copy()
            if not win_sub.empty:
                mask = np.zeros(len(df), dtype=bool)
                for s, e in win_sub[["start_utc", "end_utc"]].itertuples(index=False, name=None):
                    mask |= (df_ts_utc >= s) & (df_ts_utc <= e)
                df = df.loc[mask].copy().reset_index(drop=True)
                df_ts_utc = df_ts_utc.loc[mask].reset_index(drop=True)

    # If filtering removed everything, still write an empty file with headers.
    if df.empty:
        if "PainLevel" not in df.columns:
            df["PainLevel"] = np.nan
        if "Trial" not in df.columns:
            df["Trial"] = pd.Series(dtype="Int32")
        if keep_action and "Action" not in df.columns:
            df["Action"] = np.nan
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df.to_csv(out_csv, index=False)
        return df

    sr = load_self_report(self_report_csv, tz_local)

    # If we filtered physio by windows, also filter SR by the same windows (using timestamp_utc)
    if win_sub is not None and not win_sub.empty and not sr.empty:
        sr_mask = np.zeros(len(sr), dtype=bool)
        for s, e in win_sub[["start_utc", "end_utc"]].itertuples(index=False, name=None):
            sr_mask |= (sr["timestamp_utc"] >= s) & (sr["timestamp_utc"] <= e)
        sr = sr.loc[sr_mask].copy()

    # Drop invalid timestamps before alignment.
    if not sr.empty:
        sr = sr[sr["timestamp_utc"].notna()].copy()

    if sr.empty:
        if "PainLevel" not in df.columns:
            df["PainLevel"] = np.nan
        if "Trial" not in df.columns:
            df["Trial"] = pd.Series(dtype="Int32")
        if keep_action and "Action" not in df.columns:
            df["Action"] = np.nan
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df.to_csv(out_csv, index=False)
        return df

    # Snap each self-report row to the nearest physio timestamp using pandas-native datetime alignment.
    # This avoids any dependence on integer epoch units (ns/us/ms) and prevents unit mismatch.
    grid = pd.DatetimeIndex(df_ts_utc)
    targets = pd.DatetimeIndex(sr["timestamp_utc"])
    if max_snap_s is not None: max_snap_s = pd.Timedelta(seconds=max_snap_s) #type:ignore
    nearest_idx = grid.get_indexer(targets, method="nearest", tolerance=max_snap_s)
    valid = nearest_idx >= 0
    nearest_idx = nearest_idx[valid]
    sr_aligned = sr.iloc[np.flatnonzero(valid)].copy()

    if "PainLevel" not in df.columns:
        df["PainLevel"] = np.nan
    if "Trial" not in df.columns:
        df["Trial"] = np.nan
    if keep_action and "Action" not in df.columns:
        df["Action"] = np.nan

    df.loc[nearest_idx, "PainLevel"] = sr_aligned["PainLevel"].to_numpy()
    if "Trial" in sr_aligned.columns:
        df.loc[nearest_idx, "Trial"] = sr_aligned["Trial"].to_numpy()
    if keep_action and "Action" in sr_aligned.columns:
        df.loc[nearest_idx, "Action"] = sr_aligned["Action"].to_numpy()

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    df["Trial"] = pd.to_numeric(df["Trial"], errors="coerce").ffill().bfill()
    df["Trial"] = df["Trial"].round().astype("Int32")

    df.to_csv(out_csv, index=False)
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
