"""src.merge

Utilities for aligning (snapping) sparse self-report labels to regularly sampled
physiological time series.

The core operation is nearest-neighbor snapping of label timestamps onto an
existing `timestamp_ns` grid (typically produced by `src.preprocessing`).

Supported self-report formats
-----------------------------
1) Headered CSV (older cohort convention)
   Required columns:
     - timestamp   : local time (America/New_York by default), e.g. "2025-03-12 09:22:46.505"
     - PainLevel   : numeric
   Optional columns:
     - Trial, Action

2) Headerless CSV (young cohort convention: pain_data.csv)
   No header; rows are:
     local_timestamp, pain_level, subject_id
   Example:
     2024-07-08 13:13:39.609319,5,101

All timestamps are interpreted as local time (configurable) and converted to UTC.
"""

from __future__ import annotations

import glob
import os
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd


def _nearest_index(target_ns: np.ndarray, grid_ns: np.ndarray) -> np.ndarray:
    """Return the nearest index in `grid_ns` for each value in `target_ns`."""
    idx = np.searchsorted(grid_ns, target_ns, side="left")
    idx0 = np.clip(idx - 1, 0, grid_ns.size - 1)
    idx1 = np.clip(idx, 0, grid_ns.size - 1)

    d0 = np.abs(target_ns - grid_ns[idx0])
    d1 = np.abs(target_ns - grid_ns[idx1])
    return np.where(d1 < d0, idx1, idx0)


def _nearest_index_with_distance(target_ns: np.ndarray, grid_ns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (nearest_idx, distance_ns) for each `target_ns`."""
    idx = _nearest_index(target_ns, grid_ns)
    dist = np.abs(target_ns - grid_ns[idx])
    return idx, dist


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
    """
    Load a self-report file and standardize it for time alignment.

    Returns a DataFrame with at least:
      - timestamp_utc (tz-aware)
      - timestamp_ns  (int64 ns since epoch)
      - PainLevel     (float, may contain NaN)

    Trial/Action columns are preserved if present.
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
    sr["timestamp_ns"] = sr["timestamp_utc"].astype("int64")

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
    Load per-subject experiment windows used for optional filtering.

    Expected CSV format:
      subject,start,end
      (001,YYYY-MM-DD HH:MM:SS.***,YYYY-MM-DD HH:MM:SS.***)

    Parsing is done with `pd.to_datetime(errors="coerce")` and interpreted as
    naive local time in `tz_local`.
    """
    w = pd.read_csv(time_window_csv, dtype={"subject": str})

    required = {"subject", "start", "end"}
    if not required.issubset(set(w.columns)):
        return pd.DataFrame(columns=["subject_norm", "start_utc", "end_utc"])

    start_local = pd.to_datetime(w["start"], errors="coerce")
    end_local = pd.to_datetime(w["end"], errors="coerce")

    start_utc = (
        start_local.dt.tz_localize(tz_local, ambiguous="infer", nonexistent="shift_forward")
        .dt.tz_convert("UTC")
    )
    end_utc = (
        end_local.dt.tz_localize(tz_local, ambiguous="infer", nonexistent="shift_forward")
        .dt.tz_convert("UTC")
    )

    out = pd.DataFrame(
        {"subject_norm": w["subject"].map(_norm_subject_id), "start_utc": start_utc, "end_utc": end_utc}
    )

    out = out[out["start_utc"].notna() & out["end_utc"].notna()].copy()
    if out.empty:
        return out

    # enforce start <= end
    s = out["start_utc"].to_numpy()
    e = out["end_utc"].to_numpy()
    out["start_utc"] = np.minimum(s, e)
    out["end_utc"] = np.maximum(s, e)

    return out[["subject_norm", "start_utc", "end_utc"]]


def join_self_report_to_physio(
    physio_csv: str,
    self_report_csv: str,
    out_csv: str,
    keep_action: bool = False,
    subject_id: Optional[str] = None,
    time_window_csv: Optional[str] = None,
    tz_local: str = "America/New_York",
    max_snap_s: Optional[float] = None,
) -> pd.DataFrame:
    """
    Snap self-report labels onto a physiological time series.

    Parameters
    ----------
    physio_csv:
        Path to a merged physiological CSV that contains `timestamp_ns`.
    self_report_csv:
        Path to a self-report CSV. See module docstring for supported formats.
    out_csv:
        Output CSV path.
    keep_action:
        If True and Action exists in the self-report, include it in the output.
    subject_id, time_window_csv:
        Optional experiment-window filtering. If provided, physiological rows outside
        the subject windows are dropped before snapping; self-report rows are also filtered.
    tz_local:
        Local timezone for interpreting self-report timestamps when they are naive.
    max_snap_s:
        If provided, ignore self-report rows whose nearest grid point is farther than this threshold.

    Returns
    -------
    pd.DataFrame
        The aligned DataFrame written to `out_csv`.
    """
    df = pd.read_csv(physio_csv)
    if "timestamp_ns" not in df.columns:
        raise ValueError(f"{physio_csv} is missing required column 'timestamp_ns'")

    df["timestamp_ns"] = df["timestamp_ns"].astype("int64")
    df = df.sort_values("timestamp_ns", kind="mergesort").reset_index(drop=True)

    # Build tz-aware UTC timestamps for window filtering (pandas-native comparisons)
    df_ts_utc = pd.to_datetime(df["timestamp_ns"], unit="ns", utc=True)

    win_sub = None
    if time_window_csv and subject_id is not None:
        win = load_subject_time_windows(time_window_csv, tz_local=tz_local)
        if not win.empty:
            sid_norm = _norm_subject_id(subject_id)
            win_sub = win[win["subject_norm"] == sid_norm].copy()
            if not win_sub.empty:
                mask = np.zeros(len(df), dtype=bool)
                for s, e in win_sub[["start_utc", "end_utc"]].itertuples(index=False, name=None):
                    mask |= (df_ts_utc >= s) & (df_ts_utc <= e)
                df = df.loc[mask].copy().reset_index(drop=True)
                df_ts_utc = df_ts_utc.loc[mask].reset_index(drop=True)

    # Ensure expected columns exist even if empty after filtering.
    if "PainLevel" not in df.columns:
        df["PainLevel"] = np.nan
    if "Trial" not in df.columns:
        df["Trial"] = pd.Series(dtype="Int32")
    if keep_action and "Action" not in df.columns:
        df["Action"] = np.nan

    if df.empty:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df.to_csv(out_csv, index=False)
        return df

    grid_ns = df["timestamp_ns"].to_numpy(dtype=np.int64)

    sr = load_self_report(self_report_csv, tz_local=tz_local)

    # Window filter for self-report as well (uses timestamp_utc).
    if win_sub is not None and not win_sub.empty and not sr.empty:
        sr_mask = np.zeros(len(sr), dtype=bool)
        for s, e in win_sub[["start_utc", "end_utc"]].itertuples(index=False, name=None):
            sr_mask |= (sr["timestamp_utc"] >= s) & (sr["timestamp_utc"] <= e)
        sr = sr.loc[sr_mask].copy()

    if sr.empty:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df.to_csv(out_csv, index=False)
        return df

    target_ns = sr["timestamp_ns"].to_numpy(dtype=np.int64)

    if max_snap_s is None:
        idx = _nearest_index(target_ns, grid_ns)
        keep_mask = None
    else:
        idx, dist_ns = _nearest_index_with_distance(target_ns, grid_ns)
        keep_mask = (dist_ns.astype(np.float64) / 1e9) <= float(max_snap_s)
        idx = idx[keep_mask]

    if idx.size == 0:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df.to_csv(out_csv, index=False)
        return df

    if keep_mask is not None:
        sr2 = sr.iloc[np.where(keep_mask)[0]].copy()
    else:
        sr2 = sr

    df.loc[idx, "PainLevel"] = sr2["PainLevel"].to_numpy(dtype=np.float64)

    if "Trial" in sr2.columns:
        df.loc[idx, "Trial"] = sr2["Trial"].to_numpy()
        df["Trial"] = pd.to_numeric(df["Trial"], errors="coerce").ffill().bfill().round().astype("Int32")

    if keep_action and "Action" in sr2.columns:
        df.loc[idx, "Action"] = sr2["Action"].to_numpy()

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def batch_join(
    merged_dir: str,
    self_report_dir: str,
    out_dir: str,
    merged_glob: str = "*_merged_64hz.csv",
    time_window_csv: Optional[str] = None,
    tz_local: str = "America/New_York",
    max_snap_s: Optional[float] = None,
) -> None:
    """
    Batch join for directory layouts where self-reports are stored in a separate folder.

    This is used by the older cohort pipeline:
      merged_dir/001_merged_64hz.csv
      self_report_dir/001.csv
    """
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
            print(f"[WARN] No self report for subject {subject_id}; skipping")
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
            tz_local=tz_local,
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
