import os
import glob
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional SciPy (needed for quadratic/cubic + bandpass). Linear still works without it.
try:
    from scipy.interpolate import interp1d
    from scipy.signal import butter, filtfilt
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# ----------------------------
# Config
# ----------------------------
@dataclass
class MergeConfig:
    # EDA/TEMP mapping onto BVP grid
    map_method: str = "snap"         # "snap" or "interp"
    map_interp_kind: str = "linear"  # for EDA/TEMP if map_method="interp"

    # HR derivation / resampling
    hr_target: str = "64hz"          # "64hz" or "1hz" (64hz is default as requested)
    hr_interp_kind: str = "cubic"    # "linear" / "quadratic" / "cubic"
    hr_min_bpm: float = 30.0
    hr_max_bpm: float = 220.0
    hr_mad_z: float = 5.0            # robust spike rejection on instantaneous HR

    # Bandpass on HR after resampling to 64 Hz (Nyquist = 32 Hz)
    # These defaults are conservative HRV-style dynamics (0.04–0.4 Hz).
    hr_bp_low_hz: float = 0.04
    hr_bp_high_hz: float = 0.40
    hr_bp_order: int = 2

    # Section logic across segments
    gap_threshold_s: float = 3.0     # default per your request
    fill_short_gaps: bool = True

    # Which channels to interpolate across segment gaps (BVP usually should NOT)
    gap_fill_channels: Tuple[str, ...] = ("eda", "temperature", "hr")

    # Grid properties
    fs_bvp: float = 64.0
    extend_grid_to_union: bool = True  # extend beyond BVP span to cover earliest/latest among channels in that segment


DT_NS_64 = int(round(1e9 / 64.0))  # 15_625_000 ns


# ----------------------------
# IO helpers
# ----------------------------
def read_channel_csv(subject_dir: str, subject_id: str, channel: str) -> Optional[pd.DataFrame]:
    """
    Finds and reads one CSV for channel: *_{channel}.csv in subject_dir.
    Expected columns: timestamp_ns, datetime_utc, segment, value (or peak).
    """
    pattern = os.path.join(subject_dir, f"{subject_id}_{channel}.csv")
    if not os.path.exists(pattern):
        # fallback: any *_channel.csv
        candidates = glob.glob(os.path.join(subject_dir, f"*_{channel}.csv"))
        if not candidates:
            return None
        pattern = sorted(candidates)[0]

    df = pd.read_csv(pattern)
    if "timestamp_ns" not in df.columns or "segment" not in df.columns:
        raise ValueError(f"{pattern} missing required columns (timestamp_ns, segment).")

    df["timestamp_ns"] = df["timestamp_ns"].astype(np.int64)
    df["segment"] = df["segment"].astype(np.int64)
    return df


def ensure_sorted_unique_times(t_ns: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sort by time and drop duplicate timestamps keeping the last."""
    order = np.argsort(t_ns, kind="mergesort")
    t = t_ns[order]
    vv = v[order]
    if t.size == 0:
        return t, vv
    # keep last occurrence
    _, idx_last = np.unique(t, return_index=False), None
    # implement keep-last by reversing unique
    t_rev = t[::-1]
    vv_rev = vv[::-1]
    t_u_rev, uidx_rev = np.unique(t_rev, return_index=True)
    keep_rev = uidx_rev
    keep = (t.size - 1 - keep_rev)
    keep.sort()
    return t[keep], vv[keep]


# ----------------------------
# Grid building
# ----------------------------
def build_bvp_native_grid(
    bvp_seg: pd.DataFrame,
    start_ns: Optional[int],
    end_ns: Optional[int],
    cfg: MergeConfig,
) -> np.ndarray:
    """
    Build per-segment 64Hz grid aligned to the segment's BVP phase.
    - Native grid anchor is the first BVP timestamp in that segment.
    - Optionally extends to [start_ns, end_ns] (union across channels).
    """
    if bvp_seg is None or bvp_seg.empty:
        raise ValueError("BVP segment is empty; cannot build native 64Hz grid.")

    bvp_ts = np.asarray(bvp_seg["timestamp_ns"], dtype=np.int64)
    bvp_ts = np.unique(bvp_ts)  # BVP should already be on-grid; keep unique
    bvp_ts.sort()
    anchor = int(bvp_ts[0])

    # Default range: BVP range; optionally extend to union across channels
    s = int(bvp_ts[0]) if start_ns is None else int(start_ns)
    e = int(bvp_ts[-1]) if end_ns is None else int(end_ns)

    if not cfg.extend_grid_to_union:
        s, e = int(bvp_ts[0]), int(bvp_ts[-1])

    # Align s,e to anchor phase so grid points fall on anchor + k*dt
    if s <= anchor:
        k_back = int(np.ceil((anchor - s) / DT_NS_64))
        s_aligned = anchor - k_back * DT_NS_64
    else:
        k_fwd = int(np.floor((s - anchor) / DT_NS_64))
        s_aligned = anchor + k_fwd * DT_NS_64

    if e >= anchor:
        k_end = int(np.floor((e - anchor) / DT_NS_64))
        e_aligned = anchor + k_end * DT_NS_64
    else:
        k_end = int(-np.ceil((anchor - e) / DT_NS_64))
        e_aligned = anchor + k_end * DT_NS_64

    if e_aligned < s_aligned:
        return np.array([], dtype=np.int64)

    n = int((e_aligned - s_aligned) // DT_NS_64) + 1
    grid = s_aligned + np.arange(n, dtype=np.int64) * DT_NS_64
    return grid


# ----------------------------
# Mapping EDA/TEMP to grid
# ----------------------------
def nearest_neighbor_on_grid(t_s: np.ndarray, v: np.ndarray, t_g: np.ndarray) -> np.ndarray:
    """
    For each grid time, take value of nearest sample time.
    """
    if t_s.size == 0:
        return np.full(t_g.shape, np.nan, dtype=np.float64)

    # assume t_s sorted
    idx = np.searchsorted(t_s, t_g, side="left")
    idx0 = np.clip(idx - 1, 0, t_s.size - 1)
    idx1 = np.clip(idx, 0, t_s.size - 1)

    d0 = np.abs(t_g - t_s[idx0])
    d1 = np.abs(t_g - t_s[idx1])
    pick = np.where(d1 < d0, idx1, idx0)
    out = v[pick].astype(np.float64)
    return out


def interp_on_grid(t_s: np.ndarray, v: np.ndarray, t_g: np.ndarray, kind: str) -> np.ndarray:
    """
    Interpolate samples to grid times.
    Supports linear always. Quadratic/cubic requires SciPy.
    """
    if t_s.size < 2:
        return np.full(t_g.shape, np.nan, dtype=np.float64)

    # linear without SciPy
    if (not SCIPY_OK) or kind == "linear":
        # numpy.interp does not support nan in v well; mask finite
        m = np.isfinite(v)
        if m.sum() < 2:
            return np.full(t_g.shape, np.nan, dtype=np.float64)
        return np.interp(t_g.astype(np.float64), t_s[m].astype(np.float64), v[m].astype(np.float64),
                         left=np.nan, right=np.nan).astype(np.float64)

    # SciPy for quadratic/cubic
    f = interp1d(
        t_s.astype(np.float64),
        v.astype(np.float64),
        kind=kind,
        bounds_error=False,
        fill_value=np.nan,
        assume_sorted=True,
    )
    return f(t_g.astype(np.float64)).astype(np.float64)


def map_scalar_channel_to_grid(
    df_seg: Optional[pd.DataFrame],
    grid_ts: np.ndarray,
    method: str,
    interp_kind: str,
) -> np.ndarray:
    if df_seg is None or df_seg.empty:
        return np.full(grid_ts.shape, np.nan, dtype=np.float64)
    t = np.asarray(df_seg["timestamp_ns"], dtype=np.int64)
    v = np.asarray(df_seg["value"], dtype=np.float64)
    t, v = ensure_sorted_unique_times(t, v)

    if method == "snap":
        return nearest_neighbor_on_grid(t, v, grid_ts)
    elif method == "interp":
        return interp_on_grid(t, v, grid_ts, kind=interp_kind)
    else:
        raise ValueError(f"Unknown mapping method: {method}")


# ----------------------------
# Peaks -> HR
# ----------------------------
def robust_filter_hr(t_ns: np.ndarray, hr: np.ndarray, cfg: MergeConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Outlier filter on instantaneous HR:
      - hard bpm bounds
      - MAD-based spike rejection
    """
    m = np.isfinite(hr)
    m &= (hr >= cfg.hr_min_bpm) & (hr <= cfg.hr_max_bpm)
    t_ns = t_ns[m]
    hr = hr[m]
    if hr.size < 5:
        return t_ns, hr

    med = np.median(hr)
    mad = np.median(np.abs(hr - med))
    if mad <= 1e-9:
        return t_ns, hr
    z = np.abs(hr - med) / (1.4826 * mad)
    keep = z <= cfg.hr_mad_z
    return t_ns[keep], hr[keep]


def bandpass_filter_64hz(x: np.ndarray, cfg: MergeConfig) -> np.ndarray:
    """
    Butterworth bandpass on 64Hz series (Nyquist=32Hz).
    Applies filtering on finite segments; leaves NaNs in place.
    """
    if not SCIPY_OK:
        return x  # cannot filtfilt without SciPy

    fs = 64.0
    low = cfg.hr_bp_low_hz
    high = cfg.hr_bp_high_hz
    if low <= 0 and high <= 0:
        return x
    if high >= fs / 2:
        high = (fs / 2) * 0.99

    # If user sets low=0 -> lowpass; high=0 -> highpass
    nyq = fs / 2
    if low <= 0:
        btype = "lowpass"
        Wn = high / nyq
    elif high <= 0:
        btype = "highpass"
        Wn = low / nyq
    else:
        btype = "bandpass"
        Wn = [low / nyq, high / nyq]

    b, a = butter(cfg.hr_bp_order, Wn, btype=btype)

    y = x.copy().astype(np.float64)
    m = np.isfinite(y)
    if m.sum() < 10:
        return y

    # filter contiguous finite runs
    idx = np.where(m)[0]
    runs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    for r in runs:
        if r.size < max(10, 3 * (cfg.hr_bp_order + 1)):
            continue
        seg = y[r]
        y[r] = filtfilt(b, a, seg)
    return y


def derive_hr_on_grid(
    peaks_seg: Optional[pd.DataFrame],
    grid_ts: np.ndarray,
    cfg: MergeConfig,
) -> np.ndarray:
    """
    peaks -> IBI -> HR:
      1) instantaneous HR at beat times
      2) outlier filter
      3) resample to 1Hz grid (cubic default)
      4) upsample/interp to 64Hz BVP grid (default)
      5) bandpass filter at fs=64 (Nyquist=32)
    """
    if peaks_seg is None or peaks_seg.empty:
        return np.full(grid_ts.shape, np.nan, dtype=np.float64)

    # peaks timestamps
    t_peaks = np.asarray(peaks_seg["timestamp_ns"], dtype=np.int64)
    t_peaks = np.unique(t_peaks)
    t_peaks.sort()
    if t_peaks.size < 3:
        return np.full(grid_ts.shape, np.nan, dtype=np.float64)

    ibi_s = np.diff(t_peaks).astype(np.float64) / 1e9
    # associate HR with the later peak time
    t_hr_inst = t_peaks[1:]
    hr_inst = 60.0 / np.maximum(ibi_s, 1e-9)

    t_hr_inst, hr_inst = robust_filter_hr(t_hr_inst, hr_inst, cfg)
    if hr_inst.size < 3:
        return np.full(grid_ts.shape, np.nan, dtype=np.float64)

    # Build a 1Hz grid within the segment span (in ns)
    seg_start = int(grid_ts[0])
    seg_end = int(grid_ts[-1])
    start_s = int(np.ceil(seg_start / 1e9))
    end_s = int(np.floor(seg_end / 1e9))
    if end_s <= start_s:
        return np.full(grid_ts.shape, np.nan, dtype=np.float64)

    t1_ns = (np.arange(start_s, end_s + 1, dtype=np.int64) * 1_000_000_000)

    # Interpolate instantaneous HR onto 1Hz grid
    kind = cfg.hr_interp_kind
    # If SciPy not available, quadratic/cubic will fall back to linear inside interp_on_grid
    hr_1 = interp_on_grid(t_hr_inst, hr_inst, t1_ns, kind=kind)

    if cfg.hr_target == "1hz":
        # If you ever want a 1Hz dataset, return HR broadcast to 64Hz grid by nearest-neighbor.
        # (Still keeps your final merged DF at 64Hz.)
        hr_64 = nearest_neighbor_on_grid(t1_ns.astype(np.int64), hr_1.astype(np.float64), grid_ts)
        hr_64 = bandpass_filter_64hz(hr_64, cfg)
        return hr_64

    # Default: interpolate 1Hz HR to 64Hz grid
    hr_64 = interp_on_grid(t1_ns.astype(np.int64), hr_1.astype(np.float64), grid_ts, kind=kind)
    hr_64 = bandpass_filter_64hz(hr_64, cfg)
    return hr_64


# ----------------------------
# Per-segment merge
# ----------------------------
def segment_union_bounds(
    bvp_seg: Optional[pd.DataFrame],
    eda_seg: Optional[pd.DataFrame],
    temp_seg: Optional[pd.DataFrame],
    peaks_seg: Optional[pd.DataFrame],
) -> Tuple[Optional[int], Optional[int]]:
    """
    Return earliest and latest timestamps among available channels for this segment.
    """
    tmins = []
    tmaxs = []
    for df, col in [(bvp_seg, "timestamp_ns"), (eda_seg, "timestamp_ns"), (temp_seg, "timestamp_ns"), (peaks_seg, "timestamp_ns")]:
        if df is not None and not df.empty:
            tmins.append(int(df[col].min()))
            tmaxs.append(int(df[col].max()))
    if not tmins:
        return None, None
    return min(tmins), max(tmaxs)


def build_per_segment_df(
    seg_id: int,
    bvp_seg: pd.DataFrame,
    eda_seg: Optional[pd.DataFrame],
    temp_seg: Optional[pd.DataFrame],
    peaks_seg: Optional[pd.DataFrame],
    cfg: MergeConfig,
) -> pd.DataFrame:
    # compute union bounds (optional grid extension)
    start_ns, end_ns = segment_union_bounds(bvp_seg, eda_seg, temp_seg, peaks_seg)
    grid_ts = build_bvp_native_grid(bvp_seg, start_ns, end_ns, cfg)
    if grid_ts.size == 0:
        return pd.DataFrame()

    # BVP on grid: exact placement; if slight mismatch, fall back to nearest
    bvp_t = np.asarray(bvp_seg["timestamp_ns"], dtype=np.int64)
    bvp_v = np.asarray(bvp_seg["value"], dtype=np.float64)
    bvp_t, bvp_v = ensure_sorted_unique_times(bvp_t, bvp_v)

    # Try direct reindex-like mapping (fast path)
    bvp_map = np.full(grid_ts.shape, np.nan, dtype=np.float64)
    pos = np.searchsorted(grid_ts, bvp_t)
    ok = (pos >= 0) & (pos < grid_ts.size) & (grid_ts[pos] == bvp_t)
    bvp_map[pos[ok]] = bvp_v[ok]
    # if too sparse (meaning timestamps didn’t match), use nearest neighbor
    if np.isfinite(bvp_map).sum() < 0.9 * min(grid_ts.size, bvp_t.size):
        bvp_map = nearest_neighbor_on_grid(bvp_t, bvp_v, grid_ts)

    eda_map = map_scalar_channel_to_grid(eda_seg, grid_ts, method=cfg.map_method, interp_kind=cfg.map_interp_kind)
    temp_map = map_scalar_channel_to_grid(temp_seg, grid_ts, method=cfg.map_method, interp_kind=cfg.map_interp_kind)

    hr_map = derive_hr_on_grid(peaks_seg, grid_ts, cfg)

    out = pd.DataFrame({
        "timestamp_ns": grid_ts,
        "datetime_utc": pd.to_datetime(grid_ts, unit="ns", utc=True),
        "segment": seg_id,
        "bvp": bvp_map,
        "eda": eda_map,
        "temperature": temp_map,
        "hr": hr_map,
    })
    return out


# ----------------------------
# Across-segment: sectioning + optional gap fill
# ----------------------------
def compute_gap_s(prev_end_ns: int, next_start_ns: int) -> float:
    """
    Gap in seconds between two 64Hz grids accounting for one dt step.
    If next starts exactly one dt after prev ends, gap=0.
    """
    raw = (next_start_ns - prev_end_ns - DT_NS_64) / 1e9
    return float(max(0.0, raw))


def make_gap_rows(prev_end_ns: int, next_start_ns: int) -> np.ndarray:
    """
    Create missing 64Hz timestamps continuing prev grid phase:
      (prev_end + dt) ... (next_start - dt)
    """
    if next_start_ns <= prev_end_ns + DT_NS_64:
        return np.array([], dtype=np.int64)
    n = int((next_start_ns - prev_end_ns) // DT_NS_64) - 1
    if n <= 0:
        return np.array([], dtype=np.int64)
    return prev_end_ns + np.arange(1, n + 1, dtype=np.int64) * DT_NS_64


def assign_sections_and_merge(segments: List[pd.DataFrame], cfg: MergeConfig) -> pd.DataFrame:
    """
    - sort segments by start time
    - assign section ids based on gap_threshold_s
    - for short gaps: insert gap rows (NaNs) and interpolate specified channels across them
    - do NOT force a master macro-grid across segments; keeps each segment's timestamps as-is
    """
    seg_info = []
    for df in segments:
        if df is None or df.empty:
            continue
        seg_id = int(df["segment"].iloc[0])
        seg_info.append((int(df["timestamp_ns"].iloc[0]), int(df["timestamp_ns"].iloc[-1]), seg_id, df))

    seg_info.sort(key=lambda x: x[0])

    out_parts = []
    section_id = 0

    for i, (s0, e0, seg_id, df) in enumerate(seg_info):
        if i == 0:
            df2 = df.copy()
            df2["section"] = section_id
            out_parts.append(df2)
            continue

        prev_df = out_parts[-1]
        prev_end = int(prev_df["timestamp_ns"].iloc[-1])
        gap_s = compute_gap_s(prev_end, s0)

        # overlap handling: if overlap, we keep later segment data on duplicates (drop later duplicates after concat)
        same_section = gap_s <= cfg.gap_threshold_s

        if not same_section:
            section_id += 1
            df2 = df.copy()
            df2["section"] = section_id
            out_parts.append(df2)
            continue

        # same section: optionally fill small gaps by adding rows continuing prev grid
        if cfg.fill_short_gaps:
            gap_ts = make_gap_rows(prev_end, s0)
            if gap_ts.size > 0:
                gap_block = pd.DataFrame({
                    "timestamp_ns": gap_ts,
                    "datetime_utc": pd.to_datetime(gap_ts, unit="ns", utc=True),
                    "segment": -1,              # gap rows are synthetic
                    "bvp": np.nan,
                    "eda": np.nan,
                    "temperature": np.nan,
                    "hr": np.nan,
                    "section": section_id,
                })
                out_parts.append(gap_block)

        df2 = df.copy()
        df2["section"] = section_id
        out_parts.append(df2)

    merged = pd.concat(out_parts, ignore_index=True)
    merged = merged.sort_values("timestamp_ns", kind="mergesort")

    # Resolve duplicates (overlaps) by keeping last
    merged = merged.drop_duplicates(subset=["timestamp_ns"], keep="last").reset_index(drop=True)

    # Interpolate across inserted gap rows *within each section* for selected channels only
    if cfg.fill_short_gaps and cfg.gap_fill_channels:
        merged = merged.set_index("timestamp_ns")
        for sec, g in merged.groupby("section", sort=False):
            # interpolate only inside; do not extrapolate ends
            for ch in cfg.gap_fill_channels:
                if ch in g.columns:
                    merged.loc[g.index, ch] = g[ch].interpolate(method="index", limit_area="inside")
        merged = merged.reset_index()

    return merged


# ----------------------------
# End-to-end per subject
# ----------------------------
def process_subject(subject_dir: str, subject_id: str, out_path: str, cfg: MergeConfig) -> pd.DataFrame:
    bvp = read_channel_csv(subject_dir, subject_id, "bvp")
    eda = read_channel_csv(subject_dir, subject_id, "eda")
    temp = read_channel_csv(subject_dir, subject_id, "temperature")
    peaks = read_channel_csv(subject_dir, subject_id, "systolicPeaks")

    if bvp is None or bvp.empty:
        raise ValueError(f"Missing/empty BVP for subject {subject_id} in {subject_dir}")

    # group by segment
    seg_ids = sorted(bvp["segment"].unique().tolist())

    segments_merged = []
    for seg_id in seg_ids:
        bvp_seg = bvp[bvp["segment"] == seg_id]
        eda_seg = eda[eda["segment"] == seg_id] if eda is not None else None
        temp_seg = temp[temp["segment"] == seg_id] if temp is not None else None
        peaks_seg = peaks[peaks["segment"] == seg_id] if peaks is not None else None

        df_seg = build_per_segment_df(seg_id, bvp_seg, eda_seg, temp_seg, peaks_seg, cfg)
        if not df_seg.empty:
            segments_merged.append(df_seg)

    merged = assign_sections_and_merge(segments_merged, cfg)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    merged.to_csv(out_path, index=False)
    return merged


def process_all_subjects(output_root: str, combined_out_dir: str, cfg: MergeConfig):
    """
    Expects:
      output_root/
        1/  (subject folder)
          1_bvp.csv, 1_eda.csv, 1_temperature.csv, 1_systolicPeaks.csv
        2/
          2_bvp.csv, ...
      Writes:
        combined_out_dir/1_merged_64hz.csv, ...
    """
    subj_dirs = [d for d in glob.glob(os.path.join(output_root, "*")) if os.path.isdir(d)]
    subj_dirs.sort()

    for sd in subj_dirs:
        subject_id = os.path.basename(sd)
        out_path = os.path.join(combined_out_dir, f"{subject_id}_merged_64hz.csv")
        print(f"[SUBJECT {subject_id}] -> {out_path}")
        process_subject(sd, subject_id, out_path, cfg)


if __name__ == "__main__":
    # Example usage:
    #   python merge_physio.py --output_root ./output --combined_out ./processed_data/combined
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True, help="Root dir containing subject folders (1..7)")
    parser.add_argument("--combined_out", required=True, help="Dir to write merged outputs")
    parser.add_argument("--gap_threshold_s", type=float, default=3.0)
    parser.add_argument("--map_method", choices=["snap", "interp"], default="snap")
    parser.add_argument("--map_interp_kind", choices=["linear", "quadratic", "cubic"], default="linear")
    parser.add_argument("--hr_interp_kind", choices=["linear", "quadratic", "cubic"], default="cubic")
    args = parser.parse_args()

    cfg = MergeConfig(
        gap_threshold_s=args.gap_threshold_s,
        map_method=args.map_method,
        map_interp_kind=args.map_interp_kind,
        hr_interp_kind=args.hr_interp_kind,
    )

    if (cfg.map_method == "interp" or cfg.hr_interp_kind in ("quadratic", "cubic")) and not SCIPY_OK:
        print("[WARN] SciPy not available: quadratic/cubic will fall back to linear; bandpass disabled.")

    process_all_subjects(args.output_root, args.combined_out, cfg)
