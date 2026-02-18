"""src.preprocessing

Preprocessing and time-alignment utilities for SILVER-Pain physiological signals.

This module provides:
- Per-segment and per-session channel alignment onto a BVP-native 64 Hz grid
  (BVP, EDA, temperature, and heart rate derived from peaks or mapped from HR.csv).
- Gap-aware stitching of consecutive segments into sections.
- Optional artifact correction for RR/IBI series and HR resampling/smoothing.
- Convenience entry points for processing subject folders into per-subject merged CSVs.

Timestamps
----------
All merged outputs use `timestamp_ns` (int64, UTC nanoseconds since epoch) as the
primary time index and include `datetime_utc` for readability.
"""

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
    map_snap_kind: str = "one_to_one"   # "one_to_one" or "per_grid"

    # HR derivation / resampling
    hr_target: str = "1hz"          # "64hz" or "1hz" (64hz is default)
    hr_interp_kind: str = "cubic"    # "linear" / "quadratic" / "cubic"
    hr_min_bpm: float = 30.0
    hr_max_bpm: float = 220.0
    hr_mad_z: float = 3.0            # robust spike rejection on instantaneous HR
    
    # HR from HR.csv (Empatica)
    hr_map_method: str = "interp"          # "snap" or "interp"
    hr_map_interp_kind: str = "cubic"      # "linear" / "quadratic" / "cubic"
    
    # Pain/self-report time zone (pain_data.csv uses local ET)
    pain_tz: str = "America/New_York"
    pain_max_snap_s: Optional[float] = 1          # None means no max; otherwise ignore if farther than this

    # Default to lowpass on HR after resampling to 64 Hz (Nyquist = 32 Hz)
    hr_bp_low_hz: float = 0.0
    hr_bp_high_hz: float = 4.0
    hr_bp_order: int = 2

    # Section logic across segments
    gap_threshold_s: float = 3.0     # default
    fill_short_gaps: bool = True

    # Which channels to interpolate across segment gaps (BVP usually should NOT)
    gap_fill_channels: Tuple[str, ...] = ("eda", "temperature", "hr")

    # Grid properties
    fs_bvp: float = 64.0
    extend_grid_to_union: bool = True  # extend beyond BVP span to cover earliest/latest among channels in that segment
    
    # -------------- RR/IBI artifact correction pipeline --------------

    # Step A: robust local outlier marking on RR
    enable_rr_ratio_median: bool = True
    rr_ratio_win_beats: int = 11
    rr_ratio_thr: float = 0.25        # flag if |RR-med|/med > thr

    enable_rr_hampel: bool = False
    rr_hampel_win_beats: int = 11
    rr_hampel_k: float = 3.0          # Hampel/MAD z threshold

    # Step B: Kubios "Threshold" correction (optional)
    enable_kubios_threshold: bool = False
    kubios_threshold_level: str = "medium"      # very_low/low/medium/strong/very_strong
    kubios_threshold_sec_60bpm: Optional[float] = None
    kubios_threshold_med_win: int = 11
    kubios_threshold_scale_by_mean_rr: bool = True
    kubios_threshold_interp_kind: str = "cubic" # used to replace flagged RR

    # Step C: Kubios "Automatic" (Lipponen & Tarvainen 2019)
    enable_kubios_auto: bool = True
    kubios_auto_edit_peaks: bool = True         # remove "extra" peaks, insert midpoint for "missed"
    kubios_auto_interp_kind: str = "cubic"      # replace ectopic/longshort-only RR

    # Windowing used by the paper/method:
    # - QD thresholds use 91 surrounding beats => halfwin=45
    # - medRR uses 11-beat median => win=11
    kubios_auto_qd_halfwin: int = 45
    kubios_auto_medrr_win: int = 11
    kubios_auto_alpha: float = 5.2
    kubios_auto_c1: float = 0.13
    kubios_auto_c2: float = 0.17

    # Step D: HR resampling + optional Gaussian smoothing (default OFF)
    # If None, infer from hr_target: "1hz" -> 1.0, "64hz" -> fs_bvp
    hr_resample_hz: Optional[float] = None
    enable_hr_gaussian: bool = False
    hr_gaussian_sigma_s: float = 1.5

    # Final cleanup action after all corrections (for remaining HR outliers)
    final_outlier_action: str = "drop"  # drop | interpolate | nan

# ----------------------------
# Helpers
# ----------------------------
def _rolling_median(x: np.ndarray, win: int) -> np.ndarray:
    n = x.size
    out = np.full(n, np.nan, dtype=float)
    if n == 0:
        return out
    win = max(1, int(win))
    half = win // 2
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        out[i] = np.nanmedian(x[a:b])
    return out

def _rolling_qd_abs(x: np.ndarray, half_win: int) -> np.ndarray:
    """
    Quartile deviation (QD) of abs(x) over a centered window:
        QD = (Q3 - Q1) / 2
    Paper uses QD over 91 surrounding beats (±45) : https://doi.org/10.1080/03091902.2019.1640306
    """
    n = x.size
    out = np.full(n, np.nan, dtype=float)
    hw = max(1, int(half_win))
    ax = np.abs(x.astype(float))
    for i in range(n):
        a = max(0, i - hw)
        b = min(n, i + hw + 1)
        w = ax[a:b]
        if np.all(~np.isfinite(w)) or w.size < 4:
            out[i] = np.nan
            continue
        q1, q3 = np.nanpercentile(w, [25, 75])
        out[i] = 0.5 * (q3 - q1)
    return out

def _interp_fill(t: np.ndarray, y: np.ndarray, kind: str = "cubic") -> np.ndarray:
    """
    Fill NaNs in y via interpolation over finite points.
    Falls back to linear if cubic is not feasible.
    """
    from scipy.interpolate import interp1d

    y = y.astype(float).copy()
    m = np.isfinite(t) & np.isfinite(y)
    if m.sum() < 2:
        return y

    # choose kind based on available points
    k = kind
    if kind == "cubic" and m.sum() < 4:
        k = "linear"
    if kind == "quadratic" and m.sum() < 3:
        k = "linear"

    f = interp1d(
        t[m], y[m], kind=k, bounds_error=False, fill_value=np.nan, assume_sorted=True
    )
    y[~m] = f(t[~m])
    return y

def _mark_by_ratio_to_local_median(rr: np.ndarray, win: int, ratio: float) -> np.ndarray:
    med = _rolling_median(rr, win)
    bad = np.zeros(rr.size, dtype=bool)
    ok = np.isfinite(rr) & np.isfinite(med) & (med > 1e-12)
    bad[ok] = (np.abs(rr[ok] - med[ok]) / med[ok]) > ratio
    return bad

def _mark_by_hampel(rr: np.ndarray, win: int, k: float) -> np.ndarray:
    n = rr.size
    bad = np.zeros(n, dtype=bool)
    win = max(3, int(win))
    half = win // 2
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        w = rr[a:b]
        w = w[np.isfinite(w)]
        if w.size < 5:
            continue
        med = np.median(w)
        mad = np.median(np.abs(w - med))
        if mad <= 1e-12:
            continue
        z = np.abs(rr[i] - med) / (1.4826 * mad)
        if np.isfinite(z) and z > k:
            bad[i] = True
    return bad

def _kubios_threshold_mark(rr: np.ndarray, win_med: int, thr_sec_60bpm: float, scale_by_mean_rr: bool) -> np.ndarray:
    """
    Kubios 'Threshold' correction:
        local average = median filtered IBI series
        mark if |RR - local_avg| > threshold_sec
    Threshold levels (at 60 bpm) are documented in Kubios guide : https://www.kubios.com/downloads/Kubios_HRV_Users_Guide_3_2_0.pdf
    Guide notes thresholds are adjusted by mean HR (lower threshold for higher HR).
    We implement a reasonable scaling: thr = thr_60 * meanRR (since meanRR=1s at 60 bpm).
    """
    local_avg = _rolling_median(rr, win_med)
    thr = float(thr_sec_60bpm)
    if scale_by_mean_rr:
        mean_rr = np.nanmean(rr[np.isfinite(rr)])
        if np.isfinite(mean_rr) and mean_rr > 1e-6:
            thr = thr * mean_rr  # meanRR=1.0 at 60 bpm -> unchanged
    bad = np.isfinite(rr) & np.isfinite(local_avg) & (np.abs(rr - local_avg) > thr)
    return bad

def _kubios_auto_detect(rr: np.ndarray, cfg:MergeConfig) -> Dict[str, np.ndarray]:
    """
    Lipponen & Tarvainen 2019 (Kubios 'Automatic') detection.
    Implements the core of:
        - Th1/Th2 from QD of 91 surrounding beats (±45), alpha=5.2 : https://doi.org/10.1080/03091902.2019.1640306 (Lipponen & Tarvainen)
        - medRR from 11-beat median (±5) 
        - ectopic detection (Eq 10) with c1=0.13 c2=0.17 : (Lipponen & Tarvainen)
        - long/short detection in S2 (Eqs 11-13) : (Lipponen & Tarvainen)
        - missed/extra criteria (Eqs 14-15) : (Lipponen & Tarvainen)
    """
    n = rr.size
    eps = 1e-12

    half_qd = int(getattr(cfg, "kubios_auto_qd_halfwin", 45))      # 91 beats total
    med_win = int(getattr(cfg, "kubios_auto_medrr_win", 11))       # 11 beats total
    alpha = float(getattr(cfg, "kubios_auto_alpha", 5.2))
    c1 = float(getattr(cfg, "kubios_auto_c1", 0.13))
    c2 = float(getattr(cfg, "kubios_auto_c2", 0.17))

    # dRRs (Eq 1)
    dRRs = np.zeros(n, dtype=float)
    dRRs[1:] = rr[1:] - rr[:-1]

    # Th1 (Eq 2) and normalized dRR (Eq 3)
    Th1 = alpha * _rolling_qd_abs(dRRs, half_qd)
    Th1 = np.where(np.isfinite(Th1) & (Th1 > eps), Th1, np.nan)
    dRR = dRRs / Th1

    # medRR (Eq 4) and mRRs (Eq 4-5), Th2 (Eq 6), mRR (Eq 7)
    medRR = _rolling_median(rr, med_win)
    mRRs = rr - medRR
    mRRs = np.where(mRRs < 0, 2.0 * mRRs, mRRs)  # Eq 5
    Th2 = alpha * _rolling_qd_abs(mRRs, half_qd)  # Eq 6 uses QD of |mRRs| over 91 surrounding beats
    Th2 = np.where(np.isfinite(Th2) & (Th2 > eps), Th2, np.nan)
    mRR = mRRs / Th2

    # --- Ectopic detection (Eq 8-10): only when |dRR| > 1
    S11 = dRR.copy()
    S12 = np.full(n, np.nan, dtype=float)
    for j in range(n):
        if not np.isfinite(S11[j]):
            continue
        jm1 = j - 1
        jp1 = j + 1
        if jm1 < 0 or jp1 >= n:
            continue
        """ if S11[j] > 0:
            S12[j] = np.nanmax([dRR[jm1], dRR[jp1]])
        else:
            S12[j] = np.nanmin([dRR[jm1], dRR[jp1]]) """
        vals = np.array([dRR[jm1], dRR[jp1]], dtype=float)
        if not np.any(np.isfinite(vals)):
            continue
        if S11[j] > 0:
            S12[j] = np.nanmax(vals)
        else:
            S12[j] = np.nanmin(vals)

    ectopic = np.zeros(n, dtype=bool)
    cand = np.isfinite(S11) & np.isfinite(S12) & (np.abs(S11) > 1)
    ectopic[cand] = (
        ((S11[cand] > 1) & (S12[cand] < (-c1 * S11[cand] - c2))) |
        ((S11[cand] < -1) & (S12[cand] > (-c1 * S11[cand] + c2)))
    )

    # --- Long/short detection (Eq 11-13)
    S21 = dRR.copy()
    S22 = np.full(n, np.nan, dtype=float)
    for j in range(n):
        jp1 = j + 1
        jp2 = j + 2
        if jp2 >= n or not np.isfinite(S21[j]):
            continue
        """ if S21[j] >= 0:
            S22[j] = np.nanmin([dRR[jp1], dRR[jp2]])
        else:
            S22[j] = np.nanmax([dRR[jp1], dRR[jp2]]) """
        vals = np.array([dRR[jp1], dRR[jp2]], dtype=float)
        if not np.any(np.isfinite(vals)):
            continue
        if S21[j] >= 0:
            S22[j] = np.nanmin(vals)
        else:
            S22[j] = np.nanmax(vals)

    longshort = np.zeros(n, dtype=bool)
    ok2 = np.isfinite(S21) & np.isfinite(S22)
    # The PDF text extraction duplicates one inequality; conceptually it is symmetric:
    # (S21>1 & S22<-1) OR (S21<-1 & S22>1) OR |mRR|>3. (Lipponen & Tarvainen)
    longshort[ok2] = (
        ((S21[ok2] > 1) & (S22[ok2] < -1)) |
        ((S21[ok2] < -1) & (S22[ok2] > 1)) |
        (np.abs(mRR[ok2]) > 3)
    )

    # Also classify beat j+1 as long/short under the condition described in `Lipponen & Tarvainen`
    for j in np.where(longshort)[0]:
        if j + 2 < n:
            if np.isfinite(dRR[j+1]) and np.isfinite(dRR[j+2]) and (abs(dRR[j+1]) < abs(dRR[j+2])):
                longshort[j+1] = True

    # Missed/extra (Eq 14-15), only meaningful where long/short is true and neighbors exist
    missed = np.zeros(n, dtype=bool)
    extra = np.zeros(n, dtype=bool)

    for j in np.where(longshort & ~ectopic)[0]:
        if not (np.isfinite(rr[j]) and np.isfinite(medRR[j]) and np.isfinite(Th2[j])):
            continue
        # missed: RR(j) is long (~2x), so RR(j)/2 close to medRR
        if abs(rr[j] / 2.0 - medRR[j]) < Th2[j]:
            missed[j] = True
            continue
        # extra: RR(j) and RR(j+1) are short, sum close to medRR(j)
        if j + 1 < n and np.isfinite(rr[j+1]):
            if abs((rr[j] + rr[j+1]) - medRR[j]) < Th2[j]:
                extra[j] = True

    # long/short remaining after excluding missed/extra are corrected by interpolation (per paper in (Lipponen & Tarvainen)
    ls_only = longshort & ~missed & ~extra & ~ectopic

    return dict(
        ectopic=ectopic,
        longshort=longshort,
        missed=missed,
        extra=extra,
        longshort_only=ls_only,
        dRR=dRR,
        mRR=mRR,
    )


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
def nearest_neighbor_on_grid(
    t_s: np.ndarray,
    v: np.ndarray,
    t_g: np.ndarray,
    mode: str = "one_to_one",   # "one_to_one" or "per_grid"
) -> np.ndarray:
    """
    Nearest-neighbor snapping between sample times t_s and grid times t_g.

    Parameters
    ----------
    t_s : sorted sample timestamps (ns)
    v   : sample values
    t_g : sorted grid timestamps (ns)
    mode:
      - "per_grid": for each grid time, take value of nearest sample time.
                    A sample may be reused for many grid points.
      - "one_to_one": each sample is snapped to ONLY ONE nearest grid time (its nearest gridline).
                      Each sample used at most once; grid points without an assigned sample are NaN.

    Returns
    -------
    out : array aligned to t_g
    """
    t_s = np.asarray(t_s, dtype=np.int64)
    t_g = np.asarray(t_g, dtype=np.int64)
    v = np.asarray(v, dtype=np.float64)

    out = np.full(t_g.shape, np.nan, dtype=np.float64)
    if t_s.size == 0 or t_g.size == 0:
        return out

    if mode == "per_grid":
        # assume t_s sorted
        idx = np.searchsorted(t_s, t_g, side="left")
        idx0 = np.clip(idx - 1, 0, t_s.size - 1)
        idx1 = np.clip(idx, 0, t_s.size - 1)

        d0 = np.abs(t_g - t_s[idx0])
        d1 = np.abs(t_g - t_s[idx1])
        pick = np.where(d1 < d0, idx1, idx0)
        return v[pick].astype(np.float64)

    if mode == "one_to_one":
        # For each sample, find its nearest grid index
        j = np.searchsorted(t_g, t_s, side="left")
        j0 = np.clip(j - 1, 0, t_g.size - 1)
        j1 = np.clip(j, 0, t_g.size - 1)

        d0 = np.abs(t_s - t_g[j0])
        d1 = np.abs(t_s - t_g[j1])
        j_pick = np.where(d1 < d0, j1, j0)

        # If multiple samples snap to same grid index, keep the closest one;
        # if tie, keep the later sample (arbitrary but deterministic).
        # We'll resolve by sorting by (grid_index, distance, sample_time) then taking first per group.
        dist = np.abs(t_s - t_g[j_pick]).astype(np.int64)
        order = np.lexsort((-t_s, dist, j_pick))  # grid asc, dist asc, time desc
        j_pick_ord = j_pick[order]
        v_ord = v[order]
        dist_ord = dist[order]

        # unique per grid index: take first occurrence in sorted order
        _, first_idx = np.unique(j_pick_ord, return_index=True)
        chosen = first_idx
        out[j_pick_ord[chosen]] = v_ord[chosen]
        return out

    raise ValueError(f"Unknown mode: {mode}")


def interp_on_grid(t_s: np.ndarray, v: np.ndarray, t_g: np.ndarray, kind: str) -> np.ndarray:
    """
    Robust interpolation to grid:
      - uses relative seconds to avoid huge ns magnitudes
      - removes duplicate timestamps (keep last)
      - degrades cubic/quadratic -> linear if insufficient points
      - (optional) no extrapolation: fills outside-range with NaN
    """
    if t_s.size < 2:
        return np.full(t_g.shape, np.nan, dtype=np.float64)

    # sort + keep-last on duplicates
    t_s, v = ensure_sorted_unique_times(t_s.astype(np.int64), v.astype(np.float64))

    # finite mask
    m = np.isfinite(v)
    if m.sum() < 2:
        return np.full(t_g.shape, np.nan, dtype=np.float64)

    t_s = t_s[m]
    v = v[m]

    # interpolate in relative seconds for numerical stability
    t0 = int(t_s[0])
    xs = (t_s - t0).astype(np.float64) / 1e9
    xg = (t_g.astype(np.int64) - t0).astype(np.float64) / 1e9

    # decide effective kind based on available points & SciPy
    eff_kind = kind
    if (not SCIPY_OK) and eff_kind in ("quadratic", "cubic"):
        eff_kind = "linear"

    if eff_kind == "cubic":
        if xs.size < 4:
            eff_kind = "quadratic" if (SCIPY_OK and xs.size >= 3) else "linear"
    elif eff_kind == "quadratic":
        if xs.size < 3:
            eff_kind = "linear"

    # linear path (no SciPy required)
    if eff_kind == "linear" or (not SCIPY_OK):
        return np.interp(xg, xs, v, left=np.nan, right=np.nan).astype(np.float64)

    # SciPy quadratic/cubic
    f = interp1d(
        xs,
        v,
        kind=eff_kind,
        bounds_error=False,
        fill_value=np.nan, # or "extrapolate": BEWARE OF HUGE ARTIFACTS
        assume_sorted=True,
    )
    return f(xg).astype(np.float64)



def map_scalar_channel_to_grid(
    df_seg: Optional[pd.DataFrame],
    grid_ts: np.ndarray,
    method: str,
    interp_kind: str,
    snap_kind:str,
) -> np.ndarray:
    if df_seg is None or df_seg.empty:
        return np.full(grid_ts.shape, np.nan, dtype=np.float64)
    t = np.asarray(df_seg["timestamp_ns"], dtype=np.int64)
    v = np.asarray(df_seg["value"], dtype=np.float64)
    t, v = ensure_sorted_unique_times(t, v)

    if method == "snap":
        return nearest_neighbor_on_grid(t, v, grid_ts, snap_kind)
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
    """ m = np.isfinite(hr)
    m &= (hr >= cfg.hr_min_bpm) & (hr <= cfg.hr_max_bpm)
    t_ns = t_ns[m]
    hr = hr[m]
    if hr.size < 5:
        return t_ns, hr

    med = np.median(hr)
    mad = np.median(np.abs(hr - med))
    if mad <= 1e-9:
        return t_ns, hr
    z = np.abs(hr - med) / (1.4826 * mad) # z score of MAD
    keep = z <= cfg.hr_mad_z
    return t_ns[keep], hr[keep] """
    #----------------------------------------
    """
    Robust peaks/RR -> HR correction pipeline (Kubios-like) with optional 1Hz resampling.

    Input conventions (auto-detected):
      1. If len(t_ns) == len(hr) + 1: t_ns are PEAK timestamps (ns); hr is ignored/recomputed.
      2. Else if len(t_ns) == len(hr): treat (t_ns, hr) as instantaneous HR samples; RR is derived as 60/hr
        (note: missed/extra peak editing cannot be done without peak timestamps).
      3. Else: fall back to original behavior (HR-only global filter).
      
      Note that 1. and 2. are essentially the same if (before this function) hr are computed from t_ns,
        and t_ns[1:] and hr are provided

    Returns:
      - If cfg.hr_resample_hz is None: returns irregular instantaneous HR samples (t_ns_hr, hr_inst).
      - Else: returns uniform grid (t_ns_grid, hr_grid) at cfg.hr_resample_hz (default 1.0).
    """
    # ----------------------------
    # Defaults / config accessors
    # ----------------------------
    hr_min_bpm = float(getattr(cfg, "hr_min_bpm", 30.0))
    hr_max_bpm = float(getattr(cfg, "hr_max_bpm", 220.0))
    hr_mad_z   = float(getattr(cfg, "hr_mad_z", 5.0))

    rr_min = 60.0 / hr_max_bpm
    rr_max = 60.0 / hr_min_bpm

    enable_auto = bool(getattr(cfg, "enable_kubios_auto", True))
    enable_thresh = bool(getattr(cfg, "enable_kubios_threshold", True))

    # Step A
    enable_ratio = bool(getattr(cfg, "enable_rr_ratio_median", True))
    ratio_win = int(getattr(cfg, "rr_ratio_win_beats", 11))
    ratio_thr = float(getattr(cfg, "rr_ratio_thr", 0.25))

    enable_hampel = bool(getattr(cfg, "enable_rr_hampel", True))
    hampel_win = int(getattr(cfg, "rr_hampel_win_beats", 11))
    hampel_k = float(getattr(cfg, "rr_hampel_k", 3.0))

    # Step B (Kubios threshold correction)
    kubios_level = str(getattr(cfg, "kubios_threshold_level", "medium")).lower()
    kubios_level_map = {
        "very_low": 0.45,
        "low": 0.35,
        "medium": 0.25,
        "strong": 0.15,
        "very_strong": 0.05,
    }  # seconds @ 60 bpm : 
    thr_60 = float(getattr(cfg, "kubios_threshold_sec_60bpm", None) or kubios_level_map.get(kubios_level, 0.25))
    thr_win_med = int(getattr(cfg, "kubios_threshold_med_win", 11))
    thr_scale_by_mean_rr = bool(getattr(cfg, "kubios_threshold_scale_by_mean_rr", True))
    thr_interp_kind = str(getattr(cfg, "kubios_threshold_interp_kind", "cubic"))

    # Step C (Kubios auto)
    auto_edit_peaks = bool(getattr(cfg, "kubios_auto_edit_peaks", True))
    auto_interp_kind = str(getattr(cfg, "kubios_auto_interp_kind", "cubic"))

    # Step D (optional resample HR)
    hr_resample_hz = getattr(cfg, "hr_resample_hz", 1.0)  # None or float
    hr_interp_kind = str(getattr(cfg, "hr_interp_kind", "cubic"))
    enable_gauss = bool(getattr(cfg, "enable_hr_gaussian", False))  # default OFF 
    gauss_sigma_s = float(getattr(cfg, "hr_gaussian_sigma_s", 1.5))

    final_outlier_action = str(getattr(cfg, "final_outlier_action", "drop")).lower()  # drop|interpolate|nan

    # ----------------------------
    # Detect input mode
    # ----------------------------
    t_ns = np.asarray(t_ns)
    hr = np.asarray(hr)

    if t_ns.size < 3:
        return t_ns, hr

    # Ensure sorted unique times
    order = np.argsort(t_ns)
    t_ns = t_ns[order]
    if hr.size == t_ns.size:
        hr = hr[order]
    t_ns = t_ns.astype(np.int64)

    # Mode 1: we have peak times (best)
    have_peaks = (hr.size + 1 == t_ns.size)

    # Mode 2: HR samples only
    have_hr_samples = (hr.size == t_ns.size)

    # Fallback: unknown
    if not have_peaks and not have_hr_samples:
        # original behavior: HR-only global filter
        m = np.isfinite(hr)
        m &= (hr >= hr_min_bpm) & (hr <= hr_max_bpm)
        t0 = t_ns[m]; h0 = hr[m]
        if h0.size < 5:
            return t0, h0
        med = np.median(h0)
        mad = np.median(np.abs(h0 - med))
        if mad <= 1e-9:
            return t0, h0
        z = np.abs(h0 - med) / (1.4826 * mad)
        keep = z <= hr_mad_z
        return t0[keep], h0[keep]

    # ----------------------------
    # Build RR series
    # ----------------------------
    if have_peaks:
        peaks_ns = t_ns
        peaks_s = peaks_ns.astype(np.float64) * 1e-9
        rr = np.diff(peaks_s)  # seconds
        rr_t_s = peaks_s[1:]   # assign RR/HR to the ending peak time
    else:
        # HR samples only; derive RR (cannot edit peaks)
        # Guard divide by zero
        rr_t_s = t_ns.astype(np.float64) * 1e-9
        rr = 60.0 / np.where(np.isfinite(hr) & (hr > 1e-9), hr, np.nan)

    # Hard bounds on RR
    rr = rr.astype(float)
    rr[(rr < rr_min) | (rr > rr_max)] = np.nan

    # ----------------------------
    # Step C: Kubios Automatic (Lipponen & Tarvainen, 2019)
    # ----------------------------
    if enable_auto and rr.size >= 10:
        det = _kubios_auto_detect(rr, cfg)

        if have_peaks and auto_edit_peaks:
            # Apply missed/extra edits at the PEAK level (best fidelity).
            # extra[j] => remove peak at index (j+1) to merge RR[j] and RR[j+1]
            # missed[j] => insert peak at midpoint between peak[j] and peak[j+1]
            extra_idx = np.where(det["extra"])[0]
            missed_idx = np.where(det["missed"])[0]

            # remove peaks (descending to keep indices valid)
            remove_peak_idx = set()
            for j in extra_idx:
                k = j + 1
                if 0 < k < peaks_s.size - 1:  # don't remove endpoints
                    remove_peak_idx.add(k)
            if remove_peak_idx:
                keep = np.ones(peaks_s.size, dtype=bool)
                keep[list(remove_peak_idx)] = False
                peaks_s = peaks_s[keep]

            # insert peaks (ascending in time)
            if missed_idx.size > 0:
                # recompute RR after removals before inserting midpoints
                rr_tmp = np.diff(peaks_s)
                # map missed indices conservatively: only insert where index still valid
                inserts = []
                for j in missed_idx:
                    if 0 <= j < rr_tmp.size:
                        mid = 0.5 * (peaks_s[j] + peaks_s[j + 1])
                        inserts.append(mid)
                if inserts:
                    peaks_s = np.sort(np.concatenate([peaks_s, np.array(inserts, dtype=float)]))

            # Recompute RR after editing peaks
            rr = np.diff(peaks_s)
            rr_t_s = peaks_s[1:]
            rr[(rr < rr_min) | (rr > rr_max)] = np.nan

            # Re-run detection once to get ectopic/longshort flags on the edited series, then interpolate those
            det = _kubios_auto_detect(rr, cfg)

        # Interpolate ectopic + long/short-only beats (per paper in Lipponen & Tarvainen) 
        bad_auto = det["ectopic"] | det["longshort_only"]
        if bad_auto.any():
            rr2 = rr.copy()
            rr2[bad_auto] = np.nan
            rr = _interp_fill(rr_t_s, rr2, kind=auto_interp_kind)

    # ----------------------------
    # Step B: Kubios Threshold correction
    # ----------------------------
    if enable_thresh and rr.size >= 5:
        bad_thr = _kubios_threshold_mark(rr, thr_win_med, thr_60, thr_scale_by_mean_rr)
        if bad_thr.any():
            rr2 = rr.copy()
            rr2[bad_thr] = np.nan
            rr = _interp_fill(rr_t_s, rr2, kind=thr_interp_kind)

    # ----------------------------
    # Step A: Robust local outlier marking (ratio-to-median and/or Hampel)
    # ----------------------------
    bad_A = np.zeros(rr.size, dtype=bool)
    if enable_ratio and rr.size >= 5:
        bad_A |= _mark_by_ratio_to_local_median(rr, ratio_win, ratio_thr)
    if enable_hampel and rr.size >= 7:
        bad_A |= _mark_by_hampel(rr, hampel_win, hampel_k)

    if bad_A.any():
        rr2 = rr.copy()
        rr2[bad_A] = np.nan
        rr = _interp_fill(rr_t_s, rr2, kind="linear")  # conservative fill for A

    # Final hard bounds again (post interpolation)
    rr[(rr < rr_min) | (rr > rr_max)] = np.nan

    # ----------------------------
    # RR -> instantaneous HR (irregular)
    # ----------------------------
    hr_inst = 60.0 / rr
    t_hr_s = rr_t_s
    t_hr_ns = (t_hr_s * 1e9).astype(np.int64)

    # ----------------------------
    # Final global HR outlier cleanup
    # ----------------------------
    m = np.isfinite(hr_inst) & (hr_inst >= hr_min_bpm) & (hr_inst <= hr_max_bpm)
    if m.sum() >= 5:
        h0 = hr_inst[m]
        med = np.median(h0)
        mad = np.median(np.abs(h0 - med))
        if mad > 1e-9:
            z = np.abs(hr_inst - med) / (1.4826 * mad)
            m &= (z <= hr_mad_z)

    if final_outlier_action == "nan":
        hr_inst = hr_inst.copy()
        hr_inst[~m] = np.nan
    elif final_outlier_action == "interpolate":
        hr2 = hr_inst.copy()
        hr2[~m] = np.nan
        hr_inst = _interp_fill(t_hr_s, hr2, kind="linear")
        m = np.isfinite(hr_inst)
    else:  # "drop" (default)
        t_hr_ns = t_hr_ns[m]
        t_hr_s = t_hr_s[m]
        hr_inst = hr_inst[m]

    # ----------------------------
    # Step D: Optional resample to uniform grid (e.g., 1 Hz) + optional Gaussian smoothing
    # ----------------------------
    if hr_resample_hz is None:
        return t_hr_ns, hr_inst

    from scipy.interpolate import interp1d

    hz = float(hr_resample_hz) if float(hr_resample_hz) > 0 else 1.0
    dt = 1.0 / hz

    # Define grid within available time support (no extrapolation)
    if t_hr_s.size < 2:
        return t_hr_ns, hr_inst

    t0 = np.ceil(t_hr_s[0] / dt) * dt
    t1 = np.floor(t_hr_s[-1] / dt) * dt
    if t1 <= t0:
        return t_hr_ns, hr_inst

    grid_s = np.arange(t0, t1 + 0.5 * dt, dt)
    f = interp1d(t_hr_s, hr_inst, kind=hr_interp_kind if hr_inst.size >= 4 else "linear",
                 bounds_error=False, fill_value=np.nan, assume_sorted=True)
    hr_grid = f(grid_s)

    if enable_gauss:
        # Gaussian smoothing on uniform grid (OFF by default to avoid smearing)
        from scipy.ndimage import gaussian_filter1d
        sigma = max(0.0, gauss_sigma_s) / dt
        if sigma > 0:
            # only smooth finite segments
            finite = np.isfinite(hr_grid)
            if finite.any():
                x = hr_grid.copy()
                # fill NaNs by linear interp for smoothing, then restore NaNs
                x = _interp_fill(grid_s, x, kind="linear")
                x = gaussian_filter1d(x, sigma=sigma, mode="nearest")
                hr_grid[finite] = x[finite]

    grid_ns = (grid_s * 1e9).astype(np.int64)
    return grid_ns, hr_grid


def bandpass_filter_64hz(x: np.ndarray, cfg: MergeConfig) -> np.ndarray:
    """
    Baseline-preserving Butterworth filtering on 64Hz HR.
    If low=0 and high>0 -> lowpass smoothing.
    If low>0 and high>0 -> bandpass of deviations, then baseline added back.
    """
    if not SCIPY_OK:
        return x

    fs = 64.0
    low = float(cfg.hr_bp_low_hz)
    high = float(cfg.hr_bp_high_hz)

    # If both are 0, filtering disabled
    if (low <= 0) and (high <= 0):
        return x

    nyq = fs / 2
    if high > 0 and high >= nyq:
        high = 0.99 * nyq

    # Choose filter type
    if low <= 0 and high > 0:
        btype = "lowpass"
        Wn = high / nyq
    elif high <= 0 and low > 0:
        btype = "highpass"
        Wn = low / nyq
    else:
        btype = "bandpass"
        Wn = [low / nyq, high / nyq]

    b, a = butter(cfg.hr_bp_order, Wn, btype=btype)

    y = x.astype(np.float64).copy()
    m = np.isfinite(y)
    if m.sum() < 10:
        return y

    # Baseline to preserve absolute HR scale
    baseline = np.nanmedian(y)
    y[m] = y[m] - baseline

    # Filter contiguous finite runs
    idx = np.where(m)[0]
    runs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    for r in runs:
        if r.size < max(10, 3 * (cfg.hr_bp_order + 1)):
            continue
        y[r] = filtfilt(b, a, y[r])

    y[m] = y[m] + baseline
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

    #t_hr_inst, hr_inst = robust_filter_hr(t_hr_inst, hr_inst, cfg)
    t_hr_inst, hr_inst = robust_filter_hr(t_peaks, hr_inst, cfg)
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
    #t1_ns = (np.arange(start_s, (end_s + 1)*4, dtype=np.int64) * 250000000)

    # Interpolate instantaneous HR onto 1Hz grid
    kind = cfg.hr_interp_kind
    # If SciPy not available, quadratic/cubic will fall back to linear inside interp_on_grid
    hr_1 = interp_on_grid(t_hr_inst, hr_inst, t1_ns, kind=kind)

    if cfg.hr_target == "1hz":
        # If a 1Hz dataset is needed, return HR broadcast to 64Hz grid by nearest-neighbor.
        # (Still keeps final merged DF at 64Hz.)
        hr_64 = nearest_neighbor_on_grid(t1_ns.astype(np.int64), hr_1.astype(np.float64), grid_ts, cfg.map_snap_kind)
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

    eda_map = map_scalar_channel_to_grid(eda_seg, grid_ts, method=cfg.map_method, interp_kind=cfg.map_interp_kind, snap_kind=cfg.map_snap_kind)
    temp_map = map_scalar_channel_to_grid(temp_seg, grid_ts, method=cfg.map_method, interp_kind=cfg.map_interp_kind, snap_kind=cfg.map_snap_kind)

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
    if cfg.fill_short_gaps and cfg.gap_fill_channels and cfg.map_snap_kind != "one_to_one":
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


def process_all_subjects(root: str, combined_out_dir: str, cfg: MergeConfig):
    """
    Expects:
      root/
        1/  (subject folder)
          1_bvp.csv, 1_eda.csv, 1_temperature.csv, 1_systolicPeaks.csv
        2/
          2_bvp.csv, ...
      Writes:
        combined_out_dir/1_merged_64hz.csv, ...
    """
    subj_dirs = [d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)]
    subj_dirs.sort()

    for sd in subj_dirs:
        subject_id = os.path.basename(sd)
        out_path = os.path.join(combined_out_dir, f"{subject_id}_merged_64hz.csv")
        print(f"[SUBJECT {subject_id}] -> {out_path}")
        process_subject(sd, subject_id, out_path, cfg)
        
def _find_file_ci(folder: str, candidates: List[str]) -> Optional[str]:
    """
    Case-insensitive filename lookup.
    candidates are filenames like ["TEMP.csv", "temp.csv"].
    """
    files = {f.lower(): os.path.join(folder, f) for f in os.listdir(folder)}
    for c in candidates:
        p = files.get(c.lower())
        if p and os.path.isfile(p):
            return p
    return None


def load_empatica_scalar_csv(subject_dir: str, basename: str) -> Optional[pd.DataFrame]:
    """
    Load Empatica E4 scalar channel file:
      row0: t0 (unix seconds, UTC)
      row1: fs (Hz)
      rows2+: values
    Returns df with columns: timestamp_ns, datetime_utc, segment, value
    """
    path = _find_file_ci(subject_dir, [f"{basename}.csv"])
    if path is None:
        return None

    raw = pd.read_csv(path, header=None)
    if raw.shape[0] < 3:
        return None

    t0_s = float(raw.iloc[0, 0])
    fs = float(raw.iloc[1, 0])
    values = pd.to_numeric(raw.iloc[2:, 0], errors="coerce").to_numpy(dtype=np.float64)

    n = values.size
    if n == 0 or fs <= 0:
        return None

    t0_ns = int(round(t0_s * 1e9))
    offsets = np.rint((np.arange(n, dtype=np.float64) / fs) * 1e9).astype(np.int64)
    t_ns = t0_ns + offsets

    df = pd.DataFrame({
        "timestamp_ns": t_ns,
        "datetime_utc": pd.to_datetime(t_ns, unit="ns", utc=True),
        "segment": 0,
        "value": values,
    })
    # keep unique timestamps (just in case)
    df = df.sort_values("timestamp_ns", kind="mergesort").drop_duplicates("timestamp_ns", keep="last").reset_index(drop=True)
    return df


def build_empatica_session_merged_df(subject_dir: str, subject_id: str, cfg: MergeConfig) -> pd.DataFrame:
    """
    Merge EDA, TEMP, HR onto BVP-native 64Hz grid for Empatica E4 export session.
    Produces a single segment=0 and section=0 file.
    """
    bvp = load_empatica_scalar_csv(subject_dir, "BVP")
    eda = load_empatica_scalar_csv(subject_dir, "EDA")
    temp = load_empatica_scalar_csv(subject_dir, "TEMP")
    hr = load_empatica_scalar_csv(subject_dir, "HR")

    if bvp is None or bvp.empty:
        raise ValueError(f"[{subject_id}] Missing/empty BVP.csv in {subject_dir}")

    # union bounds for optional grid extension
    tmins = [int(bvp["timestamp_ns"].min())]
    tmaxs = [int(bvp["timestamp_ns"].max())]
    for df in (eda, temp, hr):
        if df is not None and not df.empty:
            tmins.append(int(df["timestamp_ns"].min()))
            tmaxs.append(int(df["timestamp_ns"].max()))
    start_ns, end_ns = min(tmins), max(tmaxs)

    # build 64Hz grid using BVP phase
    grid_ts = build_bvp_native_grid(bvp, start_ns, end_ns, cfg)
    if grid_ts.size == 0:
        return pd.DataFrame()

    # Map BVP onto grid (fast path, then fallback)
    bvp_t = bvp["timestamp_ns"].to_numpy(np.int64)
    bvp_v = bvp["value"].to_numpy(np.float64)
    bvp_t, bvp_v = ensure_sorted_unique_times(bvp_t, bvp_v)

    bvp_map = np.full(grid_ts.shape, np.nan, dtype=np.float64)
    pos = np.searchsorted(grid_ts, bvp_t)
    ok = (pos >= 0) & (pos < grid_ts.size) & (grid_ts[pos] == bvp_t)
    bvp_map[pos[ok]] = bvp_v[ok]
    if np.isfinite(bvp_map).sum() < 0.9 * min(grid_ts.size, bvp_t.size):
        bvp_map = nearest_neighbor_on_grid(bvp_t, bvp_v, grid_ts, mode="per_grid")

    # EDA/TEMP mapping
    eda_map = map_scalar_channel_to_grid(
        eda, grid_ts,
        method=cfg.map_method,
        interp_kind=cfg.map_interp_kind,
        snap_kind=cfg.map_snap_kind,
    )
    temp_map = map_scalar_channel_to_grid(
        temp, grid_ts,
        method=cfg.map_method,
        interp_kind=cfg.map_interp_kind,
        snap_kind=cfg.map_snap_kind,
    )

    # HR mapping (separate knobs; default cubic interpolation onto 64Hz)
    if hr is None or hr.empty:
        hr_map = np.full(grid_ts.shape, np.nan, dtype=np.float64)
    else:
        hr_map = map_scalar_channel_to_grid(
            hr, grid_ts,
            method=getattr(cfg, "hr_map_method", "snap"),
            interp_kind=getattr(cfg, "hr_map_interp_kind", "cubic"),
            snap_kind=cfg.map_snap_kind,
        )
        # optional smoothing
        hr_map = bandpass_filter_64hz(hr_map, cfg)

    merged = pd.DataFrame({
        "timestamp_ns": grid_ts,
        "datetime_utc": pd.to_datetime(grid_ts, unit="ns", utc=True),
        "segment": 0,
        "section": 0,
        "bvp": bvp_map,
        "eda": eda_map,
        "temperature": temp_map,
        "hr": hr_map,
    })
    return merged


def load_pain_data_csv(subject_dir: str, cfg: MergeConfig) -> Optional[pd.DataFrame]:
    """
    pain_data.csv has NO header:
      local_timestamp, pain_level, subject_id
    Timestamps are local ET (EDT/EST), convert to UTC ns.
    """
    path = _find_file_ci(subject_dir, ["pain_data.csv"])
    if path is None:
        return None

    p = pd.read_csv(path, header=None, names=["timestamp_local", "PainLevel", "subject_id"])
    p["PainLevel"] = pd.to_numeric(p["PainLevel"], errors="coerce")

    t_local = pd.to_datetime(p["timestamp_local"], errors="coerce")
    tz = getattr(cfg, "pain_tz", "America/New_York")
    t_utc = (t_local
             .dt.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
             .dt.tz_convert("UTC"))

    p["timestamp_utc"] = t_utc
    p["timestamp_ns"] = p["timestamp_utc"].astype("int64")
    return p


def _nearest_index_with_distance(target_ns: np.ndarray, grid_ns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (nearest_idx, distance_ns) for each target.
    """
    idx = np.searchsorted(grid_ns, target_ns, side="left")
    idx0 = np.clip(idx - 1, 0, grid_ns.size - 1)
    idx1 = np.clip(idx,     0, grid_ns.size - 1)

    d0 = np.abs(target_ns - grid_ns[idx0])
    d1 = np.abs(target_ns - grid_ns[idx1])
    pick = np.where(d1 < d0, idx1, idx0)
    dist = np.where(d1 < d0, d1, d0)
    return pick, dist


def join_pain_to_merged_df(merged: pd.DataFrame, pain: pd.DataFrame, cfg: MergeConfig) -> pd.DataFrame:
    """
    Snap each pain row to ONE nearest physio row (no interpolation).
    Adds PainLevel column (sparse by default).
    """
    out = merged.sort_values("timestamp_ns", kind="mergesort").reset_index(drop=True)
    grid_ns = out["timestamp_ns"].to_numpy(np.int64)

    out["PainLevel"] = np.nan

    if pain is None or pain.empty:
        return out

    pain_valid = pain[pain["PainLevel"].notna()].copy()
    if pain_valid.empty:
        return out

    t_ns = pain_valid["timestamp_ns"].to_numpy(np.int64)
    idx, dist_ns = _nearest_index_with_distance(t_ns, grid_ns)

    max_snap_s = float(getattr(cfg, "pain_max_snap_s", -1.0))
    if max_snap_s > 0:
        keep = (dist_ns.astype(np.float64) / 1e9) <= max_snap_s
        idx = idx[keep]
        pain_valid = pain_valid.iloc[np.where(keep)[0]]

    out.loc[idx, "PainLevel"] = pain_valid["PainLevel"].to_numpy(dtype=np.float64)
    return out



def process_empatica_e4_physio_all_subjects(root: str, out_dir: str, cfg: MergeConfig) -> None:
    """
    Merge Empatica E4 exports (BVP/EDA/TEMP/HR) onto a BVP-native 64 Hz grid.

    This produces per-subject physiological files *without* self-report labels. Use
    `src.merge.batch_join_subject_folders` to snap `pain_data.csv` (or another
    self-report file) onto the resulting grid.

    Expected input layout
    ---------------------
      root/
        101/
          BVP.csv, EDA.csv, TEMP.csv, HR.csv, ...
        102/
          ...

    Output
    ------
      out_dir/
        101_merged_64hz.csv
        102_merged_64hz.csv
        ...

    Notes
    -----
    - The function reads standard Empatica CSV exports where the first row contains
      the session start time (Unix seconds, UTC), the second row contains the sampling
      rate, and subsequent rows contain samples.
    - Output files use UTC nanosecond timestamps (`timestamp_ns`).
    """
    subj_dirs = [d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)]
    subj_dirs.sort()

    os.makedirs(out_dir, exist_ok=True)

    for sd in subj_dirs:
        subject_id = os.path.basename(sd)
        merged = build_empatica_session_merged_df(sd, subject_id, cfg)
        out_path = os.path.join(out_dir, f"{subject_id}_merged_64hz.csv")
        merged.to_csv(out_path, index=False)
        print(f"[OK] {subject_id} -> {out_path}")


def process_empatica_e4_all_subjects(root: str, out_dir: str, cfg: MergeConfig):
    """
    Expects:
      root/
        101/
          EDA.csv, BVP.csv, HR.csv, TEMP.csv, pain_data.csv, ...
        102/
          ...
    Writes:
      out_dir/101_merged_64hz_with_pain.csv
    """
    subj_dirs = [d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)]
    subj_dirs.sort()

    os.makedirs(out_dir, exist_ok=True)

    for sd in subj_dirs:
        subject_id = os.path.basename(sd)

        merged = build_empatica_session_merged_df(sd, subject_id, cfg)
        pain = load_pain_data_csv(sd, cfg)
        merged2 = join_pain_to_merged_df(merged, pain, cfg)

        out_path = os.path.join(out_dir, f"{subject_id}_merged_64hz_with_pain.csv")
        merged2.to_csv(out_path, index=False)
        print(f"[OK] {subject_id} -> {out_path}")
