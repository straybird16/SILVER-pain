import os
import glob
import numpy as np
import pandas as pd

from avro.datafile import DataFileReader
from avro.io import DatumReader


# ----------------------------
# helpers: time unit + parsing
# ----------------------------
def to_epoch_ns(ts: int) -> int:
    """
    Heuristic conversion to epoch nanoseconds.
      - peaksTimeNanos: nanoseconds (≈1e18)
      - timestampStart: often microseconds (≈1e15)
    """
    ts = int(ts)
    if ts >= 10**17:      # ns
        return ts
    if ts >= 10**14:      # us
        return ts * 1_000
    if ts >= 10**11:      # ms
        return ts * 1_000_000
    return ts * 1_000_000_000  # seconds


def segment_to_df(seg: dict, signal_name: str, segment_id: int) -> pd.DataFrame:
    """
    Convert a sampled segment dict:
      {'timestampStart': ..., 'samplingFrequency': ..., 'values': [...]}
    into DataFrame with timestamp_ns + signal columns + segment.
    """
    ts0_ns = to_epoch_ns(seg["timestampStart"])
    fs = float(seg["samplingFrequency"])
    values = seg.get("values", [])
    if not values:
        return pd.DataFrame(columns=["timestamp_ns", "segment"])

    n = len(values)
    offsets_ns = np.rint((np.arange(n, dtype=np.float64) / fs) * 1e9).astype(np.int64)
    t_ns = ts0_ns + offsets_ns

    first = values[0]

    # Case A: values are dicts like {"x":..,"y":..,"z":..}
    if isinstance(first, dict):
        cols = sorted(first.keys())
        data = {c: [v.get(c, np.nan) if isinstance(v, dict) else np.nan for v in values] for c in cols}
        df = pd.DataFrame(data)
        df.insert(0, "timestamp_ns", t_ns)
        df.insert(1, "segment", segment_id)
        return df

    # Case B: values are vectors like [x,y,z]
    if isinstance(first, (list, tuple, np.ndarray)) and len(first) in (2, 3, 4):
        k = len(first)
        mat = np.asarray(values, dtype=np.float64).reshape(n, k)
        colnames = ["c0", "c1", "c2", "c3"][:k]
        df = pd.DataFrame(mat, columns=colnames)
        df.insert(0, "timestamp_ns", t_ns)
        df.insert(1, "segment", segment_id)
        return df

    # Case C: scalar values
    df = pd.DataFrame({
        "timestamp_ns": t_ns,
        "segment": segment_id,
        "value": pd.to_numeric(values, errors="coerce")
    })
    return df


def peaks_to_df(peaks_time_nanos, segment_id: int) -> pd.DataFrame:
    """
    Convert peaksTimeNanos list into DataFrame with timestamp_ns + peak flag + segment.
    """
    if not peaks_time_nanos:
        return pd.DataFrame(columns=["timestamp_ns", "segment", "peak"])
    t_ns = np.asarray(peaks_time_nanos, dtype=np.int64)
    df = pd.DataFrame({"timestamp_ns": t_ns, "peak": 1})
    df.insert(1, "segment", segment_id)
    return df


def iter_avro_records(avro_path: str):
    """Yield each record in the avro file (don’t assume only one)."""
    reader = None
    try:
        reader = DataFileReader(open(avro_path, "rb"), DatumReader())
        for rec in reader:
            yield rec
    finally:
        if reader is not None:
            reader.close()


# ----------------------------
# Part 1 (DONE): parse folder -> per-channel DataFrames (with segment id)
# ----------------------------
def parse_folder_to_channel_dfs(folder: str, keys=None):
    """
    Returns: dict(signal_name -> DataFrame)
    Each DF has: timestamp_ns, datetime_utc, segment, and signal columns.
    """
    if keys is None:
        keys = ["accelerometer", "eda", "temperature", "bvp", "systolicPeaks"]

    avro_paths = sorted(glob.glob(os.path.join(folder, "*.avro")))
    buckets = {k: [] for k in keys}

    for segment_id, avro_path in enumerate(avro_paths):
        for rec in iter_avro_records(avro_path):
            raw = rec.get("rawData", {}) or {}

            for k in keys:
                if k not in raw or raw[k] is None:
                    continue

                payload = raw[k]

                if k == "systolicPeaks":
                    segments = payload if isinstance(payload, list) else [payload]
                    for seg in segments:
                        if isinstance(seg, dict):
                            df = peaks_to_df(seg.get("peaksTimeNanos", []), segment_id=segment_id)
                            if not df.empty:
                                buckets[k].append(df)
                    continue

                # sampled signals
                segments = payload if isinstance(payload, list) else [payload]
                for seg in segments:
                    if not isinstance(seg, dict):
                        continue
                    if "timestampStart" not in seg or "samplingFrequency" not in seg:
                        continue
                    df = segment_to_df(seg, signal_name=k, segment_id=segment_id)
                    if not df.empty:
                        buckets[k].append(df)

    out = {}
    for k, dfs in buckets.items():
        if not dfs:
            out[k] = pd.DataFrame(columns=["timestamp_ns", "datetime_utc", "segment"])
            print(f"[WARN] No data for {k}")
            continue

        df_all = pd.concat(dfs, ignore_index=True)
        df_all = df_all.sort_values(["timestamp_ns", "segment"], kind="mergesort")
        # Overlap handling within same timestamp: keep last by (timestamp, segment order)
        df_all = df_all.drop_duplicates(subset=["timestamp_ns"], keep="last")

        df_all.insert(1, "datetime_utc", pd.to_datetime(df_all["timestamp_ns"], unit="ns", utc=True))
        out[k] = df_all.reset_index(drop=True)

    return out


def write_channel_csvs(channel_dfs: dict, out_dir: str, prefix: str):
    os.makedirs(out_dir, exist_ok=True)
    for k, df in channel_dfs.items():
        out_path = os.path.join(out_dir, f"{prefix}_{k}.csv")
        df.to_csv(out_path, index=False)
        print(f"[OK] wrote {k}: {out_path} ({len(df)} rows)")

""" 
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("folder", default='./original_files/physiological_signal', help="Folder containing .avro files for ONE subject")
    parser.add_argument("--out_dir", default="./extraction_from_original_files", help="Where to write CSVs")
    args = parser.parse_args()

    for subject_id in range(1, 8):
        write_channel_csvs(
                parse_folder_to_channel_dfs(
                    os.path.join(args.folder, f'00{subject_id}-3YK9K1J2D2/raw_data/v6')
                ), 
                out_dir=os.path.join(args.out_dir, f'00{subject_id}'), 
                prefix=f'00{subject_id}'
            )
 """