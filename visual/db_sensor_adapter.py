#!/usr/bin/env python3
"""Load sensor data from SQLite .db files and convert to notebook-friendly DataFrames."""

from __future__ import annotations

import argparse
import base64
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from zoneinfo import ZoneInfo


@dataclass
class SensorFrames:
    spectrum: pd.DataFrame
    bme680: pd.DataFrame
    ppg: pd.DataFrame
    cheez_ppg: pd.DataFrame
    t117: pd.DataFrame
    icm: pd.DataFrame


def _connect(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(str(db_path))


def list_tables(conn: sqlite3.Connection) -> List[str]:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    return [r[0] for r in cur.fetchall()]


def table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    cur = conn.execute(f"PRAGMA table_info('{table}')")
    return [r[1] for r in cur.fetchall()]


def _lower_set(items: Iterable[str]) -> set[str]:
    return {i.lower() for i in items}


def _pick_time_col(cols: Iterable[str]) -> Optional[str]:
    candidates = ["ts", "timestamp", "time", "t"]
    lower_map = {c.lower(): c for c in cols}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    return None


def _looks_hex_string(value: str) -> bool:
    s = value.strip().lower()
    if s.startswith("0x"):
        s = s[2:]
    s = s.replace(" ", "")
    if len(s) < 2 or len(s) % 2 != 0:
        return False
    return all(c in "0123456789abcdef" for c in s)


def _decode_to_bytes(value: Any) -> Optional[bytes]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, str):
        s = value.strip()
        if _looks_hex_string(s):
            s = s[2:] if s.lower().startswith("0x") else s
            s = s.replace(" ", "")
            try:
                return bytes.fromhex(s)
            except ValueError:
                return None
        try:
            return base64.b64decode(s, validate=True)
        except Exception:
            return None
    return None


def _guess_blob_dtype(blob: bytes, count: Optional[int]) -> Optional[np.dtype]:
    if blob is None:
        return None
    byte_len = len(blob)
    candidates = [np.dtype("<i2"), np.dtype("<i4"), np.dtype("<f4"), np.dtype("<f8")]
    if count is not None and count > 0:
        for dt in candidates:
            if byte_len == count * dt.itemsize:
                return dt
        return None
    for dt in candidates:
        if byte_len % dt.itemsize == 0:
            length = byte_len // dt.itemsize
            if 8 <= length <= 4096:
                return dt
    return None


def _decode_blob(blob: bytes, count: Optional[int]) -> Tuple[Optional[np.ndarray], Optional[str]]:
    dt = _guess_blob_dtype(blob, count)
    if dt is None:
        return None, None
    arr = np.frombuffer(blob, dtype=dt)
    if count is not None and count > 0 and len(arr) != count:
        return None, None
    return arr, str(dt)


def detect_tables(conn: sqlite3.Connection) -> Dict[str, Optional[str]]:
    tables = list_tables(conn)
    info = {t: _lower_set(table_columns(conn, t)) for t in tables}

    spectrum_table = None
    for t, cols in info.items():
        if {"data", "count"}.issubset(cols) and ("ts" in cols or "timestamp" in cols):
            spectrum_table = t
            break

    ppg_table = None
    cheezppg_table = None
    bme_table = None
    t117_table = None
    icm_table = None

    for t, cols in info.items():
        t_lower = t.lower()
        if "cheezppg" in t_lower or "cheez_ppg" in t_lower:
            cheezppg_table = cheezppg_table or t
        if "raw_ppg" in t.lower():
            ppg_table = ppg_table or t
        elif "red" in cols and "ir" in cols:
            ppg_table = ppg_table or t
        if "env_bme" in t.lower():
            bme_table = bme_table or t
        elif ("pressure" in cols or "p" in cols) and ("humidity" in cols or "h" in cols):
            bme_table = bme_table or t
        if "imu_icm" in t.lower():
            icm_table = icm_table or t
        elif ("ax" in cols and "ay" in cols and "az" in cols) and ("gx" in cols and "gy" in cols and "gz" in cols):
            icm_table = icm_table or t
        elif any(c.startswith("acc") for c in cols) and (any(c.startswith("gyr") for c in cols) or any(c.startswith("gyro") for c in cols)):
            icm_table = icm_table or t
        if "env_temp" in t.lower():
            t117_table = t117_table or t
        elif "temperature" in cols or "temp" in cols or "t" in cols:
            if bme_table != t:
                t117_table = t117_table or t

    return {
        "spectrum": spectrum_table,
        "ppg": ppg_table,
        "cheez_ppg": cheezppg_table,
        "bme680": bme_table,
        "t117": t117_table,
        "icm": icm_table,
    }


def _read_table(conn: sqlite3.Connection, table: str, limit: Optional[int]) -> pd.DataFrame:
    if limit is None:
        return pd.read_sql_query(f"SELECT * FROM '{table}'", conn)
    return pd.read_sql_query(f"SELECT * FROM '{table}' LIMIT {int(limit)}", conn)


def _rename_time(df: pd.DataFrame, time_col: Optional[str]) -> pd.DataFrame:
    if time_col and time_col in df.columns and time_col != "timestamp":
        return df.rename(columns={time_col: "timestamp"})
    return df


def load_spectrum_from_db(conn: sqlite3.Connection, table: str, limit: Optional[int] = None) -> pd.DataFrame:
    df = _read_table(conn, table, limit)
    if df.empty:
        return df

    time_col = _pick_time_col(df.columns)
    if time_col is None:
        raise ValueError("Spectrum table missing time column")

    data_col = None
    for c in df.columns:
        if c.lower() == "data":
            data_col = c
            break
    if data_col is None:
        raise ValueError("Spectrum table missing data column")

    count_col = None
    for c in df.columns:
        if c.lower() == "count":
            count_col = c
            break

    decoded = []
    lengths = []
    for _, row in df.iterrows():
        blob = _decode_to_bytes(row.get(data_col))
        count = row.get(count_col) if count_col else None
        count = int(count) if count is not None and pd.notna(count) else None
        arr, _dtype = _decode_blob(blob, count)
        if arr is None:
            decoded.append(None)
            continue
        decoded.append(arr)
        lengths.append(len(arr))

    if not lengths:
        return pd.DataFrame()

    target_len = int(pd.Series(lengths).mode().iloc[0])
    valid_rows = [i for i, arr in enumerate(decoded) if arr is not None and len(arr) == target_len]

    if not valid_rows:
        return pd.DataFrame()

    mat = np.vstack([decoded[i] for i in valid_rows])
    out = pd.DataFrame(mat, columns=[f"power_{i}" for i in range(target_len)])
    out.insert(0, "num_points", target_len)
    out.insert(0, "timestamp", df.iloc[valid_rows][time_col].to_numpy())
    return out


def _normalize_sensor_df(df: pd.DataFrame, time_col: Optional[str], rename_map: Dict[str, str]) -> pd.DataFrame:
    df = _rename_time(df, time_col)
    for src, dst in rename_map.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})
    return df


def load_db_frames(db_path: Path, table_map: Optional[Dict[str, str]] = None, limit: Optional[int] = None) -> SensorFrames:
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    with _connect(db_path) as conn:
        detected = detect_tables(conn)
        if table_map:
            detected.update({k: v for k, v in table_map.items() if v})

        spectrum_df = pd.DataFrame()
        if detected.get("spectrum"):
            spectrum_df = load_spectrum_from_db(conn, detected["spectrum"], limit)

        bme_df = pd.DataFrame()
        if detected.get("bme680"):
            df = _read_table(conn, detected["bme680"], limit)
            time_col = _pick_time_col(df.columns)
            bme_df = _normalize_sensor_df(
                df,
                time_col,
                {
                    "temperature": "temperature_c",
                    "temp": "temperature_c",
                    "t": "temperature_c",
                    "pressure": "pressure_hpa",
                    "p": "pressure_hpa",
                    "humidity": "humidity_pct",
                    "h": "humidity_pct",
                },
            )

        ppg_df = pd.DataFrame()
        if detected.get("ppg"):
            df = _read_table(conn, detected["ppg"], limit)
            time_col = _pick_time_col(df.columns)
            ppg_df = _normalize_sensor_df(
                df,
                time_col,
                {
                    "heart_rate": "heart_rate_bpm",
                    "spo2": "spo2_percent",
                },
            )

        cheezppg_df = pd.DataFrame()
        if detected.get("cheez_ppg"):
            df = _read_table(conn, detected["cheez_ppg"], limit)
            df = df.rename(columns={c: c.lower() for c in df.columns})
            time_col = _pick_time_col(df.columns)
            cheezppg_df = _normalize_sensor_df(
                df,
                time_col,
                {
                    "hr": "heart_rate_bpm",
                    "hrv": "hrv_ms",
                },
            )

        t117_df = pd.DataFrame()
        if detected.get("t117"):
            df = _read_table(conn, detected["t117"], limit)
            time_col = _pick_time_col(df.columns)
            t117_df = _normalize_sensor_df(
                df,
                time_col,
                {
                    "temperature": "temperature_c",
                    "temp": "temperature_c",
                    "t": "temperature_c",
                },
            )

        icm_df = pd.DataFrame()
        if detected.get("icm"):
            df = _read_table(conn, detected["icm"], limit)
            time_col = _pick_time_col(df.columns)
            icm_df = _normalize_sensor_df(
                df,
                time_col,
                {
                    "gyro_x": "gyr_x",
                    "gyro_y": "gyr_y",
                    "gyro_z": "gyr_z",
                    "accel_x": "acc_x",
                    "accel_y": "acc_y",
                    "accel_z": "acc_z",
                    "gx": "gyr_x",
                    "gy": "gyr_y",
                    "gz": "gyr_z",
                    "ax": "acc_x",
                    "ay": "acc_y",
                    "az": "acc_z",
                },
            )

    return SensorFrames(
        spectrum=spectrum_df,
        bme680=bme_df,
        ppg=ppg_df,
        cheez_ppg=cheezppg_df,
        t117=t117_df,
        icm=icm_df,
    )


def _dedupe_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        return df
    return df.drop_duplicates(subset="timestamp", keep="first").reset_index(drop=True)


def _shift_timestamp_zero(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[float]]:
    if df.empty or "timestamp" not in df.columns:
        return df, None
    t0 = float(df["timestamp"].min())
    out = df.copy()
    out["timestamp"] = out["timestamp"] - t0
    return out, t0


def normalize_frames_to_zero(frames: SensorFrames) -> Tuple[SensorFrames, Dict[str, float]]:
    offsets: Dict[str, float] = {}
    spectrum, t0 = _shift_timestamp_zero(frames.spectrum)
    if t0 is not None:
        offsets["spectrum"] = t0
    bme680, t0 = _shift_timestamp_zero(frames.bme680)
    if t0 is not None:
        offsets["bme680"] = t0
    ppg, t0 = _shift_timestamp_zero(frames.ppg)
    if t0 is not None:
        offsets["ppg"] = t0
    cheez_ppg, t0 = _shift_timestamp_zero(frames.cheez_ppg)
    if t0 is not None:
        offsets["cheez_ppg"] = t0
    t117, t0 = _shift_timestamp_zero(frames.t117)
    if t0 is not None:
        offsets["t117"] = t0
    icm, t0 = _shift_timestamp_zero(frames.icm)
    if t0 is not None:
        offsets["icm"] = t0
    return (
        SensorFrames(
            spectrum=spectrum,
            bme680=bme680,
            ppg=ppg,
            cheez_ppg=cheez_ppg,
            t117=t117,
            icm=icm,
        ),
        offsets,
    )


def _window_by_time(df: pd.DataFrame, window_s: Optional[float], start_offset_s: float = 0.0) -> pd.DataFrame:
    if df.empty or "timestamp" not in df.columns:
        return df
    if window_s is None or window_s <= 0:
        return df
    t_min = float(df["timestamp"].min())
    t_max = float(df["timestamp"].max())
    if start_offset_s < 0:
        start_offset_s = 0.0
    t_start = t_min + start_offset_s
    if t_start >= t_max:
        return df.iloc[0:0].copy()
    if (t_max - t_min) <= window_s:
        return df
    t_end = t_start + window_s
    return df[(df["timestamp"] >= t_start) & (df["timestamp"] <= t_end)]


def _epoch_to_datetime(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    values = series.dropna()
    if values.empty:
        return pd.to_datetime(series, errors="coerce")
    scale = float(np.nanmedian(values.to_numpy()))
    if scale >= 1e14:
        unit = "ns"
    elif scale >= 1e11:
        unit = "ms"
    else:
        unit = "s"
    return pd.to_datetime(series, unit=unit, errors="coerce")


def _to_beijing_datetime(series: pd.Series) -> pd.Series:
    dt = _epoch_to_datetime(series)
    if dt.empty:
        return dt
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize("UTC", ambiguous="NaT", nonexistent="NaT")
    return dt.dt.tz_convert(ZoneInfo("Asia/Shanghai"))


def analyze_spectrum_data(df: pd.DataFrame, sample_rate_hz: float = 10.0) -> None:
    if df.empty:
        print("No data")
        return
    tspan = df["timestamp"].max() - df["timestamp"].min()
    print(f"Records: {len(df)} span {tspan:.1f}s (~{len(df)/sample_rate_hz:.1f}s expected)")
    dt = df["timestamp"].diff().dropna()
    gaps = (dt > (2 / sample_rate_hz)).sum()
    if gaps:
        print(f"Gaps >2x interval: {gaps}")
    power_cols = [c for c in df.columns if c.startswith("power_")]
    if power_cols:
        pdata = df[power_cols]
        print(
            f"Freq bins: {len(power_cols)} power range {pdata.min().min()}..{pdata.max().max()} avg {pdata.mean().mean():.1f}"
        )


def calculate_dip_frequency(df: pd.DataFrame, freq_range_ghz: Tuple[float, float] = (4, 6), n_lowest: int = 10) -> pd.DataFrame:
    power_cols = [c for c in df.columns if c.startswith("power_")]
    bin_indices = np.arange(len(power_cols))
    frequencies_ghz = 0.002 * bin_indices + 4
    freq_mask = (frequencies_ghz >= freq_range_ghz[0]) & (frequencies_ghz <= freq_range_ghz[1])
    valid_bins = bin_indices[freq_mask]
    valid_freqs = frequencies_ghz[freq_mask]
    valid_power_cols = [power_cols[i] for i in valid_bins]

    print(f"Frequency range: {freq_range_ghz[0]}-{freq_range_ghz[1]} GHz")
    print(f"Valid frequency points: {len(valid_bins)} (from Bin {valid_bins[0]} to Bin {valid_bins[-1]})")

    dip_results = []
    for _, row in df.iterrows():
        power_values = row[valid_power_cols].values
        n_to_take = min(n_lowest, len(power_values))
        lowest_indices = np.argsort(power_values)[:n_to_take]
        lowest_freqs = valid_freqs[lowest_indices]
        dip_freq = np.median(lowest_freqs)
        dip_results.append(
            {
                "timestamp": row["timestamp"],
                "dip_freq_ghz": dip_freq,
                "dip_power_mean": power_values[lowest_indices].mean(),
                "dip_power_std": power_values[lowest_indices].std(),
            }
        )

    return pd.DataFrame(dip_results)


def apply_dip_frequency_smoothing(
    dip_df: pd.DataFrame,
    smoothing_method: str = "median",
    smoothing_window: int = 5,
) -> pd.DataFrame:
    if len(dip_df) < smoothing_window:
        print(
            f"Warning: Data points ({len(dip_df)}) less than smoothing window ({smoothing_window}). Returning original data."
        )
        return dip_df

    print(f"Applying {smoothing_method} smoothing with window size {smoothing_window}...")
    dip_df_sorted = dip_df.sort_values("timestamp").copy()
    freq_values = dip_df_sorted["dip_freq_ghz"].values
    smoothed_freq = []

    for i in range(len(freq_values)):
        start_idx = max(0, i - smoothing_window // 2)
        end_idx = min(len(freq_values), start_idx + smoothing_window)
        if end_idx - start_idx < smoothing_window:
            start_idx = max(0, end_idx - smoothing_window)
        window_values = freq_values[start_idx:end_idx]
        if smoothing_method == "median":
            smoothed_value = np.median(window_values)
        else:
            smoothed_value = np.mean(window_values)
        smoothed_freq.append(smoothed_value)

    df_smoothed = dip_df_sorted.copy()
    df_smoothed["dip_freq_ghz"] = smoothed_freq
    print("Smoothing complete.")
    return df_smoothed


def plot_spectrum_overview(
    df: pd.DataFrame,
    dip_freq_smoothed: Optional[pd.DataFrame],
    Timestamp1: float = 120,
    max_points: int = 1000,
    time_dt: Optional[pd.Series] = None,
    save_name: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    if df.empty:
        print("No data")
        return None

    if len(df) > max_points:
        df_plot = df.sample(max_points).sort_values("timestamp")
    else:
        df_plot = df
    power_cols = [c for c in df_plot.columns if c.startswith("power_")]
    if not power_cols:
        print("No power columns")
        return None

    power_matrix = df_plot[power_cols].to_numpy()
    # power_matrix_db = -64.4 + ((power_matrix - 1100) / 18.7)
    power_matrix_db = -64.4 + (power_matrix * 0.0288)
    min_power_bin_indices = np.argmin(power_matrix_db, axis=1)
    min_frequency_ghz = 0.002 * min_power_bin_indices + 4
    min_power_values = np.min(power_matrix_db, axis=1)

    min_power_sequence = pd.DataFrame(
        {
            "timestamp": df_plot["timestamp"].values,
            "min_power_bin": min_power_bin_indices,
            "min_frequency": min_frequency_ghz,
            "min_power_value": min_power_values,
        }
    )

    unique, counts = np.unique(min_power_bin_indices, return_counts=True)
    mode_idx = unique[np.argmax(counts)]
    mode_count = counts.max()
    print(
        f"Most frequent min energy Frequency bin: Bin {mode_idx} (appears {mode_count} times, {mode_count/len(min_power_bin_indices)*100:.1f}%)"
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    avg_power = pd.Series(power_matrix_db.mean(axis=1), index=df_plot.index)
    if time_dt is not None:
        x_time = time_dt.loc[df_plot.index]
    else:
        x_time = df_plot["timestamp"]

    axes[0, 0].plot(x_time, avg_power)
    axes[0, 0].set_title("Avg Power")
    axes[0, 0].set_xlabel("Time (Beijing Time)" if time_dt is not None else "Time (s)")
    axes[0, 0].set_ylabel("Average Power (dB)")
    if time_dt is not None:
        axes[0, 0].xaxis.set_major_formatter(
            mdates.DateFormatter("%m-%d %H:%M", tz=ZoneInfo("Asia/Shanghai"))
        )
    axes[0, 0].grid(alpha=0.3)

    target_time = Timestamp1
    freq_min = 0
    freq_max = 1000
    idx = (df_plot["timestamp"] - target_time).abs().idxmin()
    latest = df_plot.loc[idx]

    if dip_freq_smoothed is not None and not dip_freq_smoothed.empty:
        dip_idx = (dip_freq_smoothed["timestamp"] - latest["timestamp"]).abs().idxmin()
        dip_freq = dip_freq_smoothed.loc[dip_idx, "dip_freq_ghz"]
        print(f"Latest dip frequency at t={latest.timestamp:.1f}s: {dip_freq:.4f} GHz")
    else:
        print("Dip frequency data unavailable")

    latest_power_db = power_matrix_db[df_plot.index.get_loc(idx)]
    freq_ghz = 0.002 * np.arange(len(power_cols)) + 4.0
    axes[0, 1].plot(freq_ghz, latest_power_db)
    if time_dt is not None:
        latest_dt = time_dt.loc[idx]
        axes[0, 1].set_title(f"Latest {latest_dt.strftime('%m-%d %H:%M')}")
    else:
        axes[0, 1].set_title(f"Latest t={latest.timestamp:.1f}s")
    axes[0, 1].set_xlabel("Frequency (GHz)")
    axes[0, 1].set_ylabel("Power (dB)")
    axes[0, 1].grid(alpha=0.3)

    energy_min = latest_power_db[freq_min:freq_max].min()
    energy_min_idx = power_cols[freq_min + int(np.argmin(latest_power_db[freq_min:freq_max]))]
    energy_min_freq = 0.002 * int(energy_min_idx.split("_")[1]) + 4
    print(f"\nAt time {latest.timestamp:.1f}s, frequency range {freq_min} to {freq_max}, min energy value: {energy_min}")
    print(
        f"Corresponding frequency: {energy_min_freq:.3f} GHz (Bin Index: {int(energy_min_idx.split('_')[1])})"
    )

    all_power = power_matrix_db.ravel()
    axes[1, 0].hist(all_power, bins=40, edgecolor="black")
    axes[1, 0].set_title("Distribution")
    axes[1, 0].set_xlabel("Power (dB)")
    axes[1, 0].set_ylabel("Count")

    if len(df_plot) > 1:
        mat = power_matrix_db.T
        freq_ghz_full = 0.002 * np.arange(mat.shape[0]) + 4.0
        freq_mask = (freq_ghz_full >= 4.0) & (freq_ghz_full <= 6.0)
        mat_cropped = mat[freq_mask, :]
        vmin, vmax = np.percentile(mat_cropped, [5, 95])

        x_extent = [df_plot["timestamp"].min(), df_plot["timestamp"].max()]
        if time_dt is not None:
            x_extent = [mdates.date2num(time_dt.min()), mdates.date2num(time_dt.max())]

        im = axes[1, 1].imshow(
            mat,
            aspect="auto",
            origin="lower",
            extent=[x_extent[0], x_extent[1], 4.0, 6.0],
            vmin=vmin,
            vmax=vmax,
        )
        axes[1, 1].set_title("Spectrogram")
        axes[1, 1].set_xlabel("Time (Beijing Time)" if time_dt is not None else "Time (s)")
        axes[1, 1].set_ylabel("Frequency (GHz)")
        if time_dt is not None:
            axes[1, 1].xaxis.set_major_formatter(
                mdates.DateFormatter("%m-%d %H:%M", tz=ZoneInfo("Asia/Shanghai"))
            )
        axes[1, 1].set_ylim(4.0, 6.0)
        plt.colorbar(im, ax=axes[1, 1], shrink=0.9, label="Power (dB)")

    if time_dt is not None:
        for ax in [axes[0, 0], axes[1, 1]]:
            plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment="right")

    if save_name:
        fig.savefig(save_name, dpi=150)
    plt.show()

    return min_power_sequence



def plot_cheezppg_vitals(
    cheez_df: pd.DataFrame,
    save_prefix: Optional[str] = None,
    time_dt: Optional[pd.Series] = None,
) -> None:
    if time_dt is not None:
        t_min = time_dt.min()
        t_max = time_dt.max()
    else:
        t_min = float(cheez_df["timestamp"].min())
        t_max = float(cheez_df["timestamp"].max())
    zoom_window_ppg = (t_min, t_max)
    zoom_window_hr = (t_min, t_max)
    
    use_spo2 = False
    nrows = 3 if use_spo2 else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(16, 8 + (4 if use_spo2 else 0)))
    x_vals = time_dt if time_dt is not None else cheez_df["timestamp"]
    x_ppg = np.arange(len(cheez_df)) * 0.008
    if time_dt is not None and len(time_dt) > 0:
        x_ppg = time_dt.iloc[0] + pd.to_timedelta(x_ppg, unit="s")
    if "raw" in cheez_df.columns:
        axes[0].plot(x_ppg, cheez_df["raw"], label="Raw", alpha=0.6)
    if "avg" in cheez_df.columns:
        axes[0].plot(x_ppg, cheez_df["avg"], label="Avg", alpha=0.8)
    if "filter" in cheez_df.columns:
        axes[0].plot(x_ppg, cheez_df["filter"], label="Filter", alpha=0.9)
    axes[0].set_ylabel("PPG")
    axes[0].set_title("PPG Signal")
    if time_dt is not None:
        axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M:%S", tz=ZoneInfo("Asia/Shanghai")))
        axes[0].set_xlabel("Beijing Time")
        if len(x_ppg) > 0:
            axes[0].set_xlim(x_ppg[0], x_ppg[-1])
    else:
        axes[0].set_xlim(0, max(0.0, x_ppg[-1] if len(x_ppg) else 0.0))
        axes[0].set_xlabel("Time (s)")
    axes[0].legend(loc="upper right", fontsize=10)
    axes[0].grid(True, alpha=0.3)

    if "heart_rate_bpm" in cheez_df.columns:
        hr_clean = cheez_df["heart_rate_bpm"].where(
            (cheez_df["heart_rate_bpm"] >= 50) & (cheez_df["heart_rate_bpm"] <= 100)
        )
        hr_interp = hr_clean.ffill().bfill()
        axes[1].plot(x_vals, hr_interp, color="blue", linewidth=2, label="HR (50-100, filled)")
        axes[1].set_ylabel("Heart Rate (bpm)")
        axes[1].set_xlim(zoom_window_hr[0], zoom_window_hr[1])
        axes[1].legend(loc="upper right", fontsize=10)
    else:
        axes[1].text(0.5, 0.5, "No HR column in CheezPPG", transform=axes[1].transAxes, ha="center", va="center")
    axes[1].grid(True, alpha=0.3)

    if time_dt is not None:
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M:%S", tz=ZoneInfo("Asia/Shanghai")))
        axes[1].set_xlabel("Beijing Time")
    plt.tight_layout()
    if save_prefix:
        fig.savefig(f"{save_prefix}_ppg_vitals.png", dpi=150)
    plt.show()


def plot_quicklook(frames: SensorFrames, db_name: str = "db") -> None:
    frames = SensorFrames(
        spectrum=_dedupe_timestamp(frames.spectrum),
        bme680=_dedupe_timestamp(frames.bme680),
        ppg=_dedupe_timestamp(frames.ppg),
        cheez_ppg=_dedupe_timestamp(frames.cheez_ppg),
        t117=_dedupe_timestamp(frames.t117),
        icm=_dedupe_timestamp(frames.icm),
    )

    frames, offsets = normalize_frames_to_zero(frames)
    if offsets:
        print(f"Timestamp zero offsets (seconds): {offsets}")

    ppg_plot = _window_by_time(frames.ppg, window_s=30.0, start_offset_s=60.0)
    cheez_plot = _window_by_time(frames.cheez_ppg, window_s=30.0, start_offset_s=60.0)
    icm_plot = _window_by_time(frames.icm, window_s=30.0, start_offset_s=60.0)

    if not frames.spectrum.empty:
        analyze_spectrum_data(frames.spectrum)
        dip_freq_df = calculate_dip_frequency(frames.spectrum, freq_range_ghz=(4, 6), n_lowest=10)
        dip_freq_smoothed = apply_dip_frequency_smoothing(dip_freq_df, "median", 50)
        dip_freq_smoothed = apply_dip_frequency_smoothing(dip_freq_smoothed, "median", 100)
        timestamp1 = float(frames.spectrum["timestamp"].median())
        spectrum_time = frames.spectrum["timestamp"] + offsets.get("spectrum", 0.0)
        spectrum_dt = _to_beijing_datetime(spectrum_time)
        plot_spectrum_overview(
            frames.spectrum,
            dip_freq_smoothed,
            Timestamp1=timestamp1,
            max_points=10000,
            time_dt=spectrum_dt,
            save_name=f"spectrum_{db_name}.png",
        )

    if not frames.bme680.empty and {"timestamp", "temperature_c"}.issubset(frames.bme680.columns):
        bme_time = frames.bme680["timestamp"] + offsets.get("bme680", 0.0)
        bme_dt = _to_beijing_datetime(bme_time)
        fig = plt.figure(figsize=(12, 4))
        ax = plt.gca()
        ax.plot(bme_dt, frames.bme680["temperature_c"], label="BME680 Temp")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=ZoneInfo("Asia/Shanghai")))
        plt.title("BME680 Temperature")
        plt.xlabel("Beijing Time")
        plt.ylabel("C")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        fig.autofmt_xdate()
        fig.savefig(f"bme680_{db_name}.png", dpi=150)
        plt.show()

    if not frames.t117.empty and {"timestamp", "temperature_c"}.issubset(frames.t117.columns):
        t117_time = frames.t117["timestamp"] + offsets.get("t117", 0.0)
        t117_dt = _to_beijing_datetime(t117_time)
        fig = plt.figure(figsize=(12, 4))
        ax = plt.gca()
        ax.plot(t117_dt, frames.t117["temperature_c"], label="T117 Temp")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=ZoneInfo("Asia/Shanghai")))
        plt.title("T117 Temperature")
        plt.xlabel("Beijing Time")
        plt.ylabel("C")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        fig.autofmt_xdate()
        fig.savefig(f"t117_{db_name}.png", dpi=150)
        plt.show()

    has_cheez_hr = not cheez_plot.empty and "heart_rate_bpm" in cheez_plot.columns

    if not cheez_plot.empty and "timestamp" in cheez_plot.columns:
        cheez_time = cheez_plot["timestamp"] + offsets.get("cheez_ppg", 0.0)
        cheez_dt = _to_beijing_datetime(cheez_time)
        # plot_cheezppg_vitals(cheez_plot, save_prefix=f"ppg_{db_name}", spo2_df=processed_spo2)
        plot_cheezppg_vitals(cheez_plot, save_prefix=f"ppg_{db_name}", time_dt=cheez_dt)

    if not icm_plot.empty and {"timestamp", "acc_x", "acc_y", "acc_z"}.issubset(icm_plot.columns):
        icm_time = icm_plot["timestamp"] + offsets.get("icm", 0.0)
        icm_dt = _to_beijing_datetime(icm_time)
        fig = plt.figure(figsize=(12, 4))
        ax = plt.gca()
        ax.plot(icm_dt, icm_plot["acc_x"], label="acc_x")
        ax.plot(icm_dt, icm_plot["acc_y"], label="acc_y")
        ax.plot(icm_dt, icm_plot["acc_z"], label="acc_z")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M:%S", tz=ZoneInfo("Asia/Shanghai")))
        plt.title("ICM Accel")
        plt.xlabel("Beijing Time")
        plt.ylabel("Acceleration")
        plt.legend(loc="upper right", fontsize=9)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        fig.autofmt_xdate()
        fig.savefig(f"icm_{db_name}_accel.png", dpi=150)
        plt.show()

    if not icm_plot.empty and {"timestamp", "gyr_x", "gyr_y", "gyr_z"}.issubset(icm_plot.columns):
        icm_time = icm_plot["timestamp"] + offsets.get("icm", 0.0)
        icm_dt = _to_beijing_datetime(icm_time)
        fig = plt.figure(figsize=(12, 4))
        ax = plt.gca()
        ax.plot(icm_dt, icm_plot["gyr_x"], label="gyr_x")
        ax.plot(icm_dt, icm_plot["gyr_y"], label="gyr_y")
        ax.plot(icm_dt, icm_plot["gyr_z"], label="gyr_z")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M:%S", tz=ZoneInfo("Asia/Shanghai")))
        plt.title("ICM Gyro")
        plt.xlabel("Beijing Time")
        plt.ylabel("Angular Velocity")
        plt.legend(loc="upper right", fontsize=9)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        fig.autofmt_xdate()
        fig.savefig(f"icm_{db_name}_gyro.png", dpi=150)
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SQLite sensor DB to notebook-ready DataFrames.")
    parser.add_argument("--db", required=True, help="Path to .db file")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for each table")
    parser.add_argument("--plot", action="store_true", help="Render quicklook plots")
    args = parser.parse_args()

    db_path = Path(args.db)
    frames = load_db_frames(db_path, limit=args.limit)
    # print(frames.ppg["timestamp"].min(), frames.ppg["timestamp"].max())

    print("Loaded frames:")
    print(f"- cheez_ppg: {len(frames.cheez_ppg)} rows, {len(frames.cheez_ppg.columns)} cols")
    print(f"- spectrum: {len(frames.spectrum)} rows, {len(frames.spectrum.columns)} cols")
    print(f"- bme680:  {len(frames.bme680)} rows, {len(frames.bme680.columns)} cols")
    print(f"- ppg:     {len(frames.ppg)} rows, {len(frames.ppg.columns)} cols")
    print(f"- t117:    {len(frames.t117)} rows, {len(frames.t117.columns)} cols")
    print(f"- icm:     {len(frames.icm)} rows, {len(frames.icm.columns)} cols")

    if args.plot:
        plot_quicklook(frames, db_name=db_path.stem)


if __name__ == "__main__":
    main()
