#!/usr/bin/env python3
"""Self-contained diagnostics for Tao_db distribution and split shift.

Outputs:
- sample_table.csv / experiment_audit_table.csv
- audit_*.png
- split_*_overview.png + split_shift_metrics.csv
- phase_mean_spectra.png + phase_effect_curve_rise_vs_fall.png
- cluster_pca_scatter.png + cluster_vs_*.csv
- diagnostic_summary.json
"""

from __future__ import annotations

import argparse
import base64
import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    sns = None

try:
    from scipy.interpolate import Akima1DInterpolator, PchipInterpolator, interp1d
except ImportError:
    Akima1DInterpolator = None
    PchipInterpolator = None
    interp1d = None

try:
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import energy_distance, ks_2samp, wasserstein_distance
except ImportError:
    jensenshannon = None
    energy_distance = None
    ks_2samp = None
    wasserstein_distance = None


PHASE_LABELS = ("rise", "flat", "fall")


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _beijing_to_utc_timestamp(beijing_time_str: str) -> float:
    dt = pd.to_datetime(beijing_time_str)
    utc_dt = dt - pd.Timedelta(hours=8)
    return float(utc_dt.timestamp())


def _ecdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vals = np.sort(values)
    ys = np.arange(1, len(vals) + 1, dtype=float) / max(len(vals), 1)
    return vals, ys


def _compute_psi(train: np.ndarray, test: np.ndarray, bins: int = 20) -> float:
    all_values = np.concatenate([train, test])
    edges = np.quantile(all_values, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0
    train_hist, _ = np.histogram(train, bins=edges)
    test_hist, _ = np.histogram(test, bins=edges)
    train_ratio = train_hist / max(train_hist.sum(), 1)
    test_ratio = test_hist / max(test_hist.sum(), 1)
    eps = 1e-6
    train_ratio = np.clip(train_ratio, eps, None)
    test_ratio = np.clip(test_ratio, eps, None)
    return float(np.sum((test_ratio - train_ratio) * np.log(test_ratio / train_ratio)))


def _scale_features(x: np.ndarray) -> np.ndarray:
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    return (x - mean) / (std + 1e-12)


def _pca_2d(x: np.ndarray) -> np.ndarray:
    x_centered = x - np.mean(x, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x_centered, full_matrices=False)
    comp = vt[:2].T
    return x_centered @ comp


def _kmeans_fit_predict(x: np.ndarray, n_clusters: int, seed: int, n_init: int = 12) -> np.ndarray:
    rng = np.random.default_rng(seed)
    best_labels = None
    best_inertia = np.inf

    for _ in range(max(n_init, 1)):
        init_idx = rng.choice(len(x), size=n_clusters, replace=False)
        centers = x[init_idx].copy()
        labels = np.zeros(len(x), dtype=int)

        for _iter in range(60):
            d2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = np.argmin(d2, axis=1)

            new_centers = centers.copy()
            for k in range(n_clusters):
                members = x[labels == k]
                if len(members) > 0:
                    new_centers[k] = np.mean(members, axis=0)
            if np.allclose(new_centers, centers, atol=1e-6):
                break
            centers = new_centers

        inertia = float(np.sum((x - centers[labels]) ** 2))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()

    return best_labels


def _silhouette_score_approx(x: np.ndarray, labels: np.ndarray, max_eval: int = 2000, seed: int = 42) -> float:
    """Approx silhouette without sklearn, sampled for O(n^2) control."""
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return float("nan")

    rng = np.random.default_rng(seed)
    n = len(x)
    if n > max_eval:
        idx = np.sort(rng.choice(n, size=max_eval, replace=False))
        x = x[idx]
        labels = labels[idx]

    d = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2) + 1e-12)
    s_vals = []
    for i in range(len(x)):
        same = labels == labels[i]
        other_clusters = [c for c in np.unique(labels) if c != labels[i]]

        if np.sum(same) > 1:
            a = np.mean(d[i, same & (np.arange(len(x)) != i)])
        else:
            a = 0.0

        b = np.inf
        for c in other_clusters:
            c_mask = labels == c
            if np.any(c_mask):
                b = min(b, float(np.mean(d[i, c_mask])))

        denom = max(a, b)
        s = (b - a) / denom if denom > 0 else 0.0
        s_vals.append(s)

    return float(np.mean(s_vals))


def _looks_hex_string(value: str) -> bool:
    s = value.strip().lower()
    if s.startswith("0x"):
        s = s[2:]
    s = s.replace(" ", "")
    if len(s) < 2 or len(s) % 2 != 0:
        return False
    return all(c in "0123456789abcdef" for c in s)


def _decode_to_bytes(value) -> Optional[bytes]:
    if value is None:
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


def _resolve_spectrum_paths(user_root: Path, exp_cfg: dict) -> List[Path]:
    if exp_cfg.get("db_files"):
        out = []
        for p in exp_cfg["db_files"]:
            path = Path(p)
            if not path.is_absolute():
                path = user_root / path
            out.append(path)
        return out

    db_path = Path(exp_cfg["db_path"])
    if not db_path.is_absolute():
        db_path = user_root / db_path

    if db_path.is_dir():
        files = sorted(db_path.glob("*.db"))
        if files:
            return files

    if db_path.suffix != ".db":
        db_path = db_path.with_suffix(".db")

    if db_path.exists():
        return [db_path]

    fallback = db_path.parent / db_path.stem / db_path.name
    if fallback.exists():
        return [fallback]

    return [db_path]


def _load_spectrum_from_single_db(db_path: Path, start_ts: float, end_ts: float) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()

    conn = sqlite3.connect(str(db_path))
    try:
        cols = pd.read_sql_query("PRAGMA table_info('spectrum')", conn)
        col_names = set(cols["name"].tolist()) if not cols.empty else set()
        if "spectrum" not in pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)["name"].tolist():
            return pd.DataFrame()

        if "count" in col_names:
            q = (
                "SELECT ts, data, count FROM spectrum "
                f"WHERE ts >= {start_ts} AND ts <= {end_ts} ORDER BY ts"
            )
            df = pd.read_sql_query(q, conn)
        else:
            q = (
                "SELECT ts, data FROM spectrum "
                f"WHERE ts >= {start_ts} AND ts <= {end_ts} ORDER BY ts"
            )
            df = pd.read_sql_query(q, conn)
    finally:
        conn.close()

    if df.empty:
        return pd.DataFrame()

    decoded = []
    lengths = []
    for _, row in df.iterrows():
        blob = _decode_to_bytes(row.get("data"))
        if blob is None:
            decoded.append(None)
            continue
        arr = np.frombuffer(blob, dtype=np.uint32)
        if len(arr) == 0:
            decoded.append(None)
            continue
        if "count" in row and pd.notna(row.get("count")):
            expected = int(row["count"])
            if expected > 0 and expected != len(arr):
                decoded.append(None)
                continue
        decoded.append(arr)
        lengths.append(len(arr))

    if not lengths:
        return pd.DataFrame()

    target_len = int(pd.Series(lengths).mode().iloc[0])
    valid_idx = [i for i, arr in enumerate(decoded) if arr is not None and len(arr) == target_len]
    if not valid_idx:
        return pd.DataFrame()

    mat = np.vstack([decoded[i] for i in valid_idx])
    out = pd.DataFrame(mat, columns=[f"power_{i}" for i in range(target_len)])
    out.insert(0, "timestamp", df.iloc[valid_idx]["ts"].to_numpy())
    return out.sort_values("timestamp").reset_index(drop=True)


def _load_spectrum_from_dbs(db_paths: Sequence[Path], start_ts: float, end_ts: float) -> pd.DataFrame:
    dfs = []
    for p in db_paths:
        s = _load_spectrum_from_single_db(p, start_ts, end_ts)
        if not s.empty:
            dfs.append(s)
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    return out


def _load_glucose_from_db(glucose_db: Path, start_ts: float, end_ts: float) -> pd.DataFrame:
    if not glucose_db.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(str(glucose_db))
    try:
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
        if "blood_sugar" not in tables["name"].tolist():
            return pd.DataFrame()
        q = (
            "SELECT create_time as ts, blood_sugar as glucose FROM blood_sugar "
            f"WHERE create_time >= {start_ts} AND create_time <= {end_ts} ORDER BY create_time"
        )
        df = pd.read_sql_query(q, conn)
    finally:
        conn.close()
    return df


def _load_glucose_from_json(json_path: Path, start_ts: float, end_ts: float) -> pd.DataFrame:
    if not json_path.exists():
        return pd.DataFrame()
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict) or "times" not in obj or "values" not in obj:
        return pd.DataFrame()
    df = pd.DataFrame({"time_str": obj["times"], "glucose": obj["values"]})
    dt = pd.to_datetime(df["time_str"])
    ts = (dt - pd.Timedelta(hours=8)).astype("int64") // 10**9
    df = pd.DataFrame({"ts": ts.astype(float), "glucose": df["glucose"].astype(float)})
    df = df[(df["ts"] >= start_ts) & (df["ts"] <= end_ts)]
    return df.sort_values("ts").reset_index(drop=True)


def _align_spectrum_glucose(
    spectrum_df: pd.DataFrame,
    glucose_df: pd.DataFrame,
    interpolation: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    power_cols = [c for c in spectrum_df.columns if c.startswith("power_")]
    spectra = spectrum_df[power_cols].to_numpy(dtype=np.float32)
    ts = spectrum_df["timestamp"].to_numpy(dtype=float)

    g_ts = glucose_df["ts"].to_numpy(dtype=float)
    g_val = glucose_df["glucose"].to_numpy(dtype=float)
    order = np.argsort(g_ts)
    g_ts = g_ts[order]
    g_val = g_val[order]

    uniq_mask = np.concatenate([[True], np.diff(g_ts) > 0])
    g_ts = g_ts[uniq_mask]
    g_val = g_val[uniq_mask]

    if len(g_ts) < 2:
        raise ValueError("Not enough glucose points for interpolation")

    if interpolation == "pchip" and PchipInterpolator is not None and len(g_ts) >= 3:
        f = PchipInterpolator(g_ts, g_val, extrapolate=True)
        labels = f(ts)
    elif interpolation == "akima" and Akima1DInterpolator is not None and len(g_ts) >= 5:
        f = Akima1DInterpolator(g_ts, g_val)
        labels = f(ts, extrapolate=True)
    elif interpolation == "cubic" and interp1d is not None and len(g_ts) >= 4:
        f = interp1d(g_ts, g_val, kind="cubic", bounds_error=False, fill_value="extrapolate")
        labels = f(ts)
    elif interpolation == "nearest" and interp1d is not None:
        f = interp1d(g_ts, g_val, kind="nearest", bounds_error=False, fill_value="extrapolate")
        labels = f(ts)
    elif interp1d is not None:
        f = interp1d(g_ts, g_val, kind="linear", bounds_error=False, fill_value="extrapolate")
        labels = f(ts)
    else:
        labels = np.interp(ts, g_ts, g_val, left=g_val[0], right=g_val[-1])

    return spectra, labels.astype(np.float32), ts


def _nearest_alignment_errors(query_ts: np.ndarray, ref_ts: np.ndarray) -> np.ndarray:
    if len(query_ts) == 0 or len(ref_ts) == 0:
        return np.zeros(0, dtype=float)
    ref_sorted = np.sort(ref_ts)
    pos = np.searchsorted(ref_sorted, query_ts)
    left = np.clip(pos - 1, 0, len(ref_sorted) - 1)
    right = np.clip(pos, 0, len(ref_sorted) - 1)
    return np.minimum(np.abs(query_ts - ref_sorted[left]), np.abs(query_ts - ref_sorted[right]))


def _derive_phase(
    glucose: np.ndarray,
    timestamps: np.ndarray,
    smooth_window: int,
    phase_quantile: float,
    min_phase_rate: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if smooth_window < 3:
        smooth_window = 3
    if smooth_window % 2 == 0:
        smooth_window += 1

    smooth = (
        pd.Series(glucose)
        .rolling(window=smooth_window, min_periods=1, center=True)
        .median()
        .to_numpy()
    )
    dgdt = np.gradient(smooth, timestamps) * 60.0
    thr = max(float(np.quantile(np.abs(dgdt), phase_quantile)), float(min_phase_rate))

    phase = np.full(len(dgdt), "flat", dtype=object)
    phase[dgdt > thr] = "rise"
    phase[dgdt < -thr] = "fall"
    return smooth, dgdt, phase, thr


def _cohens_d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(x) < 2 or len(y) < 2:
        return np.zeros(x.shape[1], dtype=float)
    x_mu = np.mean(x, axis=0)
    y_mu = np.mean(y, axis=0)
    x_var = np.var(x, axis=0, ddof=1)
    y_var = np.var(y, axis=0, ddof=1)
    pooled = np.sqrt(((len(x) - 1) * x_var + (len(y) - 1) * y_var) / (len(x) + len(y) - 2 + 1e-12))
    return (x_mu - y_mu) / (pooled + 1e-12)


def _build_group_split(groups: Iterable[str], train_ratio: float) -> Dict[str, str]:
    unique = list(dict.fromkeys(groups))
    if len(unique) <= 1:
        return {g: "train" for g in unique}
    n_train = int(np.ceil(len(unique) * train_ratio))
    n_train = min(max(n_train, 1), len(unique) - 1)
    train_set = set(unique[:n_train])
    return {g: ("train" if g in train_set else "test") for g in unique}


def _build_date_group_split_three_way(
    sample_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, str]:
    date_order = (
        sample_df[["date", "timestamp_utc"]]
        .groupby("date", as_index=False)["timestamp_utc"]
        .min()
        .sort_values("timestamp_utc")
    )
    dates = date_order["date"].tolist()
    n_dates = len(dates)
    if n_dates == 0:
        return {}
    if n_dates == 1:
        return {dates[0]: "train"}
    if n_dates == 2:
        return {dates[0]: "train", dates[1]: "test"}

    n_train = int(np.floor(n_dates * train_ratio))
    n_val = int(np.floor(n_dates * val_ratio))
    n_train = max(1, n_train)
    n_val = max(0, n_val)

    if n_train + n_val >= n_dates:
        n_val = max(0, n_dates - n_train - 1)
    if n_train <= 0:
        n_train = 1
    if n_train + n_val >= n_dates:
        n_train = n_dates - 1
        n_val = 0

    mapper: Dict[str, str] = {}
    for i, d in enumerate(dates):
        if i < n_train:
            mapper[d] = "train"
        elif i < n_train + n_val:
            mapper[d] = "val"
        else:
            mapper[d] = "test"
    return mapper


def _build_split_labels(
    sample_df: pd.DataFrame,
    strategy: str,
    train_ratio: float,
    seed: int,
    n_alternating_segments: int,
) -> np.ndarray:
    labels = np.full(len(sample_df), "test", dtype=object)

    if strategy == "random":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(sample_df))
        cut = int(np.ceil(len(sample_df) * train_ratio))
        cut = min(max(cut, 1), len(sample_df) - 1)
        labels[perm[:cut]] = "train"
        return labels

    if strategy == "date":
        date_order = (
            sample_df[["date", "timestamp_utc"]]
            .groupby("date", as_index=False)["timestamp_utc"]
            .min()
            .sort_values("timestamp_utc")
        )
        mapper = _build_group_split(date_order["date"].tolist(), train_ratio)
        return sample_df["date"].map(mapper).to_numpy()

    if strategy == "experiment":
        exp_order = (
            sample_df[["experiment_name", "timestamp_utc"]]
            .groupby("experiment_name", as_index=False)["timestamp_utc"]
            .min()
            .sort_values("timestamp_utc")
        )
        mapper = _build_group_split(exp_order["experiment_name"].tolist(), train_ratio)
        return sample_df["experiment_name"].map(mapper).to_numpy()

    if strategy == "alternating":
        seg_n = max(int(n_alternating_segments), 2)
        for _, exp_df in sample_df.groupby("experiment_name"):
            exp_df = exp_df.sort_values("timestamp_utc")
            n = len(exp_df)
            if n < 4:
                labels[exp_df.index] = "train"
                continue
            seg_ids = np.floor(np.linspace(0, seg_n, n, endpoint=False)).astype(int)
            seg_ids = np.clip(seg_ids, 0, seg_n - 1)
            labels[exp_df.index] = np.where(seg_ids % 2 == 0, "train", "test")
        return labels

    raise ValueError(f"Unsupported strategy: {strategy}")


def _extract_band_features(spectrum: np.ndarray, n_bands: int = 24) -> np.ndarray:
    n_points = spectrum.shape[1]
    edges = np.linspace(0, n_points, n_bands + 1, dtype=int)
    feats = []
    for i in range(n_bands):
        left, right = edges[i], edges[i + 1]
        band = spectrum[:, left:right]
        feats.append(np.mean(band, axis=1, keepdims=True))
        feats.append(np.std(band, axis=1, keepdims=True))
    return np.hstack(feats)


def _safe_stat_distance(train_vals: np.ndarray, other_vals: np.ndarray) -> Dict[str, float]:
    out = {
        "psi": float(_compute_psi(train_vals, other_vals)),
        "wasserstein": float("nan"),
        "ks_stat": float("nan"),
        "ks_pvalue": float("nan"),
    }
    if wasserstein_distance is not None:
        out["wasserstein"] = float(wasserstein_distance(train_vals, other_vals))
    if ks_2samp is not None:
        ks = ks_2samp(train_vals, other_vals)
        out["ks_stat"] = float(ks.statistic)
        out["ks_pvalue"] = float(ks.pvalue)
    return out


def build_master_table_with_date_split(
    sample_df: pd.DataFrame,
    spectra: np.ndarray,
    train_ratio: float,
    val_ratio: float,
) -> pd.DataFrame:
    out = sample_df.copy()
    out = out.reset_index(drop=True)
    out["sample_id"] = np.arange(len(out), dtype=int)
    out["date_id"] = out["date"].astype(str)
    out["experiment_id"] = out["experiment_name"].astype(str)
    out["phase_label"] = out["phase"].astype(str)

    split_map = _build_date_group_split_three_way(out, train_ratio=train_ratio, val_ratio=val_ratio)
    out["split_label"] = out["date"].map(split_map).fillna("test")

    n_spec = spectra.shape[1]
    c1 = n_spec // 3
    c2 = 2 * n_spec // 3
    out["spec_total_power"] = np.mean(spectra, axis=1)
    out["spec_low_power"] = np.mean(spectra[:, :c1], axis=1)
    out["spec_mid_power"] = np.mean(spectra[:, c1:c2], axis=1)
    out["spec_high_power"] = np.mean(spectra[:, c2:], axis=1)
    out["date_order"] = pd.to_datetime(out["date"]).rank(method="dense").astype(int)
    return out


def run_date_distribution_diagnostics(
    master_df: pd.DataFrame,
    out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        master_df.groupby(["date_id", "split_label"], as_index=False)
        .agg(
            n_samples=("sample_id", "count"),
            glucose_min=("glucose_interp", "min"),
            glucose_max=("glucose_interp", "max"),
            glucose_mean=("glucose_interp", "mean"),
            glucose_std=("glucose_interp", "std"),
            dgdt_mean=("dgdt_mmol_per_min", "mean"),
            dgdt_std=("dgdt_mmol_per_min", "std"),
            spec_total_mean=("spec_total_power", "mean"),
            spec_total_std=("spec_total_power", "std"),
        )
        .sort_values(["date_id", "split_label"])
        .reset_index(drop=True)
    )

    phase_counts = (
        master_df.groupby(["date_id", "split_label", "phase_label"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    phase_tot = phase_counts.groupby(["date_id", "split_label"], as_index=False)["count"].sum()
    phase_ratio = phase_counts.merge(phase_tot, on=["date_id", "split_label"], suffixes=("", "_tot"))
    phase_ratio["ratio"] = phase_ratio["count"] / np.maximum(phase_ratio["count_tot"], 1)

    # Global train-vs-val/test distance metrics for key variables
    dist_rows: List[dict] = []
    train = master_df[master_df["split_label"] == "train"]
    for tgt_split in ("val", "test"):
        tgt = master_df[master_df["split_label"] == tgt_split]
        if train.empty or tgt.empty:
            continue
        for var in [
            "glucose_interp",
            "dgdt_mmol_per_min",
            "spec_total_power",
            "spec_low_power",
            "spec_mid_power",
            "spec_high_power",
        ]:
            d = _safe_stat_distance(train[var].to_numpy(), tgt[var].to_numpy())
            dist_rows.append(
                {
                    "scope": "global",
                    "target_split": tgt_split,
                    "variable": var,
                    **d,
                }
            )

    # Date-level distance metrics: each non-train date vs train pool
    non_train_dates = master_df.loc[master_df["split_label"] != "train", "date_id"].unique().tolist()
    for d_id in non_train_dates:
        tgt = master_df[master_df["date_id"] == d_id]
        if train.empty or tgt.empty:
            continue
        for var in ["glucose_interp", "dgdt_mmol_per_min", "spec_total_power"]:
            d = _safe_stat_distance(train[var].to_numpy(), tgt[var].to_numpy())
            dist_rows.append(
                {
                    "scope": "by_date",
                    "target_split": str(tgt["split_label"].iloc[0]),
                    "target_date": d_id,
                    "variable": var,
                    **d,
                }
            )

    dist_df = pd.DataFrame(dist_rows)

    # Plot 1: glucose distribution per date (boxplot-like using violin for robustness)
    fig, ax = plt.subplots(figsize=(12, 5))
    order = sorted(master_df["date_id"].unique().tolist())
    if sns is not None:
        sns.boxplot(
            data=master_df,
            x="date_id",
            y="glucose_interp",
            hue="split_label",
            order=order,
            ax=ax,
            showfliers=False,
        )
    else:
        split_colors = {"train": "#2E86AB", "val": "#E9C46A", "test": "#F26419"}
        for i, date_id in enumerate(order):
            dd = master_df[master_df["date_id"] == date_id]
            for j, split_label in enumerate(["train", "val", "test"]):
                vals = dd.loc[dd["split_label"] == split_label, "glucose_interp"].to_numpy()
                if len(vals) == 0:
                    continue
                pos = i + (j - 1) * 0.22
                ax.boxplot(vals, positions=[pos], widths=0.18, patch_artist=True, showfliers=False,
                           boxprops={"facecolor": split_colors[split_label], "alpha": 0.6})
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_title("Date-wise Glucose Distribution by Split")
    ax.set_xlabel("Date")
    ax.set_ylabel("Glucose (mmol/L)")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(out_dir / "phase1_date_split_glucose_box.png", dpi=220)
    plt.close(fig)

    # Plot 2: phase ratio heatmap by date/split
    phase_pivot = (
        phase_ratio.pivot_table(
            index=["date_id", "split_label"],
            columns="phase_label",
            values="ratio",
            fill_value=0.0,
        )
        .reindex(columns=list(PHASE_LABELS), fill_value=0.0)
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(phase_pivot))))
    if sns is not None:
        sns.heatmap(
            phase_pivot,
            cmap="YlOrBr",
            annot=True,
            fmt=".2f",
            linewidths=0.4,
            cbar_kws={"label": "Phase ratio"},
            ax=ax,
        )
    else:
        arr = phase_pivot.to_numpy()
        im = ax.imshow(arr, cmap="YlOrBr", aspect="auto", vmin=0.0, vmax=1.0)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Phase ratio")
        ax.set_xticks(np.arange(len(phase_pivot.columns)))
        ax.set_xticklabels(phase_pivot.columns)
        ax.set_yticks(np.arange(len(phase_pivot.index)))
        ax.set_yticklabels([f"{d}|{s}" for d, s in phase_pivot.index])
        for r in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                ax.text(c, r, f"{arr[r, c]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Phase Composition by Date and Split")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Date | Split")
    plt.tight_layout()
    plt.savefig(out_dir / "phase1_date_split_phase_heatmap.png", dpi=220)
    plt.close(fig)

    # Plot 3: ECDF train/val/test for glucose
    fig, ax = plt.subplots(figsize=(8, 5))
    for split_label, color in (("train", "#2E86AB"), ("val", "#E9C46A"), ("test", "#F26419")):
        vals = master_df.loc[master_df["split_label"] == split_label, "glucose_interp"].to_numpy()
        if len(vals) == 0:
            continue
        xs, ys = _ecdf(vals)
        ax.plot(xs, ys, label=split_label, color=color)
    ax.set_title("Global Glucose ECDF by Split")
    ax.set_xlabel("Glucose (mmol/L)")
    ax.set_ylabel("ECDF")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "phase2_global_glucose_ecdf_by_split.png", dpi=220)
    plt.close(fig)

    summary.to_csv(out_dir / "phase1_date_split_summary.csv", index=False)
    phase_ratio.to_csv(out_dir / "phase1_date_split_phase_ratio.csv", index=False)
    dist_df.to_csv(out_dir / "phase2_train_vs_nontrain_distance.csv", index=False)
    return summary, dist_df


def load_tao_samples(
    dataset_root: Path,
    db_user: str,
    interpolation: str,
    smooth_window: int,
    phase_quantile: float,
    min_phase_rate: float,
    max_experiments: int,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    user_root = dataset_root / db_user
    config_path = user_root / "experiments_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        exp_cfgs = json.load(f)

    if max_experiments > 0:
        exp_cfgs = exp_cfgs[:max_experiments]

    sample_rows: List[pd.DataFrame] = []
    spectra_list: List[np.ndarray] = []
    audit_rows: List[dict] = []

    for cfg in exp_cfgs:
        exp_name = cfg["experiment_name"]
        start_ts = _beijing_to_utc_timestamp(cfg["start_time"])
        end_ts = _beijing_to_utc_timestamp(cfg["end_time"])

        spectrum_paths = _resolve_spectrum_paths(user_root, cfg)
        spectrum_df = _load_spectrum_from_dbs(spectrum_paths, start_ts, end_ts)
        if spectrum_df.empty:
            print(f"[WARN] Skip {exp_name}: spectrum empty")
            continue

        glucose_df = pd.DataFrame()
        glucose_db_path = cfg.get("glucose_db_path")
        if glucose_db_path:
            gdb = Path(glucose_db_path)
            if not gdb.is_absolute():
                gdb = user_root / gdb
            glucose_df = _load_glucose_from_db(gdb, start_ts, end_ts)

        if glucose_df.empty and cfg.get("json_path"):
            jpath = Path(cfg["json_path"])
            if not jpath.is_absolute():
                jpath = user_root / jpath
            glucose_df = _load_glucose_from_json(jpath, start_ts, end_ts)

        if glucose_df.empty:
            print(f"[WARN] Skip {exp_name}: glucose empty")
            continue

        spectra, labels, ts = _align_spectrum_glucose(spectrum_df, glucose_df, interpolation)
        smooth, dgdt, phase, threshold = _derive_phase(
            labels, ts, smooth_window, phase_quantile, min_phase_rate
        )
        align_err = _nearest_alignment_errors(ts, glucose_df["ts"].to_numpy(dtype=float))

        rows = pd.DataFrame(
            {
                "user": db_user,
                "experiment_name": exp_name,
                "date": pd.to_datetime(cfg["start_time"]).strftime("%Y-%m-%d"),
                "timestamp_utc": ts,
                "timestamp_bj": pd.to_datetime(ts, unit="s") + pd.Timedelta(hours=8),
                "glucose_interp": labels,
                "glucose_smooth": smooth,
                "dgdt_mmol_per_min": dgdt,
                "phase": phase,
                "align_error_sec": align_err,
            }
        )

        sample_rows.append(rows)
        spectra_list.append(spectra)

        spec_dt = np.diff(np.sort(ts)) if len(ts) > 1 else np.array([np.nan])
        glu_dt = (
            np.diff(np.sort(glucose_df["ts"].to_numpy(dtype=float)))
            if len(glucose_df) > 1
            else np.array([np.nan])
        )

        audit_rows.append(
            {
                "experiment_name": exp_name,
                "start_time": cfg["start_time"],
                "end_time": cfg["end_time"],
                "n_spectrum": int(len(ts)),
                "n_glucose": int(len(glucose_df)),
                "phase_threshold": float(threshold),
                "spec_dt_median_sec": float(np.nanmedian(spec_dt)),
                "glu_dt_median_sec": float(np.nanmedian(glu_dt)),
                "align_error_median_sec": float(np.median(align_err)),
                "align_error_p95_sec": float(np.quantile(align_err, 0.95)),
                "glucose_min": float(np.min(labels)),
                "glucose_max": float(np.max(labels)),
                "glucose_mean": float(np.mean(labels)),
            }
        )

        print(
            f"[OK] {exp_name}: n={len(ts)}, glucose=[{np.min(labels):.2f}, {np.max(labels):.2f}], "
            f"align_p95={np.quantile(align_err, 0.95):.1f}s"
        )

    if not sample_rows:
        raise RuntimeError("No valid experiment loaded. Check config and DB files.")

    sample_df = pd.concat(sample_rows, ignore_index=True)
    spectra = np.vstack(spectra_list)
    audit_df = pd.DataFrame(audit_rows).sort_values("start_time").reset_index(drop=True)
    return sample_df, spectra, audit_df


def plot_audit(sample_df: pd.DataFrame, audit_df: pd.DataFrame, out_dir: Path) -> None:
    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except OSError:
        plt.style.use("ggplot")
    if sns is not None:
        sns.set_palette("deep")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sample_df["align_error_sec"].to_numpy(), bins=60, color="#4C72B0", alpha=0.85)
    ax.set_title("Alignment Error Distribution (Spectrum vs Nearest Glucose)")
    ax.set_xlabel("Absolute alignment error (sec)")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "audit_alignment_error_hist.png", dpi=220)
    plt.close(fig)

    exps = audit_df["experiment_name"].tolist()
    n_cols = 2
    n_rows = int(np.ceil(len(exps) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows), sharey=True)
    axes = np.asarray(axes).reshape(-1)
    for i, exp_name in enumerate(exps):
        ax = axes[i]
        exp_df = sample_df[sample_df["experiment_name"] == exp_name].sort_values("timestamp_utc")
        t0 = exp_df["timestamp_utc"].iloc[0]
        rel_min = (exp_df["timestamp_utc"].to_numpy() - t0) / 60.0
        ax.plot(rel_min, exp_df["glucose_interp"].to_numpy(), color="#7A5195", alpha=0.35, linewidth=1.0, label="raw")
        ax.plot(rel_min, exp_df["glucose_smooth"].to_numpy(), color="#EF5675", linewidth=1.5, label="smooth")
        ax.set_title(exp_name, fontsize=10)
        ax.set_xlabel("Minutes from experiment start")
        ax.set_ylabel("Glucose (mmol/L)")
        if i == 0:
            ax.legend(loc="best", fontsize=8)
    for i in range(len(exps), len(axes)):
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "audit_glucose_traces_by_experiment.png", dpi=220)
    plt.close(fig)

    phase_pivot = (
        sample_df.groupby(["experiment_name", "phase"], as_index=False)
        .size()
        .pivot(index="experiment_name", columns="phase", values="size")
        .fillna(0)
    )
    phase_pivot = phase_pivot.reindex(columns=list(PHASE_LABELS), fill_value=0)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.6 * len(phase_pivot))))
    if sns is not None:
        sns.heatmap(
            phase_pivot,
            cmap="YlGnBu",
            annot=True,
            fmt=".0f",
            linewidths=0.4,
            cbar_kws={"label": "Sample count"},
            ax=ax,
        )
    else:
        values = phase_pivot.to_numpy()
        im = ax.imshow(values, cmap="YlGnBu", aspect="auto")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Sample count")
        ax.set_xticks(np.arange(len(phase_pivot.columns)))
        ax.set_xticklabels(phase_pivot.columns)
        ax.set_yticks(np.arange(len(phase_pivot.index)))
        ax.set_yticklabels(phase_pivot.index)
        for r in range(values.shape[0]):
            for c in range(values.shape[1]):
                ax.text(c, r, f"{values[r, c]:.0f}", ha="center", va="center", fontsize=8)

    ax.set_title("Phase Count by Experiment")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Experiment")
    plt.tight_layout()
    plt.savefig(out_dir / "audit_phase_count_heatmap.png", dpi=220)
    plt.close(fig)


def _plot_split_view(
    sample_df: pd.DataFrame,
    split_labels: np.ndarray,
    pca_xy: np.ndarray,
    plot_idx: np.ndarray,
    strategy: str,
    out_path: Path,
) -> None:
    view = sample_df.copy()
    view["split"] = split_labels

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    for split_name, color in (("train", "#2E86AB"), ("test", "#F26419")):
        vals = view.loc[view["split"] == split_name, "glucose_interp"].to_numpy()
        ax.hist(vals, bins=40, alpha=0.55, density=True, label=split_name, color=color)
    ax.set_title(f"{strategy}: Glucose Distribution")
    ax.set_xlabel("Glucose (mmol/L)")
    ax.set_ylabel("Density")
    ax.legend()

    ax = axes[0, 1]
    for split_name, color in (("train", "#2E86AB"), ("test", "#F26419")):
        vals = view.loc[view["split"] == split_name, "glucose_interp"].to_numpy()
        xs, ys = _ecdf(vals)
        ax.plot(xs, ys, label=split_name, color=color)
    ax.set_title(f"{strategy}: Glucose ECDF")
    ax.set_xlabel("Glucose (mmol/L)")
    ax.set_ylabel("ECDF")
    ax.legend()

    ax = axes[1, 0]
    train_idx = np.where(split_labels == "train")[0]
    test_idx = np.where(split_labels == "test")[0]
    ax.scatter(
        view.iloc[train_idx]["timestamp_bj"],
        view.iloc[train_idx]["glucose_interp"],
        s=6,
        alpha=0.35,
        color="#2E86AB",
        label="train",
    )
    ax.scatter(
        view.iloc[test_idx]["timestamp_bj"],
        view.iloc[test_idx]["glucose_interp"],
        s=6,
        alpha=0.35,
        color="#F26419",
        label="test",
    )
    ax.set_title(f"{strategy}: Timeline Coverage")
    ax.set_xlabel("Beijing time")
    ax.set_ylabel("Glucose (mmol/L)")
    ax.legend()

    ax = axes[1, 1]
    sampled_splits = split_labels[plot_idx]
    for split_name, color in (("train", "#2E86AB"), ("test", "#F26419")):
        mask = sampled_splits == split_name
        ax.scatter(
            pca_xy[mask, 0],
            pca_xy[mask, 1],
            s=8,
            alpha=0.45,
            color=color,
            label=split_name,
        )
    ax.set_title(f"{strategy}: Spectrum PCA (sampled)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def run_split_shift(
    sample_df: pd.DataFrame,
    spectra: np.ndarray,
    out_dir: Path,
    train_ratio: float,
    seed: int,
    n_alternating_segments: int,
    pca_sample_size: int,
) -> pd.DataFrame:
    strategies = ["random", "date", "experiment", "alternating"]

    rng = np.random.default_rng(seed)
    n = len(sample_df)
    pca_n = min(max(500, pca_sample_size), n)
    plot_idx = np.sort(rng.choice(n, size=pca_n, replace=False))

    sample_scaled = _scale_features(spectra[plot_idx])
    pca_xy = _pca_2d(sample_scaled)

    rows = []
    for strategy in strategies:
        split_labels = _build_split_labels(sample_df, strategy, train_ratio, seed, n_alternating_segments)
        train_idx = np.where(split_labels == "train")[0]
        test_idx = np.where(split_labels == "test")[0]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        y_train = sample_df.iloc[train_idx]["glucose_interp"].to_numpy()
        y_test = sample_df.iloc[test_idx]["glucose_interp"].to_numpy()

        phase_train = sample_df.iloc[train_idx]["phase"].value_counts(normalize=True)
        phase_test = sample_df.iloc[test_idx]["phase"].value_counts(normalize=True)
        p_train = np.array([phase_train.get(x, 0.0) for x in PHASE_LABELS], dtype=float)
        p_test = np.array([phase_test.get(x, 0.0) for x in PHASE_LABELS], dtype=float)
        p_train = np.clip(p_train, 1e-6, None)
        p_test = np.clip(p_test, 1e-6, None)

        sampled_split = split_labels[plot_idx]
        sampled_train = np.where(sampled_split == "train")[0]
        sampled_test = np.where(sampled_split == "test")[0]

        cent_dist = np.nan
        if len(sampled_train) > 0 and len(sampled_test) > 0:
            c_train = np.mean(pca_xy[sampled_train], axis=0)
            c_test = np.mean(pca_xy[sampled_test], axis=0)
            cent_dist = float(np.linalg.norm(c_train - c_test))

        if wasserstein_distance is not None:
            w_dist = float(wasserstein_distance(y_train, y_test))
        else:
            w_dist = float(np.nan)

        if energy_distance is not None:
            e_dist = float(energy_distance(y_train, y_test))
        else:
            e_dist = float(np.nan)

        if ks_2samp is not None:
            ks = ks_2samp(y_train, y_test)
            ks_stat = float(ks.statistic)
            ks_p = float(ks.pvalue)
        else:
            ks_stat = float(np.nan)
            ks_p = float(np.nan)

        if jensenshannon is not None:
            phase_js = float(jensenshannon(p_train, p_test))
        else:
            phase_js = float(np.nan)

        rows.append(
            {
                "strategy": strategy,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "glucose_wasserstein": w_dist,
                "glucose_energy_distance": e_dist,
                "glucose_ks_stat": ks_stat,
                "glucose_ks_pvalue": ks_p,
                "glucose_psi": float(_compute_psi(y_train, y_test)),
                "phase_js_distance": phase_js,
                "pca_centroid_distance": cent_dist,
            }
        )

        _plot_split_view(sample_df, split_labels, pca_xy, plot_idx, strategy, out_dir / f"split_{strategy}_overview.png")

    out = pd.DataFrame(rows).sort_values("glucose_wasserstein", ascending=False, na_position="last")
    out.to_csv(out_dir / "split_shift_metrics.csv", index=False)
    return out


def run_phase_spectrum(sample_df: pd.DataFrame, spectra: np.ndarray, out_dir: Path) -> pd.DataFrame:
    phase_groups = {}
    for phase in PHASE_LABELS:
        idx = np.where(sample_df["phase"].to_numpy() == phase)[0]
        if len(idx) > 0:
            phase_groups[phase] = spectra[idx]

    fig, ax = plt.subplots(figsize=(12, 5))
    for phase, color in (("rise", "#3B8EA5"), ("flat", "#7A7A7A"), ("fall", "#D1495B")):
        if phase not in phase_groups:
            continue
        mean_spec = np.mean(phase_groups[phase], axis=0)
        ax.plot(mean_spec, label=phase, color=color, linewidth=1.4)
    ax.set_title("Mean Spectrum by Glucose Phase")
    ax.set_xlabel("Spectral index")
    ax.set_ylabel("Power")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "phase_mean_spectra.png", dpi=220)
    plt.close(fig)

    effect_df = pd.DataFrame(columns=["spectral_index", "cohens_d_rise_vs_fall", "abs_effect"])
    if "rise" in phase_groups and "fall" in phase_groups:
        d_vec = _cohens_d(phase_groups["rise"], phase_groups["fall"])
        effect_df = pd.DataFrame(
            {
                "spectral_index": np.arange(len(d_vec), dtype=int),
                "cohens_d_rise_vs_fall": d_vec,
                "abs_effect": np.abs(d_vec),
            }
        ).sort_values("abs_effect", ascending=False)
        effect_df.to_csv(out_dir / "phase_effect_size_rise_vs_fall.csv", index=False)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(d_vec, color="#4C78A8", linewidth=1.2)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_title("Cohen's d: Rise vs Fall by Spectral Index")
        ax.set_xlabel("Spectral index")
        ax.set_ylabel("Cohen's d")
        plt.tight_layout()
        plt.savefig(out_dir / "phase_effect_curve_rise_vs_fall.png", dpi=220)
        plt.close(fig)

    return effect_df


def run_clustering(
    sample_df: pd.DataFrame,
    spectra: np.ndarray,
    out_dir: Path,
    seed: int,
    n_clusters: int,
    cluster_sample_size: int,
) -> Dict[str, float]:
    n = len(sample_df)
    rng = np.random.default_rng(seed)
    use_n = min(max(n_clusters * 100, 1200), min(cluster_sample_size, n))
    if use_n < n:
        idx = np.sort(rng.choice(n, size=use_n, replace=False))
    else:
        idx = np.arange(n)

    band_feats = _extract_band_features(spectra[idx], n_bands=24)
    aux = sample_df.iloc[idx][["glucose_smooth", "dgdt_mmol_per_min", "align_error_sec"]].to_numpy()
    feats = np.hstack([band_feats, aux])
    feats_scaled = _scale_features(feats)

    cluster = _kmeans_fit_predict(feats_scaled, n_clusters=n_clusters, seed=seed, n_init=12)
    sil = _silhouette_score_approx(feats_scaled, cluster, max_eval=2000, seed=seed)

    cluster_df = sample_df.iloc[idx].copy()
    cluster_df["cluster"] = cluster
    c_phase = pd.crosstab(cluster_df["cluster"], cluster_df["phase"], normalize="index")
    c_phase.to_csv(out_dir / "cluster_vs_phase_ratio.csv")
    c_exp = pd.crosstab(cluster_df["cluster"], cluster_df["experiment_name"], normalize="index")
    c_exp.to_csv(out_dir / "cluster_vs_experiment_ratio.csv")

    xy = _pca_2d(feats_scaled)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(xy[:, 0], xy[:, 1], c=cluster, s=8, alpha=0.55, cmap="tab10")
    ax.set_title(f"KMeans Clusters on Phase Features (k={n_clusters})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    legend = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best")
    ax.add_artist(legend)
    plt.tight_layout()
    plt.savefig(out_dir / "cluster_pca_scatter.png", dpi=220)
    plt.close(fig)

    return {
        "cluster_samples": float(use_n),
        "cluster_silhouette": float(sil),
    }


def save_summary(
    out_dir: Path,
    sample_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    split_metrics_df: pd.DataFrame,
    cluster_stats: Dict[str, float],
    effect_df: pd.DataFrame,
    phase1_summary_df: Optional[pd.DataFrame] = None,
    phase2_dist_df: Optional[pd.DataFrame] = None,
) -> None:
    phase1_highlights = {}
    if phase1_summary_df is not None and not phase1_summary_df.empty:
        phase1_highlights = {
            "dates": int(phase1_summary_df["date_id"].nunique()),
            "split_counts": phase1_summary_df.groupby("split_label")["n_samples"].sum().to_dict(),
            "train_glucose_mean": float(
                phase1_summary_df.loc[phase1_summary_df["split_label"] == "train", "glucose_mean"].mean()
            ) if (phase1_summary_df["split_label"] == "train").any() else None,
            "test_glucose_mean": float(
                phase1_summary_df.loc[phase1_summary_df["split_label"] == "test", "glucose_mean"].mean()
            ) if (phase1_summary_df["split_label"] == "test").any() else None,
        }

    phase2_highlights = {}
    if phase2_dist_df is not None and not phase2_dist_df.empty:
        by_var = []
        for var in sorted(phase2_dist_df["variable"].dropna().unique().tolist()):
            tmp = phase2_dist_df[(phase2_dist_df["scope"] == "global") & (phase2_dist_df["variable"] == var)]
            if tmp.empty:
                continue
            row = tmp.sort_values("psi", ascending=False).iloc[0]
            by_var.append(
                {
                    "variable": var,
                    "target_split": row.get("target_split", None),
                    "psi": float(row.get("psi", np.nan)),
                    "wasserstein": float(row.get("wasserstein", np.nan)),
                }
            )
        phase2_highlights = {
            "global_top_shift_variables": by_var,
        }

    summary = {
        "n_samples": int(len(sample_df)),
        "n_experiments": int(sample_df["experiment_name"].nunique()),
        "glucose_range": [
            float(sample_df["glucose_interp"].min()),
            float(sample_df["glucose_interp"].max()),
        ],
        "align_error_median_sec": float(sample_df["align_error_sec"].median()),
        "align_error_p95_sec": float(sample_df["align_error_sec"].quantile(0.95)),
        "phase_ratio": sample_df["phase"].value_counts(normalize=True).to_dict(),
        "worst_split_by_wasserstein": (
            split_metrics_df.iloc[0]["strategy"] if not split_metrics_df.empty else None
        ),
        "cluster_stats": cluster_stats,
        "top_effect_indices": (
            effect_df.head(20).to_dict(orient="records") if not effect_df.empty else []
        ),
        "phase1_highlights": phase1_highlights,
        "phase2_highlights": phase2_highlights,
        "audit_rows": audit_df.to_dict(orient="records"),
    }

    with open(out_dir / "diagnostic_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tao DB diagnostics for split and distribution shift")
    parser.add_argument("--dataset_root", type=str, default="./Dataset", help="Dataset root directory")
    parser.add_argument("--db_user", type=str, default="Tao_db", help="Target DB user folder")
    parser.add_argument("--output_dir", type=str, default="./results/tao_db_diagnostics", help="Output folder")
    parser.add_argument(
        "--interpolation",
        type=str,
        default="pchip",
        choices=["linear", "cubic", "pchip", "akima", "nearest"],
        help="Glucose interpolation method",
    )
    parser.add_argument("--smooth_window", type=int, default=17, help="Median smoothing window")
    parser.add_argument("--phase_quantile", type=float, default=0.70, help="Adaptive threshold quantile of |dG/dt|")
    parser.add_argument("--min_phase_rate", type=float, default=0.02, help="Minimum phase threshold in mmol/L/min")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train ratio in split diagnostics")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--alternating_segments", type=int, default=10, help="Segment count for alternating strategy")
    parser.add_argument("--pca_sample_size", type=int, default=12000, help="PCA sampling size")
    parser.add_argument("--cluster_sample_size", type=int, default=15000, help="Clustering sampling size")
    parser.add_argument("--n_clusters", type=int, default=3, help="KMeans cluster number")
    parser.add_argument("--max_experiments", type=int, default=0, help="Debug-only: limit loaded experiments, 0 means all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    out_dir = Path(args.output_dir).resolve()
    _safe_mkdir(out_dir)

    print("=" * 72)
    print("Tao DB diagnostics started")
    print(f"Dataset root: {dataset_root}")
    print(f"DB user: {args.db_user}")
    print(f"Output dir: {out_dir}")
    print("=" * 72)

    sample_df, spectra, audit_df = load_tao_samples(
        dataset_root=dataset_root,
        db_user=args.db_user,
        interpolation=args.interpolation,
        smooth_window=args.smooth_window,
        phase_quantile=args.phase_quantile,
        min_phase_rate=args.min_phase_rate,
        max_experiments=args.max_experiments,
    )

    val_ratio = max(0.0, min(1.0 - args.train_ratio, 0.1))
    master_df = build_master_table_with_date_split(
        sample_df=sample_df,
        spectra=spectra,
        train_ratio=args.train_ratio,
        val_ratio=val_ratio,
    )

    master_df.to_csv(out_dir / "sample_table.csv", index=False)
    phase1_summary_df, phase2_dist_df = run_date_distribution_diagnostics(master_df, out_dir)
    audit_df.to_csv(out_dir / "experiment_audit_table.csv", index=False)
    np.save(out_dir / "spectra.npy", spectra.astype(np.float32))

    plot_audit(sample_df, audit_df, out_dir)

    split_metrics_df = run_split_shift(
        sample_df=master_df,
        spectra=spectra,
        out_dir=out_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
        n_alternating_segments=args.alternating_segments,
        pca_sample_size=args.pca_sample_size,
    )

    effect_df = run_phase_spectrum(master_df, spectra, out_dir)
    cluster_stats = run_clustering(
        sample_df=master_df,
        spectra=spectra,
        out_dir=out_dir,
        seed=args.seed,
        n_clusters=args.n_clusters,
        cluster_sample_size=args.cluster_sample_size,
    )

    save_summary(
        out_dir,
        master_df,
        audit_df,
        split_metrics_df,
        cluster_stats,
        effect_df,
        phase1_summary_df=phase1_summary_df,
        phase2_dist_df=phase2_dist_df,
    )

    print("\nDone. Artifacts written to:")
    print(f"  {out_dir}")
    print("Key outputs:")
    print("  - diagnostic_summary.json")
    print("  - phase1_date_split_summary.csv / phase1_date_split_phase_ratio.csv")
    print("  - phase2_train_vs_nontrain_distance.csv")
    print("  - split_shift_metrics.csv")
    print("  - audit_*.png")
    print("  - phase1_date_split_glucose_box.png / phase1_date_split_phase_heatmap.png")
    print("  - phase2_global_glucose_ecdf_by_split.png")
    print("  - split_*_overview.png")
    print("  - phase_mean_spectra.png / phase_effect_curve_rise_vs_fall.png")
    print("  - cluster_pca_scatter.png / cluster_vs_*.csv")


if __name__ == "__main__":
    main()
