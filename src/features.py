from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from src.utils import safe_float


RAW_COLUMNS = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]
MEAN_AXES = ["mean_x", "mean_y", "mean_z"]
STD_AXES = ["std_x", "std_y", "std_z"]


def add_basic_stats(features: dict[str, float], name: str, values: np.ndarray) -> None:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        x = np.array([0.0])

    q10, q25, q50, q75, q90 = np.percentile(x, [10, 25, 50, 75, 90])
    diffs = np.diff(x)
    abs_diffs = np.abs(diffs) if diffs.size else np.array([0.0])

    features[f"{name}_mean"] = safe_float(np.mean(x))
    features[f"{name}_std"] = safe_float(np.std(x))
    features[f"{name}_min"] = safe_float(np.min(x))
    features[f"{name}_max"] = safe_float(np.max(x))
    features[f"{name}_median"] = safe_float(q50)
    features[f"{name}_q10"] = safe_float(q10)
    features[f"{name}_q25"] = safe_float(q25)
    features[f"{name}_q75"] = safe_float(q75)
    features[f"{name}_q90"] = safe_float(q90)
    features[f"{name}_iqr"] = safe_float(q75 - q25)
    features[f"{name}_range"] = safe_float(np.max(x) - np.min(x))
    features[f"{name}_rms"] = safe_float(np.sqrt(np.mean(x * x)))
    features[f"{name}_energy"] = safe_float(np.mean(x * x))
    features[f"{name}_mad"] = safe_float(np.mean(np.abs(x - np.mean(x))))
    features[f"{name}_skew"] = safe_float(skew(x, bias=False)) if x.size > 2 else 0.0
    features[f"{name}_kurtosis"] = safe_float(kurtosis(x, bias=False)) if x.size > 3 else 0.0
    features[f"{name}_first"] = safe_float(x[0])
    features[f"{name}_last"] = safe_float(x[-1])
    features[f"{name}_last_minus_first"] = safe_float(x[-1] - x[0])
    features[f"{name}_diff_mean_abs"] = safe_float(np.mean(abs_diffs))
    features[f"{name}_diff_max_abs"] = safe_float(np.max(abs_diffs))


def add_fft_stats(features: dict[str, float], name: str, values: np.ndarray) -> None:
    x = np.asarray(values, dtype=float)
    x = np.nan_to_num(x - np.nanmean(x), nan=0.0, posinf=0.0, neginf=0.0)
    if x.size < 4 or np.allclose(x, 0.0):
        for idx in range(6):
            features[f"{name}_fft_band_{idx}"] = 0.0
        features[f"{name}_fft_entropy"] = 0.0
        features[f"{name}_fft_centroid"] = 0.0
        return

    power = np.abs(np.fft.rfft(x)) ** 2
    power = power[1:]
    if power.size == 0 or np.sum(power) <= 0:
        probs = np.ones(1)
    else:
        probs = power / np.sum(power)

    bands = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 40), (40, None)]
    total = np.sum(power) + 1e-12
    for idx, (start, end) in enumerate(bands):
        band_power = np.sum(power[start:end])
        features[f"{name}_fft_band_{idx}"] = safe_float(band_power / total)

    freqs = np.arange(1, power.size + 1, dtype=float)
    entropy = -np.sum(probs * np.log(probs + 1e-12)) / math.log(len(probs) + 1e-12)
    features[f"{name}_fft_entropy"] = safe_float(entropy)
    features[f"{name}_fft_centroid"] = safe_float(np.sum(freqs * probs) / len(freqs))


def add_segment_stats(features: dict[str, float], name: str, values: np.ndarray, segments: int = 6) -> None:
    x = np.asarray(values, dtype=float)
    chunks = np.array_split(x, segments)
    for idx, chunk in enumerate(chunks):
        if chunk.size == 0:
            chunk = np.array([0.0])
        features[f"{name}_seg{idx}_mean"] = safe_float(np.nanmean(chunk))
        features[f"{name}_seg{idx}_std"] = safe_float(np.nanstd(chunk))
        features[f"{name}_seg{idx}_min"] = safe_float(np.nanmin(chunk))
        features[f"{name}_seg{idx}_max"] = safe_float(np.nanmax(chunk))


def add_pairwise_corr(
    features: dict[str, float],
    prefix: str,
    frame: pd.DataFrame,
    columns: Iterable[str],
) -> None:
    cols = list(columns)
    values = frame[cols].to_numpy(dtype=float)
    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1 :]:
            a = values[:, i]
            b = frame[col_b].to_numpy(dtype=float)
            if np.nanstd(a) < 1e-12 or np.nanstd(b) < 1e-12:
                corr = 0.0
            else:
                corr = np.corrcoef(a, b)[0, 1]
            features[f"{prefix}_corr_{col_a}_{col_b}"] = safe_float(corr)


def extract_features_from_frame(df: pd.DataFrame) -> dict[str, float]:
    df = df.sort_values("index").reset_index(drop=True)
    for col in RAW_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    mean_values = df[MEAN_AXES].to_numpy(dtype=float)
    std_values = df[STD_AXES].to_numpy(dtype=float)
    df["mean_mag"] = np.sqrt(np.sum(mean_values * mean_values, axis=1))
    df["std_mag"] = np.sqrt(np.sum(std_values * std_values, axis=1))
    jerk = np.diff(mean_values, axis=0)
    df["jerk_mag"] = np.r_[0.0, np.sqrt(np.sum(jerk * jerk, axis=1))]
    df["mean_xy_angle"] = np.arctan2(df["mean_y"].to_numpy(dtype=float), df["mean_x"].to_numpy(dtype=float))
    df["mean_xz_angle"] = np.arctan2(df["mean_z"].to_numpy(dtype=float), df["mean_x"].to_numpy(dtype=float))
    df["mean_yz_angle"] = np.arctan2(df["mean_z"].to_numpy(dtype=float), df["mean_y"].to_numpy(dtype=float))

    feature_cols = RAW_COLUMNS + [
        "mean_mag",
        "std_mag",
        "jerk_mag",
        "mean_xy_angle",
        "mean_xz_angle",
        "mean_yz_angle",
    ]

    features: dict[str, float] = {}
    for col in feature_cols:
        values = df[col].to_numpy(dtype=float)
        add_basic_stats(features, col, values)
        if col in MEAN_AXES + ["mean_mag", "jerk_mag"]:
            add_fft_stats(features, col, values)
        if col in MEAN_AXES + ["mean_mag", "std_mag", "jerk_mag"]:
            add_segment_stats(features, col, values)

    add_pairwise_corr(features, "mean_axes", df, MEAN_AXES)
    add_pairwise_corr(features, "std_axes", df, STD_AXES)

    mag = df["mean_mag"].to_numpy(dtype=float)
    jerk_mag = df["jerk_mag"].to_numpy(dtype=float)
    features["mean_mag_above_1g_ratio"] = safe_float(np.mean(mag > 1.0))
    features["mean_mag_near_1g_ratio"] = safe_float(np.mean(np.abs(mag - 1.0) < 0.05))
    features["jerk_top10_mean"] = safe_float(np.mean(np.sort(jerk_mag)[-10:]))
    features["jerk_zeroish_ratio"] = safe_float(np.mean(jerk_mag < 1e-3))
    return features


def read_one_csv(path: Path) -> dict[str, object]:
    df = pd.read_csv(path)
    features = extract_features_from_frame(df)
    label = int(df["label"].iloc[0]) if "label" in df.columns else None
    file_id = int(df["file_id"].iloc[0]) if "file_id" in df.columns else int(path.stem)
    return {
        "file_id": file_id,
        "user": path.parent.name,
        "label": label,
        "features": features,
    }


def build_feature_table(
    files: list[Path],
    cache_path: Path,
    rebuild_cache: bool,
    n_jobs: int,
) -> pd.DataFrame:
    if cache_path.exists() and not rebuild_cache:
        print(f"Loading cached features: {cache_path}")
        return joblib.load(cache_path)

    print(f"Extracting features from {len(files)} CSV files ...")
    rows = joblib.Parallel(n_jobs=n_jobs, verbose=5)(
        joblib.delayed(read_one_csv)(path) for path in files
    )

    records = []
    for row in rows:
        record = {
            "file_id": row["file_id"],
            "user": row["user"],
            "label": row["label"],
        }
        record.update(row["features"])
        records.append(record)

    table = pd.DataFrame.from_records(records)
    table = table.sort_values("file_id").reset_index(drop=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(table, cache_path)
    print(f"Saved cached features: {cache_path}")
    return table
