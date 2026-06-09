"""Baseline pipeline for NYCU Data Mining Assignment 3.

This script converts each 5-minute accelerometer CSV into fixed-length
statistics, trains a scikit-learn classifier, and writes a Kaggle submission.
It is intentionally dependency-light so it can run in a fresh course setup.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
except ImportError:  # pragma: no cover - older scikit-learn fallback
    HistGradientBoostingClassifier = None

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - older scikit-learn fallback
    StratifiedGroupKFold = None


RAW_COLUMNS = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]
MEAN_AXES = ["mean_x", "mean_y", "mean_z"]
STD_AXES = ["std_x", "std_y", "std_z"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a HAR baseline and create sample_submission-compatible predictions."
    )
    parser.add_argument("--data-root", type=Path, default=Path("."), help="Folder containing train/, test/, and sample_submission.csv.")
    parser.add_argument("--sample-submission", type=Path, default=None, help="Path to sample_submission.csv.")
    parser.add_argument("--output", type=Path, default=Path("submission.csv"), help="Output CSV path.")
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"), help="Feature cache folder.")
    parser.add_argument("--rebuild-cache", action="store_true", help="Ignore cached features and rebuild them.")
    parser.add_argument("--n-jobs", type=int, default=0, help="Parallel workers. 0 uses up to 4 local cores.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--folds", type=int, default=5, help="Number of grouped CV folds.")
    parser.add_argument("--skip-cv", action="store_true", help="Skip cross-validation and only train full model.")
    parser.add_argument("--model", choices=["extra_trees", "ensemble"], default="extra_trees", help="Classifier family.")
    parser.add_argument(
        "--user-zscore",
        choices=["none", "add", "replace"],
        default="add",
        help=(
            "Per-user z-score normalization on extracted features. "
            "'add' keeps raw features and appends user_z_* features; "
            "'replace' uses only user_z_* features."
        ),
    )
    parser.add_argument("--fast", action="store_true", help="Use fewer trees for faster debugging.")
    parser.add_argument("--train-limit", type=int, default=None, help="Debug: only use first N train files.")
    parser.add_argument("--test-limit", type=int, default=None, help="Debug: only use first N test files.")
    return parser.parse_args()


def auto_n_jobs(requested: int) -> int:
    if requested and requested != 0:
        return requested
    return max(1, min(4, os.cpu_count() or 1))


def find_split_dir(data_root: Path, split: str) -> Path:
    candidates = [
        data_root / split / split,
        data_root / split,
    ]
    for candidate in candidates:
        if candidate.exists() and any(candidate.glob("User_*/*.csv")):
            return candidate
    raise FileNotFoundError(
        f"Could not find {split} CSV files. Expected paths like "
        f"{data_root / split / split / 'User_001' / '00001.csv'}."
    )


def list_csv_files(split_dir: Path, limit: int | None = None) -> list[Path]:
    files = sorted(split_dir.glob("User_*/*.csv"), key=lambda p: (p.parent.name, p.stem))
    if limit is not None:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"No CSV files found under {split_dir}.")
    return files


def safe_float(value: float) -> float:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or not np.isfinite(value):
        return 0.0
    return float(value)


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
    power = power[1:]  # drop DC
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
    features[f"{name}_fft_entropy"] = safe_float(-np.sum(probs * np.log(probs + 1e-12)) / math.log(len(probs) + 1e-12))
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


def add_pairwise_corr(features: dict[str, float], prefix: str, frame: pd.DataFrame, columns: Iterable[str]) -> None:
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

    # Captures whether the device is mostly static or has bursts of motion.
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


def feature_columns(train_table: pd.DataFrame, test_table: pd.DataFrame) -> list[str]:
    blocked = {"file_id", "user", "label"}
    cols = sorted((set(train_table.columns) & set(test_table.columns)) - blocked)
    if not cols:
        raise ValueError("No feature columns were generated.")
    return cols


def append_user_zscore_features(
    train_table: pd.DataFrame,
    test_table: pd.DataFrame,
    cols: list[str],
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if mode == "none":
        return train_table, test_table, cols

    def transform_one(table: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        grouped = table.groupby("user", sort=False)
        means = grouped[cols].transform("mean")
        stds = grouped[cols].transform("std").replace(0.0, np.nan)
        z = (table[cols] - means) / stds
        z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        z_cols = [f"user_z_{col}" for col in cols]
        z.columns = z_cols
        return pd.concat([table, z.astype(np.float32)], axis=1), z_cols

    train_out, z_cols = transform_one(train_table)
    test_out, _ = transform_one(test_table)
    if mode == "replace":
        selected_cols = z_cols
    elif mode == "add":
        selected_cols = cols + z_cols
    else:
        raise ValueError(f"Unsupported user-zscore mode: {mode}")

    print(f"Per-user z-score mode: {mode} | selected features: {len(selected_cols)}")
    return train_out, test_out, selected_cols


def make_model(model_name: str, seed: int, n_jobs: int, fast: bool) -> Pipeline:
    n_extra = 250 if fast else 900
    n_rf = 200 if fast else 600

    extra_trees = ExtraTreesClassifier(
        n_estimators=n_extra,
        random_state=seed,
        n_jobs=n_jobs,
        class_weight="balanced",
        max_features="sqrt",
        min_samples_leaf=1,
        bootstrap=False,
    )

    if model_name == "extra_trees":
        clf = extra_trees
    else:
        estimators = [
            ("extra_trees", extra_trees),
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=n_rf,
                    random_state=seed + 7,
                    n_jobs=n_jobs,
                    class_weight="balanced_subsample",
                    max_features="sqrt",
                    min_samples_leaf=1,
                ),
            ),
        ]
        weights = [3, 2]
        if HistGradientBoostingClassifier is not None:
            estimators.append(
                (
                    "hist_gb",
                    HistGradientBoostingClassifier(
                        learning_rate=0.04 if not fast else 0.07,
                        max_iter=220 if not fast else 80,
                        l2_regularization=0.05,
                        random_state=seed + 13,
                    ),
                )
            )
            weights.append(1)
        clf = VotingClassifier(estimators=estimators, voting="soft", weights=weights, n_jobs=1)

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold()),
            ("classifier", clf),
        ]
    )


def make_splitter(y: np.ndarray, groups: np.ndarray, folds: int, seed: int):
    unique_groups = np.unique(groups)
    n_splits = min(folds, len(unique_groups))
    if n_splits < 2:
        raise ValueError("Need at least 2 users/groups for validation.")
    if StratifiedGroupKFold is not None:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return GroupKFold(n_splits=n_splits)


def run_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    model_name: str,
    seed: int,
    n_jobs: int,
    folds: int,
    fast: bool,
) -> None:
    splitter = make_splitter(y, groups, folds, seed)
    scores = []
    oof = np.zeros_like(y)

    print("\nGrouped cross-validation by user:")
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y, groups), start=1):
        model = make_model(model_name, seed + fold, n_jobs, fast)
        model.fit(X.iloc[train_idx], y[train_idx])
        pred = model.predict(X.iloc[valid_idx])
        score = f1_score(y[valid_idx], pred, average="macro")
        scores.append(score)
        oof[valid_idx] = pred
        print(f"  Fold {fold}: macro F1 = {score:.5f} ({len(valid_idx)} files)")

    print(f"CV macro F1: mean={np.mean(scores):.5f}, std={np.std(scores):.5f}")
    print("\nOOF classification report:")
    print(classification_report(y, oof, digits=4))


def align_test_to_submission(test_table: pd.DataFrame, sample_submission: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    sample_ids = sample_submission["Id"].astype(int).to_numpy()
    indexed = test_table.set_index("file_id")
    missing = sorted(set(sample_ids) - set(indexed.index))
    if missing:
        preview = ", ".join(map(str, missing[:10]))
        raise ValueError(f"{len(missing)} sample_submission IDs are missing from test features: {preview}")
    return indexed.loc[sample_ids, cols].reset_index(drop=True)


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    n_jobs = auto_n_jobs(args.n_jobs)
    sample_path = args.sample_submission or data_root / "sample_submission.csv"
    cache_dir = args.cache_dir if args.cache_dir.is_absolute() else data_root / args.cache_dir
    output_path = args.output if args.output.is_absolute() else data_root / args.output

    train_dir = find_split_dir(data_root, "train")
    test_dir = find_split_dir(data_root, "test")
    sample_submission = pd.read_csv(sample_path)

    train_files = list_csv_files(train_dir, args.train_limit)
    test_files = list_csv_files(test_dir, args.test_limit)
    train_cache = cache_dir / f"train_features_limit_{args.train_limit or 'all'}.joblib"
    test_cache = cache_dir / f"test_features_limit_{args.test_limit or 'all'}.joblib"

    train_table = build_feature_table(train_files, train_cache, args.rebuild_cache, n_jobs)
    test_table = build_feature_table(test_files, test_cache, args.rebuild_cache, n_jobs)
    if args.test_limit is not None:
        available_ids = set(test_table["file_id"].astype(int))
        sample_submission = sample_submission[sample_submission["Id"].astype(int).isin(available_ids)].reset_index(drop=True)
        print(f"Debug test limit enabled; writing predictions for {len(sample_submission)} sample IDs.")
    cols = feature_columns(train_table, test_table)
    train_table, test_table, cols = append_user_zscore_features(train_table, test_table, cols, args.user_zscore)

    X = train_table[cols]
    y = train_table["label"].astype(int).to_numpy()
    groups = train_table["user"].astype(str).to_numpy()
    print(f"\nTrain files: {len(train_table)} | Test files: {len(test_table)} | Features: {len(cols)}")
    print("Train label distribution:")
    print(train_table["label"].value_counts().sort_index().to_string())

    if not args.skip_cv:
        run_cv(X, y, groups, args.model, args.seed, n_jobs, args.folds, args.fast)

    print("\nTraining final model on all training files ...")
    model = make_model(args.model, args.seed, n_jobs, args.fast)
    model.fit(X, y)

    X_test = align_test_to_submission(test_table, sample_submission, cols)
    predictions = model.predict(X_test).astype(int)
    submission = sample_submission.copy()
    submission["Label"] = predictions
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Wrote submission: {output_path}")
    print("Predicted label distribution:")
    print(pd.Series(predictions).value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
