from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def safe_float(value: float) -> float:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or not np.isfinite(value):
        return 0.0
    return float(value)


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


def align_test_to_submission(
    test_table: pd.DataFrame,
    sample_submission: pd.DataFrame,
    cols: list[str],
) -> pd.DataFrame:
    sample_ids = sample_submission["Id"].astype(int).to_numpy()
    indexed = test_table.set_index("file_id")
    missing = sorted(set(sample_ids) - set(indexed.index))
    if missing:
        preview = ", ".join(map(str, missing[:10]))
        raise ValueError(f"{len(missing)} sample_submission IDs are missing from test features: {preview}")
    return indexed.loc[sample_ids, cols].reset_index(drop=True)
