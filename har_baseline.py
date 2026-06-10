"""Command line entry point for NYCU Data Mining Assignment 3 HAR baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.features import build_feature_table
from src.models import make_model, run_cv
from src.utils import (
    align_test_to_submission,
    append_user_zscore_features,
    auto_n_jobs,
    feature_columns,
    find_split_dir,
    list_csv_files,
)


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
        "--ensemble-weights",
        type=str,
        default=None,
        help=(
            "Comma-separated ensemble voting weights for extra_trees,random_forest,hist_gb. "
            "Only used when --model ensemble. Default is 3,2,1."
        ),
    )
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
        run_cv(X, y, groups, args.model, args.seed, n_jobs, args.folds, args.fast, args.ensemble_weights)

    print("\nTraining final model on all training files ...")
    model = make_model(args.model, args.seed, n_jobs, args.fast, args.ensemble_weights)
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
