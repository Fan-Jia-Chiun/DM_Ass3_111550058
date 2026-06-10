from __future__ import annotations

import numpy as np
import pandas as pd
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


ALL_ENSEMBLE_NAMES = ["extra_trees", "random_forest", "hist_gb"]


def resolve_ensemble_weight_map(ensemble_weights: str | None) -> dict[str, float]:
    if ensemble_weights is None:
        return {"extra_trees": 3.0, "random_forest": 2.0, "hist_gb": 1.0}

    try:
        weights = [float(part.strip()) for part in ensemble_weights.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError("--ensemble-weights must be comma-separated numbers, e.g. 3,2,1.") from exc

    if len(weights) != len(ALL_ENSEMBLE_NAMES):
        raise ValueError(
            f"--ensemble-weights expects {len(ALL_ENSEMBLE_NAMES)} values for "
            f"{','.join(ALL_ENSEMBLE_NAMES)}; "
            f"received {len(weights)}."
        )
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("--ensemble-weights must be non-negative and cannot sum to zero.")
    return dict(zip(ALL_ENSEMBLE_NAMES, weights))


def make_model(
    model_name: str,
    seed: int,
    n_jobs: int,
    fast: bool,
    ensemble_weights: str | None = None,
) -> Pipeline:
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
        weight_map = resolve_ensemble_weight_map(ensemble_weights)
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
        estimators = [(name, estimator) for name, estimator in estimators if weight_map[name] > 0]
        weights = [weight_map[name] for name, _ in estimators]
        if not estimators:
            raise ValueError("No active ensemble estimators. Check --ensemble-weights and installed scikit-learn support.")
        if len(estimators) == 1:
            clf = estimators[0][1]
        else:
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
    ensemble_weights: str | None = None,
) -> None:
    splitter = make_splitter(y, groups, folds, seed)
    scores = []
    oof = np.zeros_like(y)

    print("\nGrouped cross-validation by user:")
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y, groups), start=1):
        model = make_model(model_name, seed + fold, n_jobs, fast, ensemble_weights)
        model.fit(X.iloc[train_idx], y[train_idx])
        pred = model.predict(X.iloc[valid_idx])
        score = f1_score(y[valid_idx], pred, average="macro")
        scores.append(score)
        oof[valid_idx] = pred
        print(f"  Fold {fold}: macro F1 = {score:.5f} ({len(valid_idx)} files)")

    print(f"CV macro F1: mean={np.mean(scores):.5f}, std={np.std(scores):.5f}")
    print("\nOOF classification report:")
    print(classification_report(y, oof, digits=4, zero_division=0))
