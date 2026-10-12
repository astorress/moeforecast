"""Reusable, time-ordered hyperparameter searches for MoE components."""

from __future__ import annotations

from collections.abc import Mapping

from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV, TimeSeriesSplit


def default_search_spaces() -> dict[str, dict[str, object]]:
    """Finite, practical grids for the built-in expert and classifier names.

    Keys use the candidate names returned by ``default_candidates`` and
    ``default_classifiers``. Users can override or extend this mapping.
    """
    return {
        "ridge": {"ridge__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        "elasticnet": {"elasticnet__alpha": [0.001, 0.01, 0.1], "elasticnet__l1_ratio": [0.2, 0.5, 0.8]},
        "lasso": {"lasso__alpha": [0.001, 0.01, 0.1]},
        "svr": {"svr__C": [0.1, 1.0, 10.0], "svr__epsilon": [0.01, 0.1, 0.5]},
        "knn": {"kneighborsregressor__n_neighbors": [3, 5, 9, 15]},
        "random_forest": {
            "n_estimators": [100, 300],
            "max_depth": [None, 8, 16],
            "min_samples_leaf": [1, 3, 5],
        },
        "extra_trees": {
            "n_estimators": [100, 300],
            "max_depth": [None, 8, 16],
            "min_samples_leaf": [1, 3, 5],
        },
        "hist_gbm": {
            "learning_rate": [0.03, 0.1],
            "max_iter": [100, 250],
            "max_leaf_nodes": [15, 31],
            "l2_regularization": [0.0, 1.0],
        },
        "gradient_boosting": {
            "learning_rate": [0.03, 0.1],
            "n_estimators": [100, 250],
            "max_depth": [2, 3],
            "min_samples_leaf": [2, 5],
        },
        "logistic": {"logisticregression__C": [0.1, 1.0, 10.0]},
        "logistic_estimator": {"C": [0.1, 1.0, 10.0]},
        "random_forest_classifier": {
            "n_estimators": [100, 300],
            "max_depth": [None, 8, 16],
            "min_samples_leaf": [1, 3, 5],
        },
        "hist_gbm_classifier": {
            "learning_rate": [0.03, 0.1],
            "max_iter": [100, 250],
            "max_leaf_nodes": [15, 31],
            "l2_regularization": [0.0, 1.0],
        },
    }


def search_estimator(
    estimator: BaseEstimator,
    param_space: Mapping[str, object],
    X,
    y,
    *,
    strategy: str,
    n_splits: int,
    n_iter: int,
    scoring: str,
    random_state: int = 0,
):
    """Fit a grid/random search using forward-chaining CV and return the search."""
    if strategy not in {"grid", "random"}:
        raise ValueError("strategy must be 'grid' or 'random'.")
    if not param_space:
        raise ValueError("A non-empty parameter space is required for HPO.")
    if n_splits < 2 or len(X) <= n_splits:
        raise ValueError("HPO requires at least n_splits + 1 samples.")
    cv = TimeSeriesSplit(n_splits=n_splits)
    common = dict(estimator=estimator, cv=cv, scoring=scoring, refit=True, n_jobs=-1, error_score="raise")
    if strategy == "grid":
        search = GridSearchCV(**common, param_grid=dict(param_space))
    else:
        if n_iter < 1:
            raise ValueError("n_iter must be a positive integer.")
        search = RandomizedSearchCV(
            **common,
            param_distributions=dict(param_space),
            n_iter=(
                n_iter
                if any(hasattr(values, "rvs") for values in param_space.values())
                else min(n_iter, len(ParameterGrid(param_space)))
            ),
            random_state=random_state,
        )
    return search.fit(X, y)
