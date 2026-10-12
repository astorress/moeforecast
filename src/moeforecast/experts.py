"""Selection and training of one regression model (expert) per regime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from moeforecast.metrics import mae, rmse

ExpertKey = tuple[int, ...]
"""Identifier of an expert: one cluster label per regime dimension, e.g. (0, 1)."""

# (minimum number of samples, allowed candidates; None = all). The last row whose
# minimum is <= n applies. Keeps flexible models away from small regimes.
DEFAULT_COMPLEXITY: tuple[tuple[int, tuple[str, ...] | None], ...] = (
    (0, ("ridge",)),
    (200, ("ridge", "elasticnet", "hist_gbm")),
    (1000, None),
)

_METRICS = {"mae": mae, "rmse": rmse}


def default_candidates(random_state: int = 0) -> dict[str, BaseEstimator]:
    """Candidate expert models. LightGBM is included when it is installed."""
    candidates = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "elasticnet": make_pipeline(StandardScaler(), ElasticNet(alpha=0.01, max_iter=5000)),
        "random_forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=random_state
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=random_state
        ),
        "hist_gbm": HistGradientBoostingRegressor(random_state=random_state),
    }
    try:
        from lightgbm import LGBMRegressor

        candidates["lightgbm"] = LGBMRegressor(
            n_estimators=500, learning_rate=0.05, random_state=random_state, verbose=-1
        )
    except ImportError:
        pass
    return candidates


def format_key(key: ExpertKey, dimensions: Sequence[str]) -> str:
    """Readable expert name, e.g. ``format_key((0, 1), ["day", "week"]) == "day0-week1"``."""
    return "-".join(f"{dim}{label}" for dim, label in zip(dimensions, key, strict=True))


class ExpertSelector(BaseEstimator):
    """
    Choose and train the best candidate model for each regime.

    For every expert key, the candidates allowed by ``complexity`` are scored
    with forward-chaining cross-validation on that regime's samples (in time
    order), the best one by ``metric`` is selected, and it is refitted on all of
    the regime's samples.

    Parameters
    ----------
    candidates : mapping name -> regressor, optional
        Defaults to :func:`default_candidates`.
    metric : {"mae", "rmse"}
        Selection metric (lower is better).
    n_splits : int
        Number of forward-chaining folds per regime.
    complexity : sequence of (min_samples, allowed names or None)
        Which candidates are allowed for a regime with n samples.
    min_samples : int
        Regimes with fewer samples get no expert; the gating then routes their
        time steps to the remaining experts.
    """

    def __init__(
        self,
        candidates: Mapping[str, BaseEstimator] | None = None,
        metric: str = "mae",
        n_splits: int = 3,
        complexity: Sequence[tuple[int, Sequence[str] | None]] = DEFAULT_COMPLEXITY,
        min_samples: int = 50,
    ):
        self.candidates = candidates
        self.metric = metric
        self.n_splits = n_splits
        self.complexity = complexity
        self.min_samples = min_samples

    def allowed_candidates(self, n: int, names: Sequence[str]) -> list[str]:
        allowed = None
        for min_n, subset in sorted(self.complexity, key=lambda row: row[0]):
            if n >= min_n:
                allowed = subset
        return list(names) if allowed is None else [c for c in names if c in allowed]

    def fit(self, X: pd.DataFrame, y: pd.Series, keys: pd.Series):
        """
        Parameters
        ----------
        X : DataFrame of features.
        y : Series, target aligned with X.
        keys : Series of ExpertKey tuples aligned with X (NaN/None = no regime).
        """
        if self.metric not in _METRICS:
            raise ValueError(f"metric must be one of {sorted(_METRICS)}")
        candidates = dict(self.candidates or default_candidates())
        self.candidate_estimators_ = {name: clone(model) for name, model in candidates.items()}
        score_fn = _METRICS[self.metric]
        valid = keys.notna().to_numpy() & y.notna().to_numpy() & X.notna().all(axis=1).to_numpy()

        rows, self.experts_, self.selected_ = [], {}, {}
        for key in sorted(set(keys[valid])):
            mask = valid & np.fromiter((k == key for k in keys), dtype=bool, count=len(keys))
            n = int(mask.sum())
            if n < self.min_samples:
                rows.append(
                    {
                        "expert": key,
                        "candidate": None,
                        "n_samples": n,
                        "score_mean": np.nan,
                        "score_std": np.nan,
                        "selected": False,
                        "note": "too few samples",
                    }
                )
                continue
            Xk, yk = X[mask], y[mask]
            names = self.allowed_candidates(n, list(candidates))
            cv = TimeSeriesSplit(n_splits=self.n_splits)
            scores = {}
            for name in names:
                fold_scores = []
                for train_idx, test_idx in cv.split(Xk):
                    model = clone(candidates[name]).fit(Xk.iloc[train_idx], yk.iloc[train_idx])
                    fold_scores.append(score_fn(yk.iloc[test_idx], model.predict(Xk.iloc[test_idx])))
                scores[name] = (float(np.mean(fold_scores)), float(np.std(fold_scores)))
            best = min(scores, key=lambda name: scores[name][0])  # ties: first in candidate order
            for name, (mean, std) in scores.items():
                rows.append(
                    {
                        "expert": key,
                        "candidate": name,
                        "n_samples": n,
                        "score_mean": mean,
                        "score_std": std,
                        "selected": name == best,
                        "note": "",
                    }
                )
            self.selected_[key] = best
            self.experts_[key] = clone(candidates[best]).fit(Xk, yk)

        if not self.experts_:
            raise ValueError("No regime has enough samples to train an expert.")
        self.results_ = pd.DataFrame(rows).rename(
            columns={"score_mean": f"{self.metric}_mean", "score_std": f"{self.metric}_std"}
        )
        self.expert_keys_ = sorted(self.experts_)
        return self
