"""Classifiers that predict the regime (cluster) of each time step from its features."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def default_classifiers(random_state: int = 0) -> dict[str, BaseEstimator]:
    """Candidate regime classifiers. All implement ``predict_proba``."""
    return {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=random_state
        ),
        "hist_gbm": HistGradientBoostingClassifier(random_state=random_state),
    }


class RegimeClassifier(ClassifierMixin, BaseEstimator):
    """
    Predict regime probabilities for the gating network.

    Parameters
    ----------
    estimator : classifier, optional
        Any scikit-learn classifier with ``predict_proba``. Defaults to
        ``HistGradientBoostingClassifier``.
    features : sequence of str, optional
        Feature columns to use. Defaults to all columns.
    """

    def __init__(self, estimator: BaseEstimator | None = None, features: Sequence[str] | None = None):
        self.estimator = estimator
        self.features = features

    def fit(self, X: pd.DataFrame, y: pd.Series):
        base = self.estimator if self.estimator is not None else HistGradientBoostingClassifier()
        if not hasattr(base, "predict_proba"):
            raise TypeError("The regime classifier must implement predict_proba.")
        self.features_ = list(self.features) if self.features is not None else list(X.columns)
        mask = y.notna().to_numpy() & X[self.features_].notna().all(axis=1).to_numpy()
        y_fit = y[mask].astype(int)
        self.estimator_ = clone(base).fit(X.loc[mask, self.features_], y_fit)
        self.classes_ = np.asarray(self.estimator_.classes_)
        self.n_samples_fit_ = int(mask.sum())
        self.class_counts_ = y_fit.value_counts().sort_index().reindex(self.classes_, fill_value=0)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator_.predict_proba(X[self.features_])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def compare_classifiers(
    X: pd.DataFrame,
    y: pd.Series,
    candidates: Mapping[str, BaseEstimator] | None = None,
    feature_sets: Mapping[str, Sequence[str]] | None = None,
    n_splits: int = 5,
    gap: int = 0,
) -> pd.DataFrame:
    """
    Cross-validate regime classifiers with forward-chaining (time-ordered) splits.

    Folds never shuffle time, so hours of the same day or week cannot appear in
    both the training and the validation fold.

    Returns a DataFrame with mean and std of accuracy and macro F1 per
    (feature set, classifier), sorted by macro F1.
    """
    candidates = candidates or default_classifiers()
    feature_sets = feature_sets or {"all": list(X.columns)}
    cv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    rows = []
    for set_name, cols in feature_sets.items():
        mask = y.notna().to_numpy() & X[list(cols)].notna().all(axis=1).to_numpy()
        Xs, ys = X.loc[mask, list(cols)], y[mask].astype(int)
        for clf_name, clf in candidates.items():
            scores = cross_validate(clone(clf), Xs, ys, cv=cv, scoring=("accuracy", "f1_macro"))
            rows.append(
                {
                    "feature_set": set_name,
                    "classifier": clf_name,
                    "n_features": len(cols),
                    "accuracy_mean": scores["test_accuracy"].mean(),
                    "accuracy_std": scores["test_accuracy"].std(),
                    "f1_macro_mean": scores["test_f1_macro"].mean(),
                    "f1_macro_std": scores["test_f1_macro"].std(),
                }
            )
    return pd.DataFrame(rows).sort_values("f1_macro_mean", ascending=False, ignore_index=True)
