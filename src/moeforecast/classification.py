"""Classifiers that predict the regime (cluster) of each time step from its features."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from moeforecast.tuning import classification_time_series_splits, default_search_spaces, search_estimator


def default_classifiers(
    random_state: int = 0,
    *,
    exclude: Sequence[str] = (),
    extra: Mapping[str, BaseEstimator] | None = None,
) -> dict[str, BaseEstimator]:
    """Return standard probability-capable classifiers with add/remove options."""
    classifiers = {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=random_state
        ),
        "hist_gbm": HistGradientBoostingClassifier(random_state=random_state),
    }
    classifiers.update(extra or {})
    for name in exclude:
        classifiers.pop(name, None)
    return classifiers


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

    def __init__(
        self,
        estimator: BaseEstimator | None = None,
        features: Sequence[str] | None = None,
        search: str | None = None,
        search_space: Mapping[str, object] | None = None,
        n_iter: int = 20,
        n_splits: int = 3,
        random_state: int = 0,
    ):
        self.estimator = estimator
        self.features = features
        self.search = search
        self.search_space = search_space
        self.n_iter = n_iter
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series):
        base = self.estimator if self.estimator is not None else HistGradientBoostingClassifier()
        if not hasattr(base, "predict_proba"):
            raise TypeError("The regime classifier must implement predict_proba.")
        if self.search not in {None, "grid", "random"}:
            raise ValueError("search must be None, 'grid', or 'random'.")
        self.features_ = list(self.features) if self.features is not None else list(X.columns)
        mask = y.notna().to_numpy() & X[self.features_].notna().all(axis=1).to_numpy()
        y_fit = y[mask].astype(int)
        X_fit = X.loc[mask, self.features_]
        if self.search is None:
            self.estimator_ = clone(base).fit(X_fit, y_fit)
            self.search_ = None
        else:
            name = _classifier_search_name(base)
            space = self.search_space
            if space is None:
                try:
                    space = default_search_spaces()[name]
                except KeyError as exc:
                    raise ValueError(
                        f"No default search space for {type(base).__name__}; pass search_space explicitly."
                    ) from exc
            self.search_ = search_estimator(
                clone(base),
                space,
                X_fit,
                y_fit,
                strategy=self.search,
                n_splits=self.n_splits,
                n_iter=self.n_iter,
                scoring="f1_macro",
                random_state=self.random_state,
                cv=classification_time_series_splits(y_fit, self.n_splits),
            )
            self.estimator_ = self.search_.best_estimator_
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
    candidate_mode: str = "replace",
    search: str | None = None,
    search_spaces: Mapping[str, Mapping[str, object]] | None = None,
    n_iter: int = 20,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Cross-validate regime classifiers with forward-chaining (time-ordered) splits.

    Folds never shuffle time, so hours of the same day or week cannot appear in
    both the training and the validation fold.

    Returns a DataFrame with mean and std of accuracy and macro F1 per
    (feature set, classifier), sorted by macro F1. When ``search`` is enabled,
    HPO is nested inside each outer validation fold to avoid scoring a model on
    the same folds used to choose its hyperparameters.
    """
    if candidate_mode not in {"replace", "extend"}:
        raise ValueError("candidate_mode must be 'replace' or 'extend'.")
    if candidates is None:
        candidates = default_classifiers()
    elif candidate_mode == "extend":
        candidates = {**default_classifiers(), **candidates}
    if not candidates:
        raise ValueError("At least one classifier candidate is required.")
    if search not in {None, "grid", "random"}:
        raise ValueError("search must be None, 'grid', or 'random'.")
    spaces = default_search_spaces()
    spaces.update(search_spaces or {})
    feature_sets = feature_sets or {"all": list(X.columns)}
    rows = []
    for set_name, cols in feature_sets.items():
        mask = y.notna().to_numpy() & X[list(cols)].notna().all(axis=1).to_numpy()
        Xs, ys = X.loc[mask, list(cols)], y[mask].astype(int)
        cv = classification_time_series_splits(ys, n_splits=n_splits, gap=gap)
        for clf_name, clf in candidates.items():
            params = []
            if search is not None:
                space = spaces.get(clf_name)
                if space is None:
                    inferred = _classifier_search_name(clf)
                    space = spaces.get(inferred)
                if space is None:
                    raise ValueError(f"No search space configured for classifier '{clf_name}'.")
                accuracy_scores, f1_scores = [], []
                for fold, (train_idx, test_idx) in enumerate(cv):
                    inner_splits = min(n_splits, len(train_idx) - 1)
                    if inner_splits < 2:
                        raise ValueError("Not enough training samples for nested classifier HPO.")
                    inner_cv = classification_time_series_splits(ys.iloc[train_idx], n_splits=inner_splits)
                    optimized = search_estimator(
                        clone(clf),
                        space,
                        Xs.iloc[train_idx],
                        ys.iloc[train_idx],
                        strategy=search,
                        n_splits=inner_splits,
                        n_iter=n_iter,
                        scoring="f1_macro",
                        random_state=random_state + fold,
                        cv=inner_cv,
                    )
                    prediction = optimized.best_estimator_.predict(Xs.iloc[test_idx])
                    accuracy_scores.append(accuracy_score(ys.iloc[test_idx], prediction))
                    f1_scores.append(f1_score(ys.iloc[test_idx], prediction, average="macro"))
                    params.append(optimized.best_params_)
                scores = {
                    "test_accuracy": np.asarray(accuracy_scores),
                    "test_f1_macro": np.asarray(f1_scores),
                }
            else:
                scores = cross_validate(clone(clf), Xs, ys, cv=cv, scoring=("accuracy", "f1_macro"))
            rows.append(
                {
                    "feature_set": set_name,
                    "classifier": clf_name,
                    "n_features": len(cols),
                    "n_folds": len(cv),
                    "accuracy_mean": scores["test_accuracy"].mean(),
                    "accuracy_std": scores["test_accuracy"].std(),
                    "f1_macro_mean": scores["test_f1_macro"].mean(),
                    "f1_macro_std": scores["test_f1_macro"].std(),
                    "best_params_by_fold": params,
                }
            )
    return pd.DataFrame(rows).sort_values("f1_macro_mean", ascending=False, ignore_index=True)


def _classifier_search_name(estimator: BaseEstimator) -> str:
    """Map built-in classifier types to the public default search-space keys."""
    if isinstance(estimator, RandomForestClassifier):
        return "random_forest_classifier"
    if isinstance(estimator, HistGradientBoostingClassifier):
        return "hist_gbm_classifier"
    if isinstance(estimator, LogisticRegression):
        return "logistic_estimator"
    if hasattr(estimator, "named_steps"):
        final = list(estimator.named_steps.values())[-1]
        if isinstance(final, LogisticRegression):
            return "logistic"
        return _classifier_search_name(final)
    return ""
