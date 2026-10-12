"""The end-to-end Mixture-of-Experts forecaster."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from moeforecast.classification import RegimeClassifier
from moeforecast.clustering import ProfileClusterer
from moeforecast.data import prepare_series
from moeforecast.experts import ExpertSelector, format_key
from moeforecast.features import FeatureBuilder
from moeforecast.gating import AdaptiveGating


def _default_clusterers() -> list[ProfileClusterer]:
    return [ProfileClusterer("D", name="day"), ProfileClusterer("W", name="week")]


class MoEForecaster(BaseEstimator):
    """
    Mixture-of-Experts one-step-ahead forecaster.

    ``fit`` runs the whole construction on a training series:

    1. ``features`` builds the causal feature matrix.
    2. Each clusterer in ``clusterers`` finds regimes of one seasonal profile
       (by default: day types and week types).
    3. One ``RegimeClassifier`` per clusterer learns to predict the regime of
       a time step from its features (these drive the gating).
    4. ``selector`` picks and trains one expert per combination of regimes.
    5. ``gating`` combines the experts at prediction time.

    Every step only sees the series passed to ``fit``. To evaluate on a test
    period, fit on the training part and predict on the full series (see
    :func:`moeforecast.temporal_train_test_split`).

    Parameters
    ----------
    features : FeatureBuilder, optional
    clusterers : sequence of ProfileClusterer, optional
        Defaults to daily ("D") and weekly ("W") profiles.
    classifier : classifier with predict_proba, optional
        Estimator used by every RegimeClassifier.
    classifier_features : sequence of str, optional
        Features given to the regime classifiers (default: all).
    selector : ExpertSelector, optional
    gating : HardGating, SoftGating or AdaptiveGating, optional
        Defaults to ``AdaptiveGating()``.
    """

    def __init__(
        self,
        features: FeatureBuilder | None = None,
        clusterers: Sequence[ProfileClusterer] | None = None,
        classifier: BaseEstimator | None = None,
        classifier_features: Sequence[str] | None = None,
        selector: ExpertSelector | None = None,
        gating: BaseEstimator | None = None,
    ):
        self.features = features
        self.clusterers = clusterers
        self.classifier = classifier
        self.classifier_features = classifier_features
        self.selector = selector
        self.gating = gating

    # ------------------------------------------------------------------ fitting
    def fit(self, y: pd.Series):
        y = _ensure_series(y)
        self.features_ = clone(self.features or FeatureBuilder()).fit(y)
        X = self.features_.transform(y)

        self.clusterers_ = [clone(c).fit(y) for c in (self.clusterers or _default_clusterers())]
        self.dimensions_ = [c.name_ for c in self.clusterers_]
        if len(set(self.dimensions_)) != len(self.dimensions_):
            raise ValueError(f"Clusterer names must be unique, got {self.dimensions_}")
        labels = pd.concat([c.labels_for(y.index) for c in self.clusterers_], axis=1)
        self.regime_labels_ = labels

        self.classifiers_ = [
            RegimeClassifier(self.classifier, self.classifier_features).fit(X, labels[dim])
            for dim in self.dimensions_
        ]

        complete = labels.notna().all(axis=1).to_numpy()
        keys = pd.Series(
            [
                tuple(int(v) for v in row) if ok else None
                for row, ok in zip(labels.to_numpy(), complete, strict=True)
            ],
            index=y.index,
            dtype=object,
        )
        self.selector_ = clone(self.selector or ExpertSelector()).fit(X, y, keys)
        self.experts_ = self.selector_.experts_
        self.expert_keys_ = self.selector_.expert_keys_
        self.expert_names_ = [format_key(k, self.dimensions_) for k in self.expert_keys_]
        self.gating_ = clone(self.gating or AdaptiveGating())
        self.freq_ = y.index.freqstr
        return self

    # --------------------------------------------------------------- prediction
    def predict(self, y: pd.Series) -> pd.Series:
        """
        One-step-ahead predictions for every timestamp of ``y`` with enough history.

        The prediction at ``t`` uses only values of ``y`` before ``t``. Timestamps
        whose features are incomplete get NaN.
        """
        return self.predict_with_metadata(y)[0]

    def predict_with_metadata(self, y: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
        """
        Predictions plus, for each timestamp, the gating details.

        Returns
        -------
        predictions : Series
        metadata : DataFrame with columns ``expert`` (expert with the largest
            weight), ``weight`` (its weight), ``n_experts`` (experts with
            non-zero weight), ``mode`` ("hard"/"soft") and ``confidence``.
        """
        y = _ensure_series(y, freq=self.freq_)
        X = self.features_.transform(y)
        ok = X.notna().all(axis=1).to_numpy()
        preds = pd.Series(np.nan, index=y.index, name=y.name)
        meta = pd.DataFrame(index=y.index, columns=["expert", "weight", "n_experts", "mode", "confidence"])
        if not ok.any():
            return preds, meta

        Xv = X[ok]
        weights, result = self._weights(Xv)
        values = np.zeros(len(Xv))
        for e, key in enumerate(self.expert_keys_):
            used = weights[:, e] > 0
            if used.any():
                values[used] += weights[used, e] * self.experts_[key].predict(Xv[used])
        preds[ok] = values

        top = weights.argmax(axis=1)
        meta.loc[ok, "expert"] = np.asarray(self.expert_names_, dtype=object)[top]
        meta.loc[ok, "weight"] = weights[np.arange(len(top)), top]
        meta.loc[ok, "n_experts"] = (weights > 0).sum(axis=1)
        meta.loc[ok, "mode"] = result.mode
        meta.loc[ok, "confidence"] = result.confidence
        return preds, meta.infer_objects()

    def expert_weights(self, y: pd.Series) -> pd.DataFrame:
        """Gating weight of every expert at every timestamp (columns = expert names)."""
        y = _ensure_series(y, freq=self.freq_)
        X = self.features_.transform(y)
        ok = X.notna().all(axis=1).to_numpy()
        out = pd.DataFrame(np.nan, index=y.index, columns=self.expert_names_)
        if ok.any():
            out.loc[ok] = self._weights(X[ok])[0]
        return out

    def forecast(self, y: pd.Series, steps: int) -> pd.Series:
        """
        Recursive multi-step forecast of the ``steps`` periods after the end of ``y``.

        Each forecast is fed back as the most recent observation for the next
        step, so errors accumulate with the horizon.
        """
        y = _ensure_series(y, freq=self.freq_)
        history = self.features_.required_history_
        if len(y) < history:
            raise ValueError(f"At least {history} past values are needed, got {len(y)}.")
        future = pd.date_range(y.index[-1], periods=steps + 1, freq=self.freq_)[1:]
        series = pd.concat([y.iloc[-history:], pd.Series(np.nan, index=future)])
        series.index.freq = self.freq_
        series.name = y.name
        for t in future:
            window = series.loc[:t].iloc[-(history + 1) :]
            series[t] = self.predict(window).iloc[-1]
        return series[future]

    # ----------------------------------------------------------------- internal
    def _weights(self, X: pd.DataFrame):
        probas = [clf.predict_proba(X) for clf in self.classifiers_]
        classes = [clf.classes_ for clf in self.classifiers_]
        result = self.gating_.route(probas, classes, self.expert_keys_)
        return result.weights, result


def _ensure_series(y, freq: str | None = None) -> pd.Series:
    if not isinstance(y, pd.Series):
        raise TypeError("Expected a pandas Series; use moeforecast.prepare_series() first.")
    if isinstance(y.index, pd.DatetimeIndex) and y.index.freq is not None:
        if freq is not None and y.index.freqstr != freq:
            raise ValueError(f"Series frequency '{y.index.freqstr}' differs from the fitted '{freq}'.")
        return y
    return prepare_series(y, freq=freq)
