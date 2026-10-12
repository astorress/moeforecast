"""The end-to-end Mixture-of-Experts forecaster."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import ParameterGrid, ParameterSampler

from moeforecast.classification import RegimeClassifier
from moeforecast.clustering import ProfileClusterer
from moeforecast.data import prepare_series
from moeforecast.experts import ExpertSelector, format_key
from moeforecast.features import FeatureBuilder
from moeforecast.gating import AdaptiveGating
from moeforecast.metrics import mae, mape, rmse


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
        Required. Evaluate candidate cluster counts, inspect validation plots,
        then pass fitted clusterer configurations with explicit ``n_clusters``.
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
        classifier_search: str | None = None,
        classifier_search_spaces: dict[str, object] | None = None,
        classifier_search_n_iter: int = 20,
        classifier_search_n_splits: int = 3,
        random_state: int = 0,
    ):
        self.features = features
        self.clusterers = clusterers
        self.classifier = classifier
        self.classifier_features = classifier_features
        self.selector = selector
        self.gating = gating
        self.classifier_search = classifier_search
        self.classifier_search_spaces = classifier_search_spaces
        self.classifier_search_n_iter = classifier_search_n_iter
        self.classifier_search_n_splits = classifier_search_n_splits
        self.random_state = random_state

    # ------------------------------------------------------------------ fitting
    def fit(self, y: pd.Series):
        y = _ensure_series(y)
        self.training_series_ = y.copy()
        self.features_ = clone(self.features or FeatureBuilder()).fit(y)
        X = self.features_.transform(y)

        if not self.clusterers:
            raise ValueError(
                "Pass clusterers with explicit n_clusters. Use ProfileClusterer.evaluate(y) and "
                "its validation plots to choose the cluster counts first."
            )
        self.clusterers_ = [clone(c).fit(y) for c in self.clusterers]
        self.dimensions_ = [c.name_ for c in self.clusterers_]
        if len(set(self.dimensions_)) != len(self.dimensions_):
            raise ValueError(f"Clusterer names must be unique, got {self.dimensions_}")
        labels = pd.concat([c.labels_for(y.index) for c in self.clusterers_], axis=1)
        self.regime_labels_ = labels

        self.classifiers_ = [
            RegimeClassifier(
                self.classifier,
                self.classifier_features,
                search=self.classifier_search,
                search_space=self.classifier_search_spaces,
                n_iter=self.classifier_search_n_iter,
                n_splits=self.classifier_search_n_splits,
                random_state=self.random_state,
            ).fit(X, labels[dim])
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

    def inspect(self) -> dict[str, object]:
        """Expose fitted estimators and diagnostics for every MoE component.

        The returned scikit-learn estimators are the actual fitted objects, not
        summaries or copies. Diagnostic DataFrames and label Series are copied
        so editing a report does not mutate the forecaster.

        Returns
        -------
        dict
            Configuration, features, regime labels, cluster results, fitted
            classifiers, expert-selection scores, fitted experts, and gating.
            Cluster entries include the estimator, candidate-k scores,
            assignments, raw profiles, raw mean profiles, and cluster summary.
        """
        if not hasattr(self, "selector_"):
            raise RuntimeError("MoEForecaster must be fitted before it can be inspected.")

        clusters = {}
        for clusterer in self.clusterers_:
            clusters[clusterer.name_] = {
                "estimator": clusterer.kmeans_,
                "clusterer": clusterer,
                "scores": clusterer.scores_.copy(),
                "assignments": clusterer.labels_.copy(),
                "profiles": clusterer.profiles_.copy(),
                "mean_profiles": clusterer.mean_profiles_.copy(),
                "summary": clusterer.summary(),
            }

        classifiers = {}
        for dimension, classifier in zip(self.dimensions_, self.classifiers_, strict=True):
            classifiers[dimension] = {
                "wrapper": classifier,
                "estimator": classifier.estimator_,
                "features": classifier.features_.copy(),
                "classes": classifier.classes_.copy(),
                "class_counts": classifier.class_counts_.copy(),
                "n_samples_fit": classifier.n_samples_fit_,
                "search": classifier.search_,
                "best_params": classifier.search_.best_params_ if classifier.search_ is not None else {},
            }

        expert_names = dict(zip(self.expert_keys_, self.expert_names_, strict=True))
        expert_selection = self.selector_.results_.copy()
        expert_selection.insert(
            1,
            "expert_name",
            expert_selection["expert"].map(expert_names).fillna("not trained"),
        )

        return {
            "configuration": {
                "frequency": self.freq_,
                "features": self.features_,
                "clusterers": self.clusterers_,
                "classifier": self.classifier,
                "classifier_features": self.classifier_features,
                "selector": self.selector_,
                "gating": self.gating_,
                "classifier_search": self.classifier_search,
            },
            "features": self.features_.feature_names_out_.copy(),
            "regime_labels": self.regime_labels_.copy(),
            "clusters": clusters,
            "classifiers": classifiers,
            "expert_selection": expert_selection,
            "candidate_experts": self.selector_.candidate_estimators_.copy(),
            "expert_hyperparameter_searches": {
                key: {name: results.copy() for name, results in models.items()}
                for key, models in self.selector_.search_results_.items()
            },
            "experts": {expert_names[key]: estimator for key, estimator in self.experts_.items()},
            "gating": self.gating_,
        }

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

    def tune_gating(
        self,
        validation: pd.Series,
        *,
        search: str = "grid",
        metric: str = "mae",
        thresholds: Sequence[float] = (0.55, 0.65, 0.75, 0.85, 0.95),
        min_weights: Sequence[float] = (0.0, 0.025, 0.05, 0.1),
        temperatures: Sequence[float] = (0.75, 1.0, 1.5),
        n_iter: int = 20,
        random_state: int | None = None,
    ) -> pd.DataFrame:
        """Tune adaptive-gating thresholds on a separate, later validation series.

        The forecaster and its experts are not refitted. Supply a validation
        period that was not used in ``fit``; predictions are one-step-ahead, so
        each validation timestamp may use the preceding observed history.
        The best setting replaces ``gating_`` and the complete score table is
        available as ``gating_results_``.
        """
        if search not in {"grid", "random"}:
            raise ValueError("search must be 'grid' or 'random'.")
        score_fn = {"mae": mae, "rmse": rmse, "mape": mape}.get(metric)
        if score_fn is None:
            raise ValueError("metric must be 'mae', 'rmse', or 'mape'.")
        if len(thresholds) == 0 or len(min_weights) == 0 or len(temperatures) == 0:
            raise ValueError("thresholds, min_weights, and temperatures must be non-empty.")
        if any(not 0 <= value <= 1 for value in thresholds) or any(
            not 0 <= value <= 1 for value in min_weights
        ):
            raise ValueError("thresholds and min_weights must be between 0 and 1.")
        if any(value <= 0 for value in temperatures):
            raise ValueError("temperatures must be greater than zero.")
        if search == "random" and n_iter < 1:
            raise ValueError("n_iter must be a positive integer.")
        validation = _ensure_series(validation, freq=self.freq_)
        if validation.index[0] <= self.training_series_.index[-1]:
            raise ValueError("Gating validation must start strictly after the series used in fit().")
        combined = pd.concat([self.training_series_, validation])
        X = self.features_.transform(combined).loc[validation.index]
        valid = X.notna().all(axis=1).to_numpy() & validation.notna().to_numpy()
        if not valid.any():
            raise ValueError("Validation series has no rows with complete features and target values.")
        Xv = X.loc[valid]
        actual = validation.loc[valid]
        probas = [clf.predict_proba(Xv) for clf in self.classifiers_]
        classes = [clf.classes_ for clf in self.classifiers_]
        expert_predictions = np.column_stack([self.experts_[key].predict(Xv) for key in self.expert_keys_])
        parameter_space = {
            "threshold": list(thresholds),
            "min_weight": list(min_weights),
            "temperature": list(temperatures),
        }
        configs = list(ParameterGrid(parameter_space))
        if search == "random":
            configs = list(
                ParameterSampler(
                    parameter_space,
                    n_iter=min(n_iter, len(configs)),
                    random_state=self.random_state if random_state is None else random_state,
                )
            )
        rows = []
        for config in configs:
            gate = AdaptiveGating(**config)
            routed = gate.route(probas, classes, self.expert_keys_).weights
            prediction = np.sum(routed * expert_predictions, axis=1)
            rows.append({**config, f"{metric}_validation": score_fn(actual, prediction)})
        self.gating_results_ = pd.DataFrame(rows).sort_values(f"{metric}_validation", ignore_index=True)
        best = self.gating_results_.iloc[0]
        self.gating_ = AdaptiveGating(
            float(best["threshold"]), float(best["min_weight"]), float(best["temperature"])
        )
        self.gating_search_ = {"strategy": search, "metric": metric, "n_validation": len(actual)}
        return self.gating_results_.copy()

    def plot_model(self, path=None, *, dpi: int = 180):
        """Draw the fitted MoE architecture and optionally save it to ``path``."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

        if not hasattr(self, "selector_"):
            raise RuntimeError("MoEForecaster must be fitted before its diagram can be drawn.")
        n_dimensions = len(self.dimensions_)
        n_experts = len(self.expert_keys_)
        fig, ax = plt.subplots(figsize=(13, max(5, 0.42 * max(n_dimensions, n_experts))))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        def box(x, y, width, height, title, detail, color):
            patch = FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                linewidth=1.2,
                edgecolor=color,
                facecolor=color + "18",
            )
            ax.add_patch(patch)
            ax.text(x + width / 2, y + height * 0.62, title, ha="center", va="center", fontweight="bold")
            ax.text(x + width / 2, y + height * 0.28, detail, ha="center", va="center", fontsize=8, wrap=True)
            return (x, y, width, height)

        def connect(left, right):
            ax.add_patch(
                FancyArrowPatch(
                    (left[0] + left[2], left[1] + left[3] / 2),
                    (right[0], right[1] + right[3] / 2),
                    arrowstyle="-|>",
                    mutation_scale=11,
                    linewidth=0.9,
                    color="#778899",
                )
            )

        left_x, mid_x, gate_x, right_x = 0.03, 0.29, 0.57, 0.79
        h = min(0.105, 0.72 / max(n_dimensions, n_experts))
        def positions(count):
            gap = (0.84 - count * h) / max(count - 1, 1)
            return [0.92 - h - i * (h + gap) for i in range(count)]

        feature = box(
            left_x,
            0.45,
            0.19,
            0.12,
            "Input features",
            f"{len(self.features_.feature_names_out_)} causal predictors",
            "#315b7d",
        )
        cluster_details = "\n".join(
            f"{cluster.name_}: k={cluster.n_clusters_}, {cluster.profile}, {cluster.distance}"
            for cluster in self.clusterers_
        )
        cluster_box = box(
            left_x,
            0.08,
            0.19,
            0.18,
            "Profile clusters",
            cluster_details,
            "#7b6b9a",
        )
        classifier_boxes = []
        for dim, clf, y_pos in zip(self.dimensions_, self.classifiers_, positions(n_dimensions), strict=True):
            fitted = clf.estimator_
            best_params = clf.search_.best_params_ if clf.search_ is not None else {}
            detail = (
                f"{type(fitted).__name__} - {len(clf.classes_)} regimes\n"
                f"{best_params or 'default parameters'}"
            )
            classifier_boxes.append(box(mid_x, y_pos, 0.23, h, f"Classifier: {dim}", detail, "#397b68"))
        gate = box(
            gate_x,
            0.42,
            0.18,
            0.16,
            type(self.gating_).__name__,
            str(self.gating_.get_params()),
            "#ad7133",
        )
        expert_boxes = []
        name_by_key = dict(zip(self.expert_keys_, self.expert_names_, strict=True))
        for key, y_pos in zip(self.expert_keys_, positions(n_experts), strict=True):
            estimator = self.experts_[key]
            selected_row = self.selector_.results_.loc[
                self.selector_.results_["expert"].map(key.__eq__)
                & self.selector_.results_["selected"]
            ].iloc[0]
            params = selected_row.get("best_params", {})
            detail = f"{type(estimator).__name__}\n{params or 'default parameters'}"
            expert_boxes.append(box(right_x, y_pos, 0.19, h, name_by_key[key], detail, "#9b5364"))
        for item in classifier_boxes:
            connect(feature, item)
            connect(cluster_box, item)
            connect(item, gate)
        for item in expert_boxes:
            connect(gate, item)
        ax.text(
            0.5,
            0.97,
            "Fitted Mixture-of-Experts Architecture",
            ha="center",
            va="top",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout()
        if path is not None:
            fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        return fig

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
