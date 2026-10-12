"""Regime discovery by clustering seasonal profiles (e.g. daily or weekly curves)."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)

from moeforecast._plotting import PALETTE, style_axis
from moeforecast.data import step_of

_NORMALIZATIONS = (None, "zscore", "mean")


class ProfileClusterer(BaseEstimator):
    """
    Cluster the profiles of a seasonal period into regimes.

    The series is cut into consecutive calendar periods (``profile="D"`` gives
    one 24-value curve per day for hourly data, ``profile="W"`` one 168-value
    curve per Monday-to-Sunday week). Incomplete periods are discarded. Each
    remaining profile is clustered with k-means.

    Labels are ordered deterministically by the mean level of the cluster's
    raw profile (label 0 = lowest mean demand), so they are stable across runs.

    Parameters
    ----------
    profile : str
        Calendar period that defines one profile: daily ("D"), weekly ("W")
        or monthly ("M").
    n_clusters : int, optional
        Fixed number of clusters. If None, it is chosen in ``k_range`` by
        ``selection_metric``.
    k_range : iterable of int
        Candidate numbers of clusters evaluated before choosing ``n_clusters``.
    normalize : {None, "zscore", "mean"}
        Profile normalization before clustering. None clusters raw values
        (level and shape); "zscore" clusters shape only; "mean" divides by the
        profile mean (relative shape, keeps amplitude ratios).
    selection_metric : {"silhouette", "davies_bouldin", "calinski_harabasz"}
        Criterion used to choose k when ``n_clusters`` is None.
    name : str, optional
        Name of the regime dimension (used in expert identifiers and plots).
        Defaults to ``profile``.
    random_state : int
        Seed for k-means.
    distance : {"euclidean", "dtw"}
        Distance used for clustering. DTW requires the optional ``tslearn`` extra.
    auto_select : bool
        Automatically select ``k`` by ``selection_metric`` when no
        ``n_clusters`` is provided. Disabled by default so users can inspect
        validation plots and choose a cluster count explicitly.
    """

    def __init__(
        self,
        profile: str = "D",
        n_clusters: int | None = None,
        k_range: Iterable[int] = range(2, 11),
        normalize: str | None = None,
        selection_metric: str = "silhouette",
        name: str | None = None,
        random_state: int = 0,
        distance: str = "euclidean",
        auto_select: bool = False,
    ):
        self.profile = profile
        self.n_clusters = n_clusters
        self.k_range = k_range
        self.normalize = normalize
        self.selection_metric = selection_metric
        self.name = name
        self.random_state = random_state
        self.distance = distance
        self.auto_select = auto_select

    # ------------------------------------------------------------------ fitting
    def evaluate(self, y: pd.Series):
        """Score candidate cluster counts without fitting a final clusterer.

        Inspect the returned table and call ``plot_scores`` and
        ``plot_silhouette`` before choosing the final cluster count. Then set
        ``n_clusters`` and call ``fit(y)``.
        """
        if self.normalize not in _NORMALIZATIONS:
            raise ValueError(f"normalize must be one of {_NORMALIZATIONS}")
        if self.distance not in {"euclidean", "dtw"}:
            raise ValueError("distance must be 'euclidean' or 'dtw'.")
        self.name_ = self.name or self.profile
        profiles = self.profile_matrix(y)
        if len(profiles) < 3:
            raise ValueError(f"Only {len(profiles)} complete '{self.profile}' profiles in the data.")
        X = self._normalize(profiles.to_numpy())
        ks = sorted({int(k) for k in self.k_range if 2 <= int(k) < len(X)})
        if not ks:
            raise ValueError(
                "k_range must contain at least one candidate k >= 2 and smaller than profile count."
            )

        distance_matrix = None
        if self.distance == "dtw":
            try:
                from tslearn.clustering import TimeSeriesKMeans
                from tslearn.metrics import cdist_dtw
            except ImportError as err:
                raise ImportError(
                    "DTW clustering requires the optional dependency: pip install moeforecast[dtw]"
                ) from err
            distance_matrix = cdist_dtw(X[:, :, None])
            if self.selection_metric != "silhouette":
                raise ValueError(
                    "DTW supports silhouette selection only; Davies-Bouldin and Calinski-Harabasz "
                    "are Euclidean metrics."
                )
        else:
            TimeSeriesKMeans = None

        rows = []
        models = {}
        for k in ks:
            if self.distance == "euclidean":
                model = KMeans(n_clusters=k, n_init=10, random_state=self.random_state).fit(X)
            else:
                model = TimeSeriesKMeans(
                    n_clusters=k, metric="dtw", n_init=2, max_iter=20, random_state=self.random_state
                ).fit(X[:, :, None])
            models[k] = model
            labels = model.labels_
            rows.append(
                {
                    "k": k,
                    "silhouette": silhouette_score(
                        distance_matrix if distance_matrix is not None else X,
                        labels,
                        metric="precomputed" if distance_matrix is not None else "euclidean",
                    ),
                    "davies_bouldin": davies_bouldin_score(X, labels) if distance_matrix is None else np.nan,
                    "calinski_harabasz": calinski_harabasz_score(X, labels)
                    if distance_matrix is None
                    else np.nan,
                    "inertia": float(model.inertia_),
                }
            )
        self.evaluation_scores_ = pd.DataFrame(rows).set_index("k")
        self.evaluation_profiles_ = profiles
        self.evaluation_matrix_ = X
        self._evaluation_models_ = models
        self.dtw_distances_ = distance_matrix
        self.n_profiles_ = len(profiles)
        return self.evaluation_scores_.copy()

    def fit(self, y: pd.Series):
        if self.normalize not in _NORMALIZATIONS:
            raise ValueError(f"normalize must be one of {_NORMALIZATIONS}")
        if self.n_clusters is None and not self.auto_select:
            self.evaluate(y)
            raise ValueError(
                "Choose the cluster count explicitly: call evaluate(y), inspect plot_scores(), "
                "then set n_clusters and call fit(y). Set auto_select=True to opt into automatic selection."
            )
        scores = self.evaluate(y)
        profiles = self.evaluation_profiles_
        X = self.evaluation_matrix_
        if self.n_clusters is not None:
            best_k = int(self.n_clusters)
            if best_k < 2 or best_k >= len(X):
                raise ValueError(f"n_clusters must be between 2 and {len(X) - 1}.")
        elif self.selection_metric == "davies_bouldin":
            best_k = int(scores["davies_bouldin"].idxmin())
        elif self.selection_metric in ("silhouette", "calinski_harabasz"):
            best_k = int(scores[self.selection_metric].idxmax())
        else:
            raise ValueError(f"Unknown selection_metric '{self.selection_metric}'")

        if best_k in self._evaluation_models_:
            km = self._evaluation_models_[best_k]
        elif self.distance == "euclidean":
            km = KMeans(n_clusters=best_k, n_init=10, random_state=self.random_state).fit(X)
        else:
            from tslearn.clustering import TimeSeriesKMeans

            km = TimeSeriesKMeans(
                n_clusters=best_k, metric="dtw", n_init=2, max_iter=20, random_state=self.random_state
            ).fit(X[:, :, None])
        raw_means = pd.DataFrame(profiles.to_numpy()).groupby(km.labels_).mean().mean(axis=1)
        order = np.argsort(raw_means.to_numpy())  # old label at each new position
        relabel = np.empty_like(order)
        relabel[order] = np.arange(len(order))

        self.n_clusters_ = best_k
        self.kmeans_ = km
        self.scores_ = scores
        self.relabel_ = relabel
        self.labels_ = pd.Series(relabel[km.labels_], index=profiles.index, name=self.name_)
        centers = np.asarray(km.cluster_centers_).reshape(best_k, -1)
        self.centroids_ = centers[order]
        self.profiles_ = profiles
        self.mean_profiles_ = profiles.groupby(self.labels_.to_numpy()).mean()
        self.classes_ = np.arange(best_k)
        counts = self.labels_.value_counts().reindex(self.classes_, fill_value=0)
        self.cluster_summary_ = pd.DataFrame(
            {
                "n_profiles": counts,
                "mean_level": self.mean_profiles_.mean(axis=1).reindex(self.classes_),
            }
        ).rename_axis("cluster")
        return self

    def summary(self) -> pd.DataFrame:
        """Return profile counts and mean raw value for each fitted cluster."""
        return self.cluster_summary_.copy()

    # --------------------------------------------------------------- utilities
    def profile_matrix(self, y: pd.Series) -> pd.DataFrame:
        """Rows = complete periods, columns = position (step) within the period."""
        monthly_profile = str(self.profile).upper() in {"M", "ME", "MS"}
        monthly_data = monthly_profile and str(y.index.freqstr).upper() in {"M", "ME", "MS"}
        profile_y = y if monthly_data else y.resample("D").mean() if monthly_profile else y
        step = pd.Timedelta(1, "ns") if monthly_data else step_of(y.index.freqstr)
        periods = y.index.to_period("M" if monthly_profile else self.profile)
        if monthly_profile:
            periods = profile_y.index.to_period("M")
            grouped = pd.DataFrame(
                {"period": periods, "value": profile_y.to_numpy()}, index=profile_y.index
            ).groupby("period")
            length = 1 if monthly_data else max(period.days_in_month for period in periods.unique())
            canonical = np.linspace(0.0, 1.0, length) if length > 1 else np.array([0.0])
            rows = []
            labels = []
            for period, values in grouped:
                idx = values.index
                expected_start = period.start_time
                expected_end = period.end_time.floor("D")
                incomplete = not monthly_data and (idx[0] > expected_start or idx[-1] < expected_end)
                if incomplete or values["value"].isna().any():
                    continue
                span = period.end_time - period.start_time
                source_positions = np.asarray((idx - period.start_time) / span, dtype=float)
                if len(values) == 1:
                    profile_values = np.repeat(values["value"].iloc[0], length)
                else:
                    profile_values = np.interp(canonical, source_positions, values["value"].to_numpy())
                rows.append(profile_values)
                labels.append(period)
            return pd.DataFrame(rows, index=pd.PeriodIndex(labels, freq="M"), columns=range(length))
        start = periods.start_time
        position = np.asarray((y.index - start) / step, dtype=int)
        length = int(round((periods[0].end_time - periods[0].start_time + pd.Timedelta(1, "ns")) / step))
        frame = pd.DataFrame({"period": periods, "pos": position, "y": y.to_numpy()})
        matrix = frame.pivot_table(index="period", columns="pos", values="y", aggfunc="first")
        matrix = matrix.reindex(columns=range(length))
        return matrix.dropna()

    def labels_for(self, index: pd.DatetimeIndex) -> pd.Series:
        """Cluster label of each timestamp; NaN for periods not seen in ``fit``."""
        periods = index.to_period(self.profile)
        return pd.Series(self.labels_.reindex(periods).to_numpy(), index=index, name=self.name_)

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        if self.normalize == "zscore":
            std = X.std(axis=1, keepdims=True)
            return (X - X.mean(axis=1, keepdims=True)) / np.where(std == 0, 1, std)
        if self.normalize == "mean":
            mean = X.mean(axis=1, keepdims=True)
            return X / np.where(mean == 0, 1, mean)
        return X

    # ----------------------------------------------------------------- plotting
    def plot_profiles(self, ax=None):
        """Plot cluster means with individual profiles and interquartile bands."""
        import matplotlib.pyplot as plt

        ax = ax or plt.subplots(figsize=(11, 4), layout="constrained")[1]
        positions = np.arange(self.profiles_.shape[1])
        for label, curve in self.mean_profiles_.iterrows():
            members = self.profiles_[self.labels_.to_numpy() == label]
            color = PALETTE[int(label) % len(PALETTE)]
            ax.plot(positions, members.T.to_numpy(), color=color, alpha=0.035, linewidth=0.6)
            ax.fill_between(
                positions,
                members.quantile(0.25).to_numpy(),
                members.quantile(0.75).to_numpy(),
                color=color,
                alpha=0.18,
                linewidth=0,
            )
            ax.plot(
                positions,
                curve.to_numpy(),
                color=color,
                linewidth=2.5,
                label=f"{self.name_} {label} (n={len(members)})",
            )
        if self.profile == "D" and len(positions) == 24:
            ticks = np.arange(0, 24, 3)
            ax.set_xticks(ticks, [f"{hour:02d}:00" for hour in ticks])
            ax.set_xlabel("Hour of day")
        elif self.profile == "W" and len(positions) == 168:
            ticks = np.arange(12, 168, 24)
            ax.set_xticks(ticks, ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
            ax.set_xlabel("Day of week")
        else:
            ax.set_xlabel("Step within period")
        ax.set_title(f"{self.profile} profiles by cluster", loc="left", fontweight="bold")
        ax.set_ylabel(str(self.profiles_.columns.name or "Value"))
        style_axis(ax, grid_axis="y")
        ax.legend(frameon=False, ncol=min(self.n_clusters_, 4))
        return ax

    def plot_scores(self, axes=None):
        """Selection metrics for every candidate k."""
        import matplotlib.pyplot as plt

        scores = getattr(self, "scores_", getattr(self, "evaluation_scores_", None))
        if scores is None:
            raise RuntimeError("Call evaluate(y) or fit(y) before plotting cluster validation scores.")
        metrics = ["silhouette", "davies_bouldin", "calinski_harabasz"]
        if axes is None:
            _, axes = plt.subplots(1, len(metrics), figsize=(14, 4), layout="constrained")
        for ax, metric in zip(np.atleast_1d(axes), metrics, strict=True):
            values = scores[metric].dropna()
            if values.empty:
                ax.text(0.5, 0.5, "Not defined for DTW", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(metric.replace("_", " ").title())
                continue
            color = {
                "silhouette": PALETTE[0],
                "davies_bouldin": "#b85c5c",
                "calinski_harabasz": PALETTE[2],
            }[metric]
            ax.plot(values.index, values.values, marker="o", linewidth=2, color=color)
            if metric == "silhouette":
                best_k = values.idxmax()
                ax.scatter(
                    [best_k],
                    [values.loc[best_k]],
                    s=80,
                    color=PALETTE[1],
                    edgecolor="white",
                    zorder=4,
                    label="Highest silhouette candidate",
                )
                ax.legend(frameon=False)
            if self.n_clusters is not None:
                ax.axvline(
                    self.n_clusters, color="#404040", linestyle="--", linewidth=1.2, label="Selected k"
                )
            ax.set_title(
                {
                    "silhouette": "Silhouette (higher is better)",
                    "davies_bouldin": "Davies-Bouldin (lower is better)",
                    "calinski_harabasz": "Calinski-Harabasz (higher is better)",
                }[metric],
                loc="left",
                fontsize=10,
                fontweight="bold",
            )
            ax.set_xlabel("Number of clusters (k)")
            ax.set_ylabel("Score")
            style_axis(ax, grid_axis="y")
        return axes

    def plot_silhouette(self, k: int | None = None, ax=None):
        """Plot silhouette distributions for candidate k values before selection.

        When called after :meth:`evaluate`, the default is to show all
        evaluated candidates (up to nine panels). Pass ``k`` to inspect one.
        After :meth:`fit`, the selected model is shown by default.
        """
        import matplotlib.pyplot as plt

        if not hasattr(self, "evaluation_scores_"):
            raise RuntimeError("Call evaluate(y) or fit(y) before plotting silhouette diagnostics.")
        models = self._evaluation_models_
        if k is not None:
            if k not in models:
                raise ValueError(f"k={k} was not evaluated; available values are {list(models)}.")
            selected = [k]
        elif hasattr(self, "n_clusters_"):
            selected = [self.n_clusters_]
        else:
            selected = list(models)

        distances = getattr(self, "dtw_distances_", None)
        if ax is not None and len(selected) > 1:
            raise ValueError("Pass a single k when providing a custom axis.")
        if ax is None:
            if len(selected) == 1:
                _, axes = plt.subplots(figsize=(9, 5), layout="constrained")
                axes = [axes]
            else:
                cols = min(3, len(selected))
                rows = int(np.ceil(len(selected) / cols))
                _, axes = plt.subplots(
                    rows, cols, figsize=(5.2 * cols, 3.6 * rows), layout="constrained", squeeze=False
                )
                axes = list(axes.flat)
        else:
            axes = [ax]

        for axis, candidate in zip(axes, selected, strict=False):
            model = models[candidate]
            labels = model.labels_
            samples = silhouette_samples(
                distances if distances is not None else self.evaluation_matrix_,
                labels,
                metric="precomputed" if distances is not None else "euclidean",
            )
            y_lower = 10
            for label in range(candidate):
                vals = np.sort(samples[labels == label])
                y_upper = y_lower + len(vals)
                axis.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    vals,
                    color=PALETTE[label % len(PALETTE)],
                    alpha=0.82,
                )
                axis.text(-0.05, y_lower + len(vals) / 2, str(label), va="center", ha="right")
                y_lower = y_upper + 10
            average = float(samples.mean())
            axis.axvline(
                average,
                color="#d1495b",
                linestyle="--",
                linewidth=1.5,
                label=f"Mean = {average:.3f}",
            )
            axis.axvline(0, color="#555555", linewidth=0.8)
            axis.set_title(f"k={candidate} | mean={average:.3f}", loc="left", fontweight="bold")
            axis.set_xlabel("Silhouette coefficient")
            axis.set_ylabel("Profiles grouped by cluster")
            style_axis(axis, grid_axis="x")
            axis.legend(frameon=False)
        for axis in axes[len(selected) :]:
            axis.set_visible(False)
        return axes[0] if len(axes) == 1 else np.asarray(axes)
