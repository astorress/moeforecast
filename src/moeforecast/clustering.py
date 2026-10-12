"""Regime discovery by clustering seasonal profiles (e.g. daily or weekly curves)."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

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
        Pandas period alias that defines one profile, e.g. "D" or "W".
    n_clusters : int, optional
        Fixed number of clusters. If None, it is chosen in ``k_range`` by
        ``selection_metric``.
    k_range : iterable of int
        Candidate numbers of clusters.
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
    """

    def __init__(
        self,
        profile: str = "D",
        n_clusters: int | None = None,
        k_range: Iterable[int] = range(2, 9),
        normalize: str | None = None,
        selection_metric: str = "silhouette",
        name: str | None = None,
        random_state: int = 0,
    ):
        self.profile = profile
        self.n_clusters = n_clusters
        self.k_range = k_range
        self.normalize = normalize
        self.selection_metric = selection_metric
        self.name = name
        self.random_state = random_state

    # ------------------------------------------------------------------ fitting
    def fit(self, y: pd.Series):
        if self.normalize not in _NORMALIZATIONS:
            raise ValueError(f"normalize must be one of {_NORMALIZATIONS}")
        self.name_ = self.name or self.profile
        profiles = self.profile_matrix(y)
        if len(profiles) < 3:
            raise ValueError(f"Only {len(profiles)} complete '{self.profile}' profiles in the data.")
        X = self._normalize(profiles.to_numpy())

        rows = []
        models = {}
        ks = [self.n_clusters] if self.n_clusters else [k for k in self.k_range if k < len(X)]
        for k in ks:
            km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state).fit(X)
            models[k] = km
            rows.append(
                {
                    "k": k,
                    "silhouette": silhouette_score(X, km.labels_),
                    "davies_bouldin": davies_bouldin_score(X, km.labels_),
                    "calinski_harabasz": calinski_harabasz_score(X, km.labels_),
                    "inertia": km.inertia_,
                }
            )
        self.scores_ = pd.DataFrame(rows).set_index("k")

        if self.n_clusters:
            best_k = self.n_clusters
        elif self.selection_metric == "davies_bouldin":
            best_k = int(self.scores_["davies_bouldin"].idxmin())
        elif self.selection_metric in ("silhouette", "calinski_harabasz"):
            best_k = int(self.scores_[self.selection_metric].idxmax())
        else:
            raise ValueError(f"Unknown selection_metric '{self.selection_metric}'")

        km = models[best_k]
        raw_means = pd.DataFrame(profiles.to_numpy()).groupby(km.labels_).mean().mean(axis=1)
        order = np.argsort(raw_means.to_numpy())  # old label at each new position
        relabel = np.empty_like(order)
        relabel[order] = np.arange(len(order))

        self.n_clusters_ = best_k
        self.kmeans_ = km
        self.relabel_ = relabel
        self.labels_ = pd.Series(relabel[km.labels_], index=profiles.index, name=self.name_)
        self.centroids_ = km.cluster_centers_[order]
        self.profiles_ = profiles
        self.mean_profiles_ = profiles.groupby(self.labels_.to_numpy()).mean()
        self.classes_ = np.arange(best_k)
        return self

    # --------------------------------------------------------------- utilities
    def profile_matrix(self, y: pd.Series) -> pd.DataFrame:
        """Rows = complete periods, columns = position (step) within the period."""
        step = step_of(y.index.freqstr)
        periods = y.index.to_period(self.profile)
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
        """Mean profile of each cluster over all individual profiles."""
        import matplotlib.pyplot as plt

        ax = ax or plt.subplots(figsize=(10, 4))[1]
        for label, curve in self.mean_profiles_.iterrows():
            members = self.profiles_[self.labels_.to_numpy() == label]
            ax.plot(members.T.to_numpy(), color=f"C{label}", alpha=0.03, linewidth=0.5)
            ax.plot(
                curve.to_numpy(),
                color=f"C{label}",
                linewidth=2.5,
                label=f"{self.name_} {label} (n={len(members)})",
            )
        ax.set_title(f"Mean '{self.name_}' profile per cluster")
        ax.set_xlabel("Step within period")
        ax.set_ylabel("Value")
        ax.legend()
        return ax

    def plot_scores(self, axes=None):
        """Selection metrics for every candidate k."""
        import matplotlib.pyplot as plt

        metrics = ["silhouette", "davies_bouldin", "calinski_harabasz"]
        if axes is None:
            _, axes = plt.subplots(1, len(metrics), figsize=(13, 3.5))
        for ax, metric in zip(axes, metrics, strict=True):
            self.scores_[metric].plot(ax=ax, marker="o")
            ax.axvline(self.n_clusters_, color="gray", linestyle="--")
            ax.set_title(metric)
            ax.set_xlabel("k")
        return axes
