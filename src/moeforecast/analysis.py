"""
Exploratory analysis aimed at configuring the model.

``SeriesAnalyzer`` suggests seasonal periods and lags from the autocorrelation
structure; ``feature_relevance`` ranks candidate features. Both only make
recommendations: the user decides what goes into ``FeatureBuilder``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from statsmodels.tsa.stattools import acf, pacf

from moeforecast.data import step_of


class SeriesAnalyzer:
    """
    Autocorrelation-based recommendations for seasonal periods and lags.

    Parameters
    ----------
    max_lag : int, optional
        Largest lag analysed. Defaults to two weeks of steps for sub-daily
        data and 400 steps otherwise.
    n_periods : int
        Maximum number of seasonal periods to suggest.
    n_lags : int
        Maximum number of lags to suggest from the partial autocorrelation.
    """

    def __init__(self, max_lag: int | None = None, n_periods: int = 3, n_lags: int = 8):
        self.max_lag = max_lag
        self.n_periods = n_periods
        self.n_lags = n_lags

    def fit(self, y: pd.Series):
        values = y.interpolate(limit_area="inside").dropna()
        step = step_of(y.index.freqstr)
        steps_per_day = pd.Timedelta("1D") / step
        default = int(14 * steps_per_day) if steps_per_day > 1 else 400
        self.max_lag_ = min(self.max_lag or default, len(values) // 3)
        self.y_ = y
        self.acf_ = pd.Series(acf(values, nlags=self.max_lag_, fft=True), name="acf")
        # "ywm" (Yule-Walker with the biased autocovariance) keeps |PACF| <= 1 at long lags
        self.pacf_ = pd.Series(
            pacf(values, nlags=min(self.max_lag_, len(values) // 2 - 1), method="ywm"), name="pacf"
        )
        self.confidence_bound_ = 1.96 / np.sqrt(len(values))

        # Seasonal periods: local maxima of the ACF, strongest first
        a = self.acf_.to_numpy()
        peaks = [k for k in range(2, len(a) - 1) if a[k] > a[k - 1] and a[k] >= a[k + 1] and a[k] > 0]
        peaks = sorted(peaks, key=lambda k: a[k], reverse=True)
        periods: list[int] = []
        for k in peaks:  # skip near-duplicates and multiples of a stronger period
            if all(abs(k - p) > 2 and k % p not in (0, 1, p - 1) for p in periods):
                periods.append(k)
            if len(periods) == self.n_periods:
                break
        self.suggested_periods_ = sorted(periods)

        # Lags: strongest significant partial autocorrelations, plus each period and its neighbours
        p = self.pacf_.iloc[1:]
        significant = p[p.abs() > self.confidence_bound_].abs().sort_values(ascending=False)
        lags = set(significant.index[: self.n_lags].astype(int))
        for period in self.suggested_periods_:
            lags.update({period - 1, period, period + 1})
        self.suggested_lags_ = sorted(lag for lag in lags if 1 <= lag <= self.max_lag_)
        return self

    def summary(self) -> str:
        return (
            f"Suggested seasonal periods (steps): {self.suggested_periods_}\n"
            f"Suggested lags (steps): {self.suggested_lags_}"
        )

    # ----------------------------------------------------------------- plotting
    def plot_acf_pacf(self, axes=None):
        import matplotlib.pyplot as plt

        if axes is None:
            _, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        for ax, series, title in (
            (axes[0], self.acf_, "Autocorrelation (ACF)"),
            (axes[1], self.pacf_, "Partial autocorrelation (PACF)"),
        ):
            ax.vlines(series.index, 0, series.to_numpy(), linewidth=0.8)
            ax.axhspan(-self.confidence_bound_, self.confidence_bound_, color="gray", alpha=0.2)
            ax.set_title(title)
        for period in self.suggested_periods_:
            axes[0].axvline(period, color="C3", linestyle="--", linewidth=1)
        axes[1].set_xlabel("Lag (steps)")
        return axes

    def plot_seasonal_profiles(self, axes=None):
        """Distribution of the series by hour of day, day of week and month."""
        import matplotlib.pyplot as plt

        y = self.y_.dropna()
        groups = {"Hour of day": y.index.hour, "Day of week": y.index.dayofweek, "Month": y.index.month}
        if pd.Timedelta("1D") / step_of(self.y_.index.freqstr) <= 1:
            groups.pop("Hour of day")
        if axes is None:
            _, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 3.5))
        for ax, (title, key) in zip(np.atleast_1d(axes), groups.items(), strict=True):
            frame = pd.DataFrame({"y": y.to_numpy(), "g": key})
            frame.boxplot(column="y", by="g", ax=ax, showfliers=False)
            ax.set_title(title)
            ax.set_xlabel("")
        plt.suptitle("")
        return axes

    def plot_decomposition(self, periods: Sequence[int] | None = None):
        """MSTL decomposition with the given (or suggested) seasonal periods."""
        from statsmodels.tsa.seasonal import MSTL

        periods = list(periods or self.suggested_periods_[:2])
        y = self.y_.interpolate(limit_area="inside").dropna()
        return MSTL(y, periods=periods).fit().plot()


def feature_relevance(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = "mutual_info",
    estimator=None,
    test_fraction: float = 0.2,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Rank candidate features by their relevance for the target.

    Parameters
    ----------
    method : {"mutual_info", "permutation"}
        "mutual_info" measures non-linear dependence of each feature with the
        target on its own. "permutation" fits ``estimator`` on the first part of
        the data and measures the loss of R² on the last ``test_fraction`` when
        each feature is shuffled; it accounts for redundancy between features.
    estimator : regressor, optional
        Model for the permutation method (default: HistGradientBoostingRegressor).

    Returns
    -------
    DataFrame with columns ``feature`` and ``importance``, sorted descending.
    """
    mask = X.notna().all(axis=1).to_numpy() & y.notna().to_numpy()
    Xv, yv = X[mask], y[mask]
    if method == "mutual_info":
        scores = mutual_info_regression(Xv, yv, random_state=random_state)
        out = pd.DataFrame({"feature": X.columns, "importance": scores})
    elif method == "permutation":
        from sklearn.ensemble import HistGradientBoostingRegressor

        split = int(len(Xv) * (1 - test_fraction))
        model = estimator or HistGradientBoostingRegressor(random_state=random_state)
        model.fit(Xv.iloc[:split], yv.iloc[:split])
        result = permutation_importance(
            model, Xv.iloc[split:], yv.iloc[split:], n_repeats=5, random_state=random_state
        )
        out = pd.DataFrame(
            {
                "feature": X.columns,
                "importance": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
    else:
        raise ValueError("method must be 'mutual_info' or 'permutation'")
    return out.sort_values("importance", ascending=False, ignore_index=True)
