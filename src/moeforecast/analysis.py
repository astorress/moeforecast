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

from moeforecast._plotting import PALETTE, style_axis
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
    def plot_acf_pacf(self, axes=None, max_lag: int = 30):
        """Plot at most ``max_lag`` ACF/PACF steps (30 by default)."""
        import matplotlib.pyplot as plt

        if max_lag < 1:
            raise ValueError("max_lag must be a positive integer.")
        if axes is None:
            _, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, layout="constrained")
        for ax, series, title in (
            (axes[0], self.acf_, "Autocorrelation (ACF)"),
            (axes[1], self.pacf_, "Partial autocorrelation (PACF)"),
        ):
            visible = series.iloc[: max_lag + 1]
            ax.axhspan(
                -self.confidence_bound_,
                self.confidence_bound_,
                color=PALETTE[2],
                alpha=0.22,
                label="95% bound",
            )
            ax.vlines(visible.index, 0, visible.to_numpy(), color=PALETTE[0], linewidth=1.2)
            ax.scatter(visible.index, visible.to_numpy(), color=PALETTE[0], s=15, zorder=3)
            ax.axhline(0, color="#404040", linewidth=0.8)
            ax.set_xlim(0, max_lag)
            ax.set_ylabel("Correlation")
            ax.set_title(title, loc="left", fontweight="bold")
            style_axis(ax)
        for period in self.suggested_periods_:
            if period <= max_lag:
                axes[0].axvline(period, color="#b85c5c", linestyle="--", linewidth=1.2)
        axes[1].set_xlabel("Lag (steps)")
        axes[0].set_xlabel("Lag (steps)")
        axes[0].legend(frameon=False, loc="upper right")
        return axes

    def plot_seasonal_profiles(self, axes=None):
        """Distribution of the series by hour of day, day of week and month."""
        import matplotlib.pyplot as plt

        y = self.y_.dropna()
        groups = {
            "Hour of day": (y.index.hour, [f"{h:02d}" for h in range(24)]),
            "Day of week": (y.index.dayofweek, ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]),
            "Month": (
                y.index.month - 1,
                ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            ),
        }
        if pd.Timedelta("1D") / step_of(self.y_.index.freqstr) <= 1:
            groups.pop("Hour of day")
        if axes is None:
            _, axes = plt.subplots(1, len(groups), figsize=(5.5 * len(groups), 4), layout="constrained")
        for ax, (title, (key, tick_labels)) in zip(np.atleast_1d(axes), groups.items(), strict=True):
            frame = pd.DataFrame({"y": y.to_numpy(), "g": key}, index=y.index)
            present = sorted(frame["g"].unique())
            box = ax.boxplot(
                [frame.loc[frame["g"] == group, "y"].to_numpy() for group in present],
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "#17324d", "linewidth": 1.8},
                whiskerprops={"color": "#566573"},
                capprops={"color": "#566573"},
            )
            for i, patch in enumerate(box["boxes"]):
                patch.set_facecolor(PALETTE[i % len(PALETTE)])
                patch.set_alpha(0.8)
            medians = frame.groupby("g", sort=True)["y"].median().reindex(present)
            ax.plot(
                range(1, len(present) + 1),
                medians.to_numpy(),
                color="#c44536",
                marker="o",
                linewidth=2,
                markersize=4,
                label="Group median",
                zorder=4,
            )
            ax.set_xticks(range(1, len(present) + 1), [tick_labels[i] for i in present])
            ax.set_title(title, loc="left", fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel(str(y.name or "Value"))
            style_axis(ax, grid_axis="y")
            ax.legend(frameon=False)
        return axes

    def plot_decomposition(self, periods: Sequence[int] | None = None):
        """MSTL decomposition with the given (or suggested) seasonal periods."""
        from statsmodels.tsa.seasonal import MSTL

        periods = list(periods or self.suggested_periods_[:2])
        y = self.y_.interpolate(limit_area="inside").dropna()
        if not periods:
            raise ValueError("At least one seasonal period is required for decomposition.")
        fig = MSTL(y, periods=periods).fit().plot()
        axes = np.atleast_1d(fig.axes)
        labels = ["Observed", "Trend", *[f"Seasonal ({period})" for period in periods], "Residual"]
        line_colors = ["#4682b4", "#ff8c00", "#2e8b57", "#2e8b57", "#cd5c5c"]
        for index, ax in enumerate(axes):
            ax.set_title(labels[index] if index < len(labels) else "Component", loc="left", fontsize=9)
            ax.set_ylabel("")
            for line in ax.lines:
                line.set_color(line_colors[min(index, len(line_colors) - 1)])
                line.set_linewidth(0.8)
            style_axis(ax, grid_axis="y")
            if index < len(axes) - 1:
                ax.tick_params(axis="x", labelbottom=False)
        axes[-1].set_xlabel("Date")
        fig.suptitle(
            f"{y.name or 'Series'} | MSTL decomposition",
            x=0.01,
            ha="left",
            fontweight="bold",
            fontfamily="serif",
        )
        fig.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.08, hspace=0.3)
        return fig


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
