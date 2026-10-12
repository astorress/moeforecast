"""Causal feature construction shared by training, prediction and forecasting."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from moeforecast.data import step_of

_EPOCH = pd.Timestamp("1970-01-01")

# Calendar attributes: (name, getter, period). Encoded as sine/cosine pairs.
_CALENDAR = {
    "minute": (lambda idx: idx.minute, 60),
    "hour": (lambda idx: idx.hour, 24),
    "dayofweek": (lambda idx: idx.dayofweek, 7),
    "dayofyear": (lambda idx: idx.dayofyear - 1, 366),
    "month": (lambda idx: idx.month - 1, 12),
}


class FeatureBuilder(TransformerMixin, BaseEstimator):
    """
    Build causal features for one-step-ahead forecasting.

    Every feature at time ``t`` uses only the calendar of ``t`` and target
    values strictly before ``t``, so the same function serves training,
    prediction and recursive forecasting.

    Parameters
    ----------
    lags : sequence of int
        Target lags, in steps.
    rolling_windows : sequence of int
        Window lengths (steps) for rolling statistics of the lagged target.
    rolling_stats : sequence of {"mean", "std", "min", "max"}
        Statistics computed over each window.
    calendar : sequence of str
        Calendar attributes encoded as sine/cosine pairs. Available: "minute",
        "hour", "dayofweek", "dayofyear", "month".
    fourier_periods : sequence of int
        Seasonal periods (steps) for Fourier terms. The phase is measured from a
        fixed origin (1970-01-01), so the terms do not depend on where the
        training data starts.
    fourier_order : int
        Number of harmonics per Fourier period.
    holidays_country : str, optional
        ISO country code for public holidays (requires the ``holidays`` package),
        e.g. "CO". Adds ``is_holiday``, ``days_to_holiday`` and ``days_from_holiday``.
    holidays_subdiv : str, optional
        Subdivision (state/province) passed to ``holidays``.
    holiday_horizon : int
        Cap, in days, for the distance-to-holiday features.
    """

    def __init__(
        self,
        lags: Sequence[int] = (1, 2, 3, 24, 48, 168),
        rolling_windows: Sequence[int] = (24, 168),
        rolling_stats: Sequence[str] = ("mean", "std"),
        calendar: Sequence[str] = ("hour", "dayofweek", "month"),
        fourier_periods: Sequence[int] = (),
        fourier_order: int = 2,
        holidays_country: str | None = None,
        holidays_subdiv: str | None = None,
        holiday_horizon: int = 30,
    ):
        self.lags = lags
        self.rolling_windows = rolling_windows
        self.rolling_stats = rolling_stats
        self.calendar = calendar
        self.fourier_periods = fourier_periods
        self.fourier_order = fourier_order
        self.holidays_country = holidays_country
        self.holidays_subdiv = holidays_subdiv
        self.holiday_horizon = holiday_horizon

    def fit(self, y: pd.Series, X=None):
        if not isinstance(y.index, pd.DatetimeIndex) or y.index.freq is None:
            raise ValueError("`y` must have a DatetimeIndex with a frequency; use prepare_series().")
        unknown = set(self.calendar) - set(_CALENDAR)
        if unknown:
            raise ValueError(f"Unknown calendar attributes: {sorted(unknown)}")
        if any(lag < 1 for lag in self.lags):
            raise ValueError("Lags must be >= 1 so features only use past values.")
        self.freq_ = y.index.freqstr
        self.step_ = step_of(self.freq_)
        self.target_name_ = y.name or "y"
        self.required_history_ = max([*self.lags, *(w + 1 for w in self.rolling_windows), 1])
        self.feature_names_out_ = list(self.transform(y.iloc[: self.required_history_ + 1]).columns)
        return self

    def transform(self, y: pd.Series) -> pd.DataFrame:
        """Return the feature matrix aligned with ``y.index``."""
        idx = y.index
        cols: dict[str, np.ndarray | pd.Series] = {}

        for name in self.calendar:
            getter, period = _CALENDAR[name]
            angle = 2 * np.pi * np.asarray(getter(idx), dtype=float) / period
            cols[f"{name}_sin"] = np.sin(angle)
            cols[f"{name}_cos"] = np.cos(angle)

        if self.fourier_periods:
            t = np.asarray((idx - _EPOCH) / self.step_, dtype=float)
            for period in self.fourier_periods:
                for k in range(1, self.fourier_order + 1):
                    angle = 2 * np.pi * k * t / period
                    cols[f"fourier_{period}_sin{k}"] = np.sin(angle)
                    cols[f"fourier_{period}_cos{k}"] = np.cos(angle)

        if self.holidays_country:
            cols.update(self._holiday_features(idx))

        for lag in self.lags:
            cols[f"lag_{lag}"] = y.shift(lag).to_numpy()

        past = y.shift(1)
        for window in self.rolling_windows:
            roll = past.rolling(window, min_periods=window)
            for stat in self.rolling_stats:
                cols[f"rolling_{stat}_{window}"] = getattr(roll, stat)().to_numpy()

        return pd.DataFrame(cols, index=idx)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)

    def _holiday_features(self, idx: pd.DatetimeIndex) -> dict[str, np.ndarray]:
        try:
            import holidays
        except ImportError as err:
            raise ImportError(
                "holidays_country requires the 'holidays' package: pip install moeforecast[holidays]"
            ) from err
        horizon = self.holiday_horizon
        days = idx.normalize()
        years = range(days.min().year - 1, days.max().year + 2)
        calendar = holidays.country_holidays(self.holidays_country, subdiv=self.holidays_subdiv, years=years)
        holiday_days = pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in calendar))
        day_values = days.to_numpy()
        hol_values = holiday_days.to_numpy()
        pos = np.searchsorted(hol_values, day_values, side="left")
        nxt = hol_values[np.minimum(pos, len(hol_values) - 1)]
        is_holiday = (pos < len(hol_values)) & (nxt == day_values)
        to_next = (nxt - day_values) / np.timedelta64(1, "D")
        prev_pos = np.searchsorted(hol_values, day_values, side="right") - 1
        prev = hol_values[np.maximum(prev_pos, 0)]
        from_prev = (day_values - prev) / np.timedelta64(1, "D")
        return {
            "is_holiday": is_holiday.astype(float),
            "days_to_holiday": np.clip(to_next, 0, horizon),
            "days_from_holiday": np.clip(from_prev, 0, horizon),
        }
