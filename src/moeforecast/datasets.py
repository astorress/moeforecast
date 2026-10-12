"""Synthetic data for examples and tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_demand_series(
    start: str = "2022-01-03", n_days: int = 365, noise: float = 0.02, random_state: int = 0
) -> pd.Series:
    """
    Hourly demand-like series with distinct weekday, Saturday and Sunday shapes,
    a yearly cycle, a slow trend and multiplicative noise.
    """
    rng = np.random.default_rng(random_state)
    idx = pd.date_range(start, periods=24 * n_days, freq="h")
    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    weekday = 0.75 + 0.25 * np.exp(-((hour - 11) ** 2) / 18) + 0.2 * np.exp(-((hour - 19) ** 2) / 6)
    saturday = 0.70 + 0.15 * np.exp(-((hour - 12) ** 2) / 20) + 0.15 * np.exp(-((hour - 19) ** 2) / 6)
    sunday = 0.62 + 0.08 * np.exp(-((hour - 13) ** 2) / 25) + 0.15 * np.exp(-((hour - 19) ** 2) / 6)
    shape = np.select([dow < 5, dow == 5], [weekday, saturday], sunday)
    t = np.arange(len(idx)) / (24 * 365.25)
    level = 1000 * (1 + 0.03 * t) * (1 + 0.05 * np.sin(2 * np.pi * t))
    values = level * shape * (1 + rng.normal(0, noise, len(idx)))
    return pd.Series(values, index=pd.DatetimeIndex(idx, freq="h"), name="demand")
