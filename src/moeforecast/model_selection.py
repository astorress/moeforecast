"""Time-ordered data splitting."""

from __future__ import annotations

import pandas as pd


def temporal_train_test_split(
    y: pd.Series, test_size: int | float | str | pd.Timestamp
) -> tuple[pd.Series, pd.Series]:
    """
    Split a series into a training part and a later test part, without shuffling.

    Parameters
    ----------
    y : Series with a DatetimeIndex.
    test_size : int, float or timestamp
        Number of test steps (int), fraction of the series (float in (0, 1)),
        or the first test timestamp (str or Timestamp).

    Returns
    -------
    train, test : Series
        To score one-step-ahead predictions on the test period, fit on ``train``
        and call ``model.predict(y)`` on the full series, then keep
        ``predictions[test.index]``: the test features need the preceding history.
    """
    if isinstance(test_size, (str, pd.Timestamp)):
        start = pd.Timestamp(test_size)
        return y[y.index < start], y[y.index >= start]
    if isinstance(test_size, float):
        if not 0 < test_size < 1:
            raise ValueError("A float test_size must be in (0, 1).")
        n_test = int(round(len(y) * test_size))
    else:
        n_test = int(test_size)
    if not 0 < n_test < len(y):
        raise ValueError(f"test_size gives {n_test} test steps out of {len(y)}.")
    return y.iloc[:-n_test], y.iloc[-n_test:]
