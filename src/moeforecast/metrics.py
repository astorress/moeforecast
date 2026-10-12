"""Point-forecast error metrics."""

from __future__ import annotations

import numpy as np


def _arrays(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    return y_true[mask], y_pred[mask]


def mae(y_true, y_pred) -> float:
    """Mean absolute error (pairs with a NaN are ignored)."""
    t, p = _arrays(y_true, y_pred)
    return float(np.mean(np.abs(t - p)))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error (pairs with a NaN are ignored)."""
    t, p = _arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((t - p) ** 2)))


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error as a fraction (0.05 = 5 %)."""
    t, p = _arrays(y_true, y_pred)
    if np.any(t == 0):
        raise ValueError("MAPE is undefined when y_true contains zeros.")
    return float(np.mean(np.abs((t - p) / t)))


def mase(y_true, y_pred, y_train, m: int = 1) -> float:
    """
    Mean absolute scaled error.

    The scale is the in-sample MAE of the seasonal naive forecast
    ``y[t] = y[t - m]`` on the contiguous training series ``y_train``, as in
    Hyndman & Koehler (2006). Values below 1 beat the seasonal naive forecast.
    """
    train = np.asarray(y_train, dtype=float)
    if len(train) <= m:
        raise ValueError("y_train must be longer than the seasonal period m.")
    naive_errors = np.abs(train[m:] - train[:-m])
    scale = np.nanmean(naive_errors)
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("The seasonal naive scale is zero or undefined.")
    return mae(y_true, y_pred) / scale
