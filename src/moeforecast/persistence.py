"""Saving and loading fitted models together with the library version."""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib


def save(model, path: str | Path) -> Path:
    """Serialize a fitted model with joblib, recording the moeforecast version."""
    from moeforecast import __version__

    path = Path(path)
    joblib.dump({"moeforecast_version": __version__, "model": model}, path)
    return path


def load(path: str | Path):
    """Load a model saved with :func:`save`; warns if the library version differs."""
    from moeforecast import __version__

    payload = joblib.load(Path(path))
    saved = payload.get("moeforecast_version")
    if saved != __version__:
        warnings.warn(
            f"Model saved with moeforecast {saved}, loading with {__version__}.",
            UserWarning,
            stacklevel=2,
        )
    return payload["model"]
