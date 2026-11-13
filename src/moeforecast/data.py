"""Input validation: turn user data into a regular, sorted time series."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import pandas as pd
from pandas.tseries.frequencies import to_offset


@dataclass(frozen=True)
class DataReport:
    """Summary of the checks performed by :func:`prepare_series`."""

    start: pd.Timestamp
    end: pd.Timestamp
    freq: str
    n_steps: int
    n_missing: int
    n_gaps: int
    longest_gap: int

    def __str__(self) -> str:
        return (
            f"{self.n_steps:,} steps at freq '{self.freq}' from {self.start} to {self.end}; "
            f"{self.n_missing:,} missing values in {self.n_gaps:,} gaps "
            f"(longest: {self.longest_gap:,} steps)"
        )


def step_of(freq: str) -> pd.Timedelta:
    """Duration of one step of a fixed frequency such as 'h', '15min' or 'D'."""
    offset = to_offset(freq)
    try:
        # Convert through nanoseconds: pandas 3 no longer accepts a Day offset
        # directly in Timedelta, although it is a fixed-duration frequency.
        return pd.Timedelta(offset.nanos, unit="ns")
    except (AttributeError, ValueError) as err:
        raise ValueError(
            f"Frequency '{freq}' has no fixed duration. moeforecast supports fixed "
            "frequencies such as 'min', 'h' or 'D'."
        ) from err


def _gap_lengths(missing: pd.Series) -> pd.Series:
    """Length of each run of consecutive missing values."""
    run_id = (missing != missing.shift()).cumsum()
    return missing.groupby(run_id).sum()[lambda s: s > 0]


def prepare_series(
    data: pd.Series | pd.DataFrame,
    target: str | None = None,
    time_col: str | None = None,
    freq: str | None = None,
    return_report: bool = False,
    warn_on_gaps: bool = True,
) -> pd.Series | tuple[pd.Series, DataReport]:
    """
    Check and regularize a time series.

    Parameters
    ----------
    data : Series or DataFrame
        A Series with a DatetimeIndex, or a DataFrame that contains the target
        column and either a DatetimeIndex or a time column.
    target : str, optional
        Target column, required when ``data`` is a DataFrame.
    time_col : str, optional
        Time column of a DataFrame. If omitted, the index is used.
    freq : str, optional
        Sampling frequency (e.g. ``"h"``). Inferred from the data when omitted.
    return_report : bool
        Also return a :class:`DataReport`.
    warn_on_gaps : bool
        Emit a ``UserWarning`` when the regularized series contains missing
        values. Missing values are never imputed automatically.

    Returns
    -------
    y : Series
        Float series on a regular grid (missing steps are NaN), sorted, with
        ``y.index.freq`` set and ``y.name`` equal to the target name.
    report : DataReport
        Only when ``return_report=True``.

    Raises
    ------
    ValueError
        On duplicated timestamps, a non-datetime index, timezone-aware
        timestamps or a frequency that cannot be inferred.
    """
    if isinstance(data, pd.DataFrame):
        if target is None:
            raise ValueError("`target` is required when `data` is a DataFrame.")
        frame = data.set_index(time_col) if time_col is not None else data
        y = frame[target]
    elif isinstance(data, pd.Series):
        y = data
    else:
        raise TypeError("`data` must be a pandas Series or DataFrame.")

    if not isinstance(y.index, pd.DatetimeIndex):
        try:
            y = y.copy()
            y.index = pd.to_datetime(y.index)
        except (TypeError, ValueError) as err:
            raise ValueError("The time index cannot be converted to datetimes.") from err
    if y.index.tz is not None:
        raise ValueError(
            "Timezone-aware timestamps are not supported: convert to local time and drop "
            "the timezone (y.index.tz_localize(None)) so daily and weekly profiles align."
        )
    if y.index.has_duplicates:
        dups = y.index[y.index.duplicated()].unique()
        raise ValueError(f"{len(dups)} duplicated timestamps, e.g. {list(dups[:3])}.")

    y = y.sort_index().astype(float)
    if freq is None:
        # Most common spacing between timestamps; robust to gaps, unlike pd.infer_freq
        deltas = y.index.to_series().diff().dropna()
        if deltas.empty:
            raise ValueError("Cannot infer the frequency of a single timestamp; pass `freq`.")
        # A calendar frequency (e.g. month starts) must not be mistaken for its most common
        # spacing in hours: pd.infer_freq reports it, and step_of() below rejects it clearly.
        try:
            exact = pd.infer_freq(y.index) if len(y.index) >= 3 else None
        except (TypeError, ValueError):
            exact = None
        freq = exact or to_offset(deltas.mode().iloc[0]).freqstr
    step_of(freq)  # validate
    y = y.asfreq(freq)
    y.name = target or y.name or "y"

    missing = y.isna()
    gaps = _gap_lengths(missing)
    report = DataReport(
        start=y.index[0],
        end=y.index[-1],
        freq=freq,
        n_steps=len(y),
        n_missing=int(missing.sum()),
        n_gaps=len(gaps),
        longest_gap=int(gaps.max()) if len(gaps) else 0,
    )
    if warn_on_gaps and report.n_missing:
        warnings.warn(
            f"prepare_series found {report.n_missing} missing value(s) across "
            f"{report.n_gaps} gap(s) (longest: {report.longest_gap} steps). "
            "Review the returned series and decide whether to impute them; "
            "see moeforecast.fill_gaps. No values were imputed.",
            UserWarning,
            stacklevel=2,
        )
    if not return_report:
        return y
    return y, report


def fill_gaps(y: pd.Series, max_gap: int | None = None, method: str = "time") -> pd.Series:
    """
    Interpolate missing values, optionally only in gaps of at most ``max_gap`` steps.

    Longer gaps stay NaN; the rows they affect are dropped when fitting.
    """
    filled = y.interpolate(method=method, limit_area="inside")
    if max_gap is None:
        return filled
    missing = y.isna()
    run_id = (missing != missing.shift()).cumsum()
    run_len = missing.groupby(run_id).transform("sum")
    too_long = missing & (run_len > max_gap)
    return filled.mask(too_long)
