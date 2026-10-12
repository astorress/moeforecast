import numpy as np
import pandas as pd
import pytest

from moeforecast import fill_gaps, prepare_series


def test_prepare_series_from_dataframe_reports_gaps():
    idx = pd.date_range("2024-01-01", periods=48, freq="h")
    df = pd.DataFrame({"time": idx, "load": np.arange(48.0)}).drop(index=[5, 6, 20])
    y, report = prepare_series(
        df.sample(frac=1, random_state=0), target="load", time_col="time", return_report=True
    )
    assert y.index.freqstr == "h" and len(y) == 48 and y.name == "load"
    assert y.index.is_monotonic_increasing
    assert report.n_missing == 3 and report.n_gaps == 2 and report.longest_gap == 2


def test_prepare_series_rejects_duplicates():
    idx = pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 00:00", "2024-01-01 01:00"])
    with pytest.raises(ValueError, match="duplicated"):
        prepare_series(pd.Series([1.0, 2.0, 3.0], index=idx), freq="h")


def test_fill_gaps_respects_max_gap():
    y = pd.Series(
        [1.0, np.nan, 3.0, np.nan, np.nan, np.nan, 7.0],
        index=pd.date_range("2024-01-01", periods=7, freq="h"),
    )
    filled = fill_gaps(y, max_gap=1)
    assert filled.iloc[1] == 2.0
    assert filled.iloc[3:6].isna().all()
