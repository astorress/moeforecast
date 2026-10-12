import numpy as np
import pandas as pd
import pytest

from moeforecast import fill_gaps, prepare_series
from moeforecast.data import step_of


@pytest.mark.parametrize(
    ("freq", "duration"),
    [("D", pd.Timedelta(days=1)), ("24h", pd.Timedelta(days=1)), ("h", pd.Timedelta(hours=1))],
)
def test_step_of_fixed_frequencies(freq, duration):
    assert step_of(freq) == duration


def test_prepare_daily_series_with_day_alias():
    data = pd.Series(
        np.arange(5.0),
        index=pd.date_range("2024-01-01", periods=5, freq="D"),
    )

    result = prepare_series(data, freq="D")

    assert result.index.freqstr == "D"
    assert len(result) == 5


def test_prepare_series_from_dataframe_reports_gaps():
    idx = pd.date_range("2024-01-01", periods=48, freq="h")
    df = pd.DataFrame({"time": idx, "load": np.arange(48.0)}).drop(index=[5, 6, 20])
    y, report = prepare_series(
        df.sample(frac=1, random_state=0),
        target="load",
        time_col="time",
        return_report=True,
        warn_on_gaps=False,
    )
    assert y.index.freqstr == "h" and len(y) == 48 and y.name == "load"
    assert y.index.is_monotonic_increasing
    assert report.n_missing == 3 and report.n_gaps == 2 and report.longest_gap == 2


def test_prepare_series_warns_but_does_not_impute_gaps():
    idx = pd.date_range("2024-01-01", periods=5, freq="h").delete(2)
    with pytest.warns(UserWarning, match="No values were imputed"):
        y = prepare_series(pd.Series(np.arange(4.0), index=idx), freq="h")
    assert y.isna().sum() == 1


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
