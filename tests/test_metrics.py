import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from moeforecast.experts import ExpertSelector, default_candidates
from moeforecast.metrics import mae, mape, mase, rmse


def test_basic_metrics_ignore_nan():
    y, p = np.array([1.0, 2.0, np.nan, 4.0]), np.array([1.0, 3.0, 5.0, 2.0])
    assert mae(y, p) == pytest.approx(1.0)
    assert rmse(y, p) == pytest.approx(np.sqrt(5 / 3))
    assert mape(y, p) == pytest.approx((0 + 0.5 + 0.5) / 3)


def test_mase_uses_training_seasonal_naive_scale():
    train = np.tile([1.0, 3.0], 10)  # seasonal naive with m=2 is perfect -> scale 0
    with pytest.raises(ValueError):
        mase([1.0], [1.5], train, m=2)
    assert mase([2.0], [3.0], np.arange(10.0), m=1) == pytest.approx(1.0)


def test_default_expert_candidates_and_mape_selection():
    defaults = default_candidates(exclude=("linear",))
    assert len(defaults) >= 9
    index = pd.date_range("2024-01-01", periods=36, freq="h")
    X = pd.DataFrame({"x": np.arange(36.0)}, index=index)
    y = pd.Series(np.arange(1.0, 37.0), index=index)
    keys = pd.Series([(0,)] * len(y), index=index)
    selector = ExpertSelector(
        candidates={"mean": DummyRegressor()}, metric="mape", n_splits=2, min_samples=2
    ).fit(X, y, keys)
    assert "mape_mean" in selector.results_
