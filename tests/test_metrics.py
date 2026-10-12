import numpy as np
import pytest

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
