import numpy as np
import pytest

from moeforecast import (
    ExpertSelector,
    FeatureBuilder,
    MoEForecaster,
    ProfileClusterer,
    load,
    save,
    temporal_train_test_split,
)
from moeforecast.metrics import mae, mase


@pytest.fixture(scope="module")
def fitted(demand):
    train, test = temporal_train_test_split(demand, test_size=24 * 28)
    model = MoEForecaster(
        features=FeatureBuilder(lags=(1, 2, 24, 168), rolling_windows=(24,)),
        clusterers=[
            ProfileClusterer("D", n_clusters=3, name="day"),
            ProfileClusterer("W", n_clusters=2, name="week"),
        ],
        selector=ExpertSelector(n_splits=2),
    ).fit(train)
    return model, train, test


def test_experts_are_routed_not_averaged(fitted, demand):
    """Bug fixed: expert ids built in two formats made every prediction a plain average."""
    model, train, test = fitted
    _, meta = model.predict_with_metadata(demand)
    meta = meta.loc[test.index]
    assert set(meta["expert"]) <= set(model.expert_names_)
    assert (meta["mode"] == "hard").mean() > 0.5
    assert (meta.loc[meta["mode"] == "hard", "n_experts"] == 1).all()


def test_out_of_sample_accuracy(fitted, demand):
    model, train, test = fitted
    pred = model.predict(demand).loc[test.index]
    assert pred.notna().all()
    assert mase(test, pred, train, m=168) < 1.0  # beats the weekly seasonal naive
    assert mae(test, pred) / test.mean() < 0.05


def test_training_does_not_see_test_period(fitted):
    model, train, test = fitted
    for clusterer in model.clusterers_:
        assert clusterer.labels_.index.to_timestamp().max() < test.index[0]


def test_forecast_and_persistence(fitted, demand, tmp_path):
    model, train, test = fitted
    fc = model.forecast(train, steps=48)
    assert len(fc) == 48 and fc.index[0] == test.index[0] and fc.notna().all()
    path = save(model, tmp_path / "model.joblib")
    again = load(path)
    assert np.allclose(again.predict(demand).dropna(), model.predict(demand).dropna())
