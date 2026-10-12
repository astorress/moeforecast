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
    """Each hour is routed to its experts instead of averaging all of them."""
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


def test_inspect_exposes_fitted_models_and_diagnostics(fitted):
    model, _, _ = fitted
    report = model.inspect()

    assert report["features"] == model.features_.feature_names_out_
    assert report["regime_labels"].equals(model.regime_labels_)
    assert set(report["clusters"]) == set(model.dimensions_)
    for clusterer in model.clusterers_:
        cluster = report["clusters"][clusterer.name_]
        assert cluster["estimator"] is clusterer.kmeans_
        assert cluster["assignments"].equals(clusterer.labels_)
        assert cluster["summary"].equals(clusterer.summary())

    for dimension, classifier in zip(model.dimensions_, model.classifiers_, strict=True):
        details = report["classifiers"][dimension]
        assert details["estimator"] is classifier.estimator_
        assert details["class_counts"].sum() == details["n_samples_fit"]

    assert report["expert_selection"]["expert_name"].notna().all()
    assert set(report["experts"]) == set(model.expert_names_)
    assert all(
        report["experts"][name] is model.experts_[key]
        for name, key in zip(model.expert_names_, model.expert_keys_, strict=True)
    )
    assert set(report["candidate_experts"]) == set(model.selector_.candidate_estimators_)


def test_inspect_requires_fitted_model():
    with pytest.raises(RuntimeError, match="must be fitted"):
        MoEForecaster().inspect()


def test_pipeline_requires_explicitly_reviewed_clusterers(demand):
    with pytest.raises(ValueError, match="Pass clusterers with explicit n_clusters"):
        MoEForecaster().fit(demand)


def test_forecast_and_persistence(fitted, demand, tmp_path):
    model, train, test = fitted
    fc = model.forecast(train, steps=48)
    assert len(fc) == 48 and fc.index[0] == test.index[0] and fc.notna().all()
    path = save(model, tmp_path / "model.joblib")
    again = load(path)
    assert np.allclose(again.predict(demand).dropna(), model.predict(demand).dropna())


def test_gating_hpo_uses_later_validation_and_returns_ranked_scores(fitted, tmp_path):
    model, _, validation = fitted
    scores = model.tune_gating(
        validation,
        search="grid",
        thresholds=(0.5, 0.8),
        min_weights=(0.0, 0.1),
    )
    assert len(scores) == 12
    assert scores.iloc[0]["mae_validation"] <= scores.iloc[-1]["mae_validation"]
    assert model.gating_.threshold == scores.iloc[0]["threshold"]


def test_fitted_model_diagram_can_be_saved(fitted, tmp_path):
    model, _, _ = fitted
    destination = tmp_path / "moe-architecture.png"
    fig = model.plot_model(destination)
    assert destination.is_file() and destination.stat().st_size > 0
    assert fig._suptitle.get_text() == "Fitted Mixture-of-Experts Architecture"
