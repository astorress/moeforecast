import numpy as np
import pandas as pd
import pytest

from moeforecast import ProfileClusterer


def test_daily_clusters_separate_weekdays_from_weekends(demand):
    pc = ProfileClusterer("D", n_clusters=3).fit(demand)
    labels = pc.labels_
    dow = labels.index.to_timestamp().dayofweek
    weekday_labels = set(labels[dow < 5])
    assert len(weekday_labels) == 1
    assert labels[dow == 6].iloc[0] not in weekday_labels


def test_labels_are_ordered_by_level(demand):
    pc = ProfileClusterer("D", n_clusters=3).fit(demand)
    means = pc.mean_profiles_.mean(axis=1).to_numpy()
    assert (means[:-1] < means[1:]).all()
    assert pc.summary()["n_profiles"].sum() == len(pc.profiles_)
    assert pc.summary()["mean_level"].is_monotonic_increasing


def test_k_selection_and_weekly_profiles(demand):
    preview = ProfileClusterer("W", k_range=range(2, 11))
    scores = preview.evaluate(demand)
    assert list(scores.index) == list(range(2, 11))
    pc = ProfileClusterer("W", n_clusters=3, k_range=range(2, 11)).fit(demand)
    assert pc.profiles_.shape[1] == 168
    assert pc.n_clusters_ == 3 and set(pc.scores_.index) == set(range(2, 11))
    lab = pc.labels_for(demand.index)
    assert lab.notna().mean() > 0.95


def test_manual_cluster_choice_is_required_by_default(demand):
    clusterer = ProfileClusterer("D")
    scores = clusterer.evaluate(demand)
    assert scores.index.max() == 10
    silhouette_axes = clusterer.plot_silhouette()
    assert len(silhouette_axes) == len(scores)
    with pytest.raises(ValueError, match="Choose the cluster count explicitly"):
        clusterer.fit(demand)
    clusterer.set_params(n_clusters=3).fit(demand)
    assert clusterer.n_clusters_ == 3


def test_monthly_profiles_are_supported():
    index = pd.date_range("2020-01-01", "2023-12-31 23:00", freq="h")
    y = pd.Series(np.sin(np.arange(len(index)) * 2 * np.pi / 24), index=index)
    clusterer = ProfileClusterer("M", n_clusters=2).fit(y)
    assert clusterer.profiles_.shape == (48, 31)
    assert clusterer.labels_.notna().all()


def test_monthly_frequency_can_cluster_monthly_values():
    index = pd.date_range("2020-01-31", periods=48, freq="ME")
    y = pd.Series(np.linspace(1, 10, len(index)), index=index)
    clusterer = ProfileClusterer("M", n_clusters=2).fit(y)
    assert clusterer.profiles_.shape == (48, 1)
