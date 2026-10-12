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
    pc = ProfileClusterer("W", k_range=range(2, 5)).fit(demand)
    assert pc.profiles_.shape[1] == 168
    assert pc.n_clusters_ in (2, 3, 4) and set(pc.scores_.index) == {2, 3, 4}
    lab = pc.labels_for(demand.index)
    assert lab.notna().mean() > 0.95
