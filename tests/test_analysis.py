import matplotlib

matplotlib.use("Agg")

from moeforecast import (  # noqa: E402
    FeatureBuilder,
    ProfileClusterer,
    SeriesAnalyzer,
    compare_classifiers,
    feature_relevance,
)


def test_analyzer_finds_daily_and_weekly_seasonality(demand):
    an = SeriesAnalyzer().fit(demand)
    assert 24 in an.suggested_periods_ and 168 in an.suggested_periods_
    assert {1, 24, 168}.issubset(an.suggested_lags_)


def test_feature_relevance_rankings(demand):
    X = FeatureBuilder().fit(demand).transform(demand)
    mi = feature_relevance(X, demand, method="mutual_info")
    assert mi.iloc[0]["feature"].startswith(("lag_", "rolling_"))
    perm = feature_relevance(X.iloc[-3000:], demand.iloc[-3000:], method="permutation")
    assert set(perm.columns) == {"feature", "importance", "importance_std"}


def test_compare_classifiers_uses_time_ordered_folds(demand):
    X = FeatureBuilder(lags=(1, 24), rolling_windows=()).fit(demand).transform(demand)
    labels = ProfileClusterer("D", n_clusters=3).fit(demand).labels_for(demand.index)
    table = compare_classifiers(X, labels, n_splits=3)
    assert table["f1_macro_mean"].is_monotonic_decreasing
    assert table["accuracy_mean"].max() > 0.8


def test_plots_render(demand):
    an = SeriesAnalyzer().fit(demand)
    an.plot_acf_pacf()
    an.plot_seasonal_profiles()
    an.plot_decomposition(periods=[24, 168])
    pc = ProfileClusterer("D", n_clusters=3).fit(demand)
    pc.plot_silhouette()
    pc.plot_profiles()
    pc.plot_scores()


def test_acf_pacf_plot_is_limited_to_30_lags_by_default(demand):
    an = SeriesAnalyzer().fit(demand)
    axes = an.plot_acf_pacf()
    assert all(ax.get_xlim()[1] == 30 for ax in axes)
    assert all(not ax.spines["top"].get_visible() for ax in axes)
    assert all(ax.title.get_fontfamily() == ["serif"] for ax in axes)


def test_decomposition_has_one_labeled_panel_per_component(demand):
    analyzer = SeriesAnalyzer().fit(demand)
    fig = analyzer.plot_decomposition(periods=[24, 168])
    assert [ax.get_title() for ax in fig.axes] == [
        "Observed",
        "Trend",
        "Seasonal - 24 steps",
        "Seasonal - 168 steps",
        "Residual",
    ]
    assert all(ax.lines and ax.xaxis.get_shared_x_axes().joined(ax, fig.axes[0]) for ax in fig.axes)
    assert all(not ax.spines["top"].get_visible() for ax in fig.axes)
