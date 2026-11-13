import numpy as np

from moeforecast import FeatureBuilder


def test_features_are_causal(demand):
    """Changing the future must not change features of the past or present."""
    fb = FeatureBuilder(fourier_periods=(24, 168)).fit(demand)
    base = fb.transform(demand)
    t = 2000
    perturbed = demand.copy()
    perturbed.iloc[t:] += 1e6
    after = fb.transform(perturbed)
    assert np.allclose(base.iloc[: t + 1].fillna(0), after.iloc[: t + 1].fillna(0))


def test_fourier_terms_do_not_depend_on_series_start(demand):
    """Fourier phase is anchored to a fixed origin, not to the first row of the training data."""
    fb = FeatureBuilder(fourier_periods=(168,), lags=(1,), rolling_windows=()).fit(demand)
    full = fb.transform(demand)
    tail = fb.transform(demand.iloc[500:])
    cols = [c for c in full.columns if c.startswith("fourier")]
    assert np.allclose(full[cols].iloc[500:].to_numpy(), tail[cols].to_numpy())


def test_holiday_features(demand):
    fb = FeatureBuilder(holidays_country="CO").fit(demand)
    X = fb.transform(demand)
    new_year = X.loc["2023-01-01 10:00"]  # Sunday, also a holiday in Colombia
    assert new_year["is_holiday"] == 1.0 and new_year["days_to_holiday"] == 0
    assert X["days_to_holiday"].max() <= 30
