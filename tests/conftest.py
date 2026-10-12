import pytest

from moeforecast.datasets import make_demand_series


@pytest.fixture(scope="session")
def demand():
    """One year of hourly synthetic demand with weekday/Saturday/Sunday shapes."""
    return make_demand_series(n_days=364)
