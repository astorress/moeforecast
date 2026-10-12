import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from moeforecast import ExpertSelector, RegimeClassifier, default_search_spaces


def test_default_search_spaces_cover_builtin_models():
    spaces = default_search_spaces()
    assert "ridge__alpha" in spaces["ridge"]
    assert "max_iter" in spaces["hist_gbm"]
    assert "logisticregression__C" in spaces["logistic"]


def test_expert_selector_searches_parameters_for_each_expert():
    index = pd.date_range("2024-01-01", periods=48, freq="h")
    X = pd.DataFrame({"x": np.arange(48, dtype=float)}, index=index)
    y = pd.Series(2 * X["x"] + np.sin(X["x"]), index=index)
    keys = pd.Series([(0,)] * 48, index=index, dtype=object)
    selector = ExpertSelector(
        candidates={"ridge": make_pipeline(StandardScaler(), Ridge())},
        search="grid",
        search_spaces={"ridge": {"ridge__alpha": [0.1, 1.0]}},
        n_splits=2,
        min_samples=10,
    ).fit(X, y, keys)
    assert selector.selected_[(0,)] == "ridge"
    best_params = selector.results_.loc[0, "best_params"]
    assert best_params["ridge__alpha"] in {0.1, 1.0}
    assert selector.experts_[(0,)].named_steps["ridge"].alpha == best_params["ridge__alpha"]
    assert len(selector.search_results_[(0,)]["ridge"]) == 2


def test_regime_classifier_can_random_search_parameters():
    index = pd.date_range("2024-01-01", periods=80, freq="h")
    X = pd.DataFrame({"x": np.sin(np.arange(80) / 5)}, index=index)
    labels = pd.Series((np.arange(80) // 10) % 2, index=index)
    classifier = RegimeClassifier(
        estimator=make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)),
        search="random",
        search_space={"logisticregression__C": [0.1, 1.0]},
        n_iter=2,
        n_splits=2,
    ).fit(X, labels)
    assert classifier.search_.best_params_["logisticregression__C"] in {0.1, 1.0}
    assert classifier.search_.cv_results_["params"]
    assert classifier.predict_proba(X).shape == (len(X), 2)
