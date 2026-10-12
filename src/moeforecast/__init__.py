"""
moeforecast: Mixture-of-Experts time series forecasting.

The library provides the building blocks to create a Mixture-of-Experts (MoE)
forecaster from a single time series:

1. ``prepare_series`` checks and regularizes the input series.
2. ``SeriesAnalyzer`` and ``feature_relevance`` help choose seasonal periods,
   lags and predictors.
3. ``FeatureBuilder`` creates causal calendar, holiday, Fourier, lag and
   rolling-window features.
4. ``ProfileClusterer`` groups seasonal profiles (e.g. days, weeks) into regimes.
5. ``RegimeClassifier`` predicts the regime of each time step from its features.
6. ``ExpertSelector`` chooses and trains one regression model (expert) per regime.
7. ``HardGating``, ``SoftGating`` and ``AdaptiveGating`` combine the experts.
8. ``MoEForecaster`` ties every step together with ``fit``, ``predict`` and
   ``forecast``.
"""

from moeforecast.analysis import SeriesAnalyzer, feature_relevance
from moeforecast.classification import RegimeClassifier, compare_classifiers
from moeforecast.clustering import ProfileClusterer
from moeforecast.data import DataReport, fill_gaps, prepare_series
from moeforecast.experts import ExpertSelector, default_candidates
from moeforecast.features import FeatureBuilder
from moeforecast.gating import AdaptiveGating, HardGating, SoftGating
from moeforecast.model_selection import temporal_train_test_split
from moeforecast.persistence import load, save
from moeforecast.pipeline import MoEForecaster

__version__ = "0.1.0.dev0"

__all__ = [
    "AdaptiveGating",
    "DataReport",
    "ExpertSelector",
    "FeatureBuilder",
    "HardGating",
    "MoEForecaster",
    "ProfileClusterer",
    "RegimeClassifier",
    "SeriesAnalyzer",
    "SoftGating",
    "compare_classifiers",
    "default_candidates",
    "feature_relevance",
    "fill_gaps",
    "load",
    "prepare_series",
    "save",
    "temporal_train_test_split",
]
