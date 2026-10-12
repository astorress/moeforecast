# moeforecast

**Mixture-of-Experts time series forecasting with a scikit-learn style API.**

`moeforecast` builds a forecaster that learns the *regimes* of a series and trains one specialised model (an *expert*) per regime. A regime is, for example, a type of day (working day, weekend, holiday) or a type of week. A gating network then combines the experts at prediction time. The method comes from hourly electricity demand forecasting, but it applies to any regularly sampled series with recurring seasonal profiles.

> Status: pre-alpha (0.1.0.dev0). The API may change.

```python
import pandas as pd
from moeforecast import prepare_series, temporal_train_test_split

df = pd.read_csv("load.csv")  # your data
y = prepare_series(df, target="load", time_col="timestamp")  # regular, sorted series

train, test = temporal_train_test_split(y, test_size="2025-01-01")
# Continue with the analysis and cluster-selection workflow below.
```

## What the library does (and does not do)

`moeforecast` gives you the tools to **build** the model and to choose its configuration, like scikit-learn does. Benchmarking against other models, statistical comparison tests and robustness studies are left to the user.

| Step | Tool | What it gives you |
|---|---|---|
| 1. Prepare the data | `prepare_series`, `fill_gaps` | Sorted series on a regular grid; warns about gaps and never imputes silently |
| 2. Analyse the series | `SeriesAnalyzer` | Suggested seasonal periods and lags (ACF/PACF); ACF/PACF, seasonal-profile and MSTL plots |
| 3. Build features | `FeatureBuilder` | Causal calendar, Fourier, holiday (any country), lag and rolling features. The same code serves training, prediction and forecasting |
| 4. Rank features | `feature_relevance` | Mutual information or permutation importance |
| 5. Find regimes | `ProfileClusterer` | Daily, weekly or monthly profile diagnostics, distance choice, silhouette plots and explicit selection of k |
| 6. Predict regimes | `RegimeClassifier`, `compare_classifiers` | Regime probabilities for the gating; time-ordered comparison of classifiers |
| 7. Train experts | `ExpertSelector` | Best candidate model per regime (forward-chaining CV); ten default regressors, user additions/removals and MAE/RMSE/MAPE selection |
| 8. Combine experts | `HardGating`, `SoftGating`, `AdaptiveGating` | Expert weights per time step |
| 9. All together | `MoEForecaster` | `fit`, `predict`, `predict_with_metadata`, `expert_weights`, `forecast` |
| 10. Persist | `save`, `load` | Serialization with version check |
| Metrics | `moeforecast.metrics` | MAE, RMSE, MAPE, MASE |

## Installation

```bash
pip install "moeforecast[all]"          # with LightGBM and public holidays
pip install "moeforecast[dtw]"          # optional DTW profile clustering
pip install moeforecast                 # core only: numpy, pandas, scikit-learn, statsmodels, matplotlib
```

Not on PyPI yet. Install from source:

```bash
git clone https://github.com/astorress/moeforecast.git
cd moeforecast
pip install -e ".[dev]"
pytest
```

## Step by step

```python
from moeforecast import (
    SeriesAnalyzer,
    FeatureBuilder,
    feature_relevance,
    ProfileClusterer,
    compare_classifiers,
    ExpertSelector,
    AdaptiveGating,
    MoEForecaster,
)

# 1. Which periods and lags matter?
analysis = SeriesAnalyzer().fit(train)
print(analysis.summary())  # e.g. periods [24, 168], lags [1, 2, 23, 24, 25, 167, 168, 169]
analysis.plot_acf_pacf()

# 2. Features and their relevance
features = FeatureBuilder(lags=analysis.suggested_lags_, fourier_periods=(24, 168), holidays_country="CO")
X = features.fit(train).transform(train)
feature_relevance(X, train, method="permutation").head(15)

# 3. Inspect cluster-validation metrics before choosing the cluster count.
days = ProfileClusterer("D", name="day", k_range=range(2, 11))
days.evaluate(train)
days.plot_scores()
days.plot_silhouette()  # inspect all candidate k values before deciding
days.set_params(n_clusters=3).fit(train)  # choose k after reviewing the scores
days.plot_profiles()

weeks = ProfileClusterer("W", name="week", k_range=range(2, 11))
weeks.evaluate(train)
weeks.plot_scores()
weeks.set_params(n_clusters=2).fit(train)

# 4. Can the regimes be predicted from the features?
compare_classifiers(X, days.labels_for(train.index))

# 5. The full model with the chosen configuration
model = MoEForecaster(
    features=features,
    clusterers=[days, weeks],
    selector=ExpertSelector(metric="mae"),
    gating=AdaptiveGating(threshold=0.75),
).fit(train)

model.selector_.results_  # every candidate's CV score per expert
pred, meta = model.predict_with_metadata(y)
meta["mode"].value_counts()  # hard vs. soft gating decisions
```

## Inspect every fitted component

The fitted estimators are not hidden behind the forecaster. `inspect()` returns
the actual scikit-learn objects together with copied diagnostic tables:

```python
report = model.inspect()

# Cluster assignments, candidate-k scores, source profiles and cluster means
report["clusters"]["day"]["summary"]
report["clusters"]["day"]["scores"]
report["clusters"]["day"]["assignments"]
report["clusters"]["day"]["mean_profiles"]

# Fitted classifier and its input features / class counts
classifier = report["classifiers"]["day"]
print(classifier["estimator"])
classifier["features"], classifier["class_counts"]

# Every candidate's time-series CV results and each final fitted expert
report["expert_selection"]
for expert_name, estimator in report["experts"].items():
    print(expert_name, estimator)

# Candidate model templates, fitted feature builder, and gating estimator
report["candidate_experts"]
report["configuration"]["features"].get_params()
report["gating"].get_params()
```

`inspect()` also exposes the full `ProfileClusterer` objects and fitted k-means
estimators. `RegimeClassifier.estimator_`, `ExpertSelector.experts_`,
`ExpertSelector.candidate_estimators_`, `ProfileClusterer.labels_`,
`ProfileClusterer.scores_`, and `ProfileClusterer.mean_profiles_` remain
available directly for users who prefer to inspect individual components.

See the [quickstart notebook](examples/quickstart.ipynb) for a complete,
synthetic-data walkthrough of the public modeling API, including diagnostics,
regime classifiers, all gating strategies, model inspection, forecasting and
persistence.

## Design

These principles prevent the errors that are easy to make when a Mixture of Experts is built by hand:

- **No look-ahead.** Every feature at time *t* uses only values before *t*. `fit` only sees the training series. The classifier comparison and the expert selection use forward-chaining (time-ordered) cross-validation.
- **One feature function.** `FeatureBuilder.transform` is used for training, prediction and recursive forecasting, so there is no train/serve skew. Fourier terms are anchored to a fixed origin, not to where the data starts.
- **Typed expert identifiers.** An expert is a tuple of regime labels, e.g. `(0, 1)`, used everywhere. `"day0-week1"` is only a display name.
- **Deterministic choices.** Cluster labels are ordered by level and ties are broken by candidate order. Selections come from the stated metric.

[docs/design.md](docs/design.md) explains each decision.

## Release process

Development builds are published to TestPyPI from the `release` branch. Stable
versions are published to PyPI only after merging to `master` or `main`. See
[the release guide](docs/release-process.md) for the full workflow and the
one-time Trusted Publisher setup.

## Roadmap

- Deep-learning experts (LSTM/GRU) as an optional extra (`moeforecast[dl]`).
- Hyperparameter optimization of the experts (`HalvingRandomSearchCV`).
- Tuning of the gating parameters on a validation period.
- DTW and k-Shape clustering (optional `tslearn` extra).
- HTML report of the analysis and the fitted model.

## Citation

If you use `moeforecast` in research, please cite the accompanying article (reference to be added).

## License

[MIT](LICENSE)
