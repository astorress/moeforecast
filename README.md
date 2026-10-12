# moeforecast

**Mixture-of-Experts time series forecasting with a scikit-learn style API.**

`moeforecast` builds a forecaster that learns the *regimes* of a series and trains one specialised model (an *expert*) per regime. A regime is, for example, a type of day (working day, weekend, holiday) or a type of week. A gating network then combines the experts at prediction time. The method comes from hourly electricity demand forecasting, but it applies to any regularly sampled series with recurring seasonal profiles.

> Status: pre-alpha (0.1.0.dev0). The API may change.

```python
import pandas as pd
from moeforecast import MoEForecaster, prepare_series, temporal_train_test_split
from moeforecast.metrics import mae, mase

df = pd.read_csv("load.csv")  # your data
y = prepare_series(df, target="load", time_col="timestamp")  # regular, sorted series

train, test = temporal_train_test_split(y, test_size="2025-01-01")
model = MoEForecaster().fit(train)

pred = model.predict(y)[test.index]  # one-step-ahead on the test period
print(mae(test, pred), mase(test, pred, train, m=168))

next_week = model.forecast(y, steps=168)  # recursive multi-step forecast
```

## What the library does (and does not do)

`moeforecast` gives you the tools to **build** the model and to choose its configuration, like scikit-learn does. Benchmarking against other models, statistical comparison tests and robustness studies are left to the user.

| Step | Tool | What it gives you |
|---|---|---|
| 1. Prepare the data | `prepare_series`, `fill_gaps` | Sorted series on a regular grid; report of missing values and gaps |
| 2. Analyse the series | `SeriesAnalyzer` | Suggested seasonal periods and lags (ACF/PACF); ACF/PACF, seasonal-profile and MSTL plots |
| 3. Build features | `FeatureBuilder` | Causal calendar, Fourier, holiday (any country), lag and rolling features. The same code serves training, prediction and forecasting |
| 4. Rank features | `feature_relevance` | Mutual information or permutation importance |
| 5. Find regimes | `ProfileClusterer` | Clusters of daily/weekly/... profiles, choice of k, profile and metric plots |
| 6. Predict regimes | `RegimeClassifier`, `compare_classifiers` | Regime probabilities for the gating; time-ordered comparison of classifiers |
| 7. Train experts | `ExpertSelector` | Best candidate model per regime (forward-chaining CV), with complexity limits for small regimes |
| 8. Combine experts | `HardGating`, `SoftGating`, `AdaptiveGating` | Expert weights per time step |
| 9. All together | `MoEForecaster` | `fit`, `predict`, `predict_with_metadata`, `expert_weights`, `forecast` |
| 10. Persist | `save`, `load` | Serialization with version check |
| Metrics | `moeforecast.metrics` | MAE, RMSE, MAPE, MASE |

## Installation

```bash
pip install "moeforecast[all]"          # with LightGBM and public holidays
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

# 3. Regimes
days = ProfileClusterer("D", name="day").fit(train)
weeks = ProfileClusterer("W", name="week", n_clusters=2).fit(train)
days.plot_scores()
days.plot_profiles()

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

The [quickstart notebook](examples/quickstart.ipynb) walks through this on synthetic data.

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
