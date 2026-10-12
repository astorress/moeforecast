# Design decisions

`moeforecast` generalizes the Mixture-of-Experts pipeline developed for hourly electricity demand in Colombia's SIN ([MoEForecaster](https://github.com/astorress/MoEForecaster)). A review of that pipeline's notebooks found issues that are easy to introduce when the steps live in separate notebooks. The library is designed so that they cannot happen. This document maps each issue to the decision that prevents it.

## Scope

The library covers **building** the model: data preparation, analysis to choose the configuration, features, regimes, classifiers, experts, gating, prediction and persistence. Benchmarks, statistical comparison tests and robustness studies belong to each user's research and are out of scope, as in scikit-learn.

## Issues in the original pipeline and how the library handles them

| # | Issue in the notebooks | Decision in `moeforecast` |
|---|---|---|
| 1 | Expert ids built as `day0_week1` in one place and `0_1` in another. Every prediction fell back to the mean of all experts. | `ExpertKey` is a tuple of regime labels, used by the selector, the gating and the forecaster. Names are only for display (`format_key`). Test: `test_experts_are_routed_not_averaged`. |
| 2 | Experts and classifiers were trained on all data, including the later test period. | `MoEForecaster.fit` receives only the training series; every component is fitted inside it. Evaluation means fitting on `train` and predicting on the full series (`temporal_train_test_split`). Test: `test_training_does_not_see_test_period`. |
| 3 | The "best" gating configuration was the 9th in the ranking (`nsmallest(9).index[8]`). | No hidden selection logic: the user picks the gating rule and its parameters explicitly. Gating tuning (roadmap) will select the minimum of a stated metric. |
| 4 | The heterogeneous global MASE used the last candidate evaluated, not the selected one. | `ExpertSelector.results_` keeps every candidate's score with a `selected` flag; the selected model is the one refitted. |
| 5 | HPO parameters were saved as `reg__alpha` and read as `alpha`; several were silently dropped or swapped for another model family. | Experts are scikit-learn estimators passed and cloned whole, so their parameters travel with them. (HPO on the roadmap will use the same estimator objects.) |
| 6 | The gating tuned in one notebook (centroid distance softmax) was not the one used in production (classifier confidence). | A single gating object, `MoEForecaster.gating`, is used both to evaluate and to predict. |
| 9 | Clustering ran on raw profiles while the documentation and metadata said "z-normalized". | `ProfileClusterer(normalize=None | "zscore" | "mean")` is an explicit parameter, stored on the object. |
| 10 | The number of clusters was overridden by hand and the summary printed metrics of another configuration. | `n_clusters` fixes k explicitly; otherwise it is chosen by `selection_metric`. `scores_` holds every candidate k. |
| 11 | Classifier CV used shuffled stratified folds, mixing hours of the same day across folds. | `compare_classifiers` uses `TimeSeriesSplit` (with an optional gap). |
| 12 | Distance-based soft gating compared feature vectors with centroids from another space. | The gating uses the classifiers' probabilities only. |
| 13 | `is_bridge` and other features were computed differently for training and for forecasting. | One `FeatureBuilder.transform` for all uses. Holiday features come from the `holidays` package for any country. |
| — | Fourier terms counted from the first row of the training data. | The phase is anchored to 1970-01-01. Test: `test_fourier_terms_do_not_depend_on_series_start`. |
| — | MASE scale computed on non-contiguous test subsets. | `metrics.mase` uses the in-sample seasonal naive error of the contiguous training series (Hyndman & Koehler, 2006). |
| — | Synthetic-data fallbacks when an artifact was missing. | Not applicable: one object, no intermediate files. |
| — | Range checks in the wrong unit and claims in the model card that the code did not support. | The library makes no claims beyond what its methods compute. `predict_with_metadata` reports what the gating actually did. |

Issues 7, 8 and the remaining ones in notebook 09 (statistical tests, hard-coded timings, SHAP selection, COVID analysis) belong to the benchmark, which is out of scope.

## Components and their contracts

- **`prepare_series`** returns a float Series on a regular grid (missing steps are NaN) with `index.freq` set. Every other component expects this.
- **`FeatureBuilder`** is a scikit-learn transformer. `transform(y)` returns features aligned with `y.index`. Rows without enough history contain NaN and are skipped by every consumer.
- **`ProfileClusterer`** clusters complete calendar periods (`"D"`, `"W"`, ...). `labels_for(index)` maps timestamps to regime labels.
- **`RegimeClassifier`** wraps any classifier with `predict_proba`. The gating aligns probabilities by `classes_`, so a regime missing from training gets zero weight.
- **`ExpertSelector`** follows the rule *candidates allowed by complexity → forward-chaining CV → best by metric → refit on the whole regime*.
- **Gating** maps per-dimension probabilities to weights over the existing experts. Weights are re-normalized over the experts that exist, so a missing combination never routes to nothing.
- **`MoEForecaster`** ties the components together. It has no fallback "average of all experts": hard gating picks the most probable *existing* expert.
