# Design decisions

A Mixture-of-Experts forecaster built by hand, with each step in a separate script or notebook, is easy to get subtly wrong. `moeforecast` is designed so that the most common mistakes cannot happen. This document maps each pitfall to the decision that prevents it.

## Scope

The library covers **building** the model: data preparation, analysis to choose the configuration, features, regimes, classifiers, experts, gating, prediction and persistence. Benchmarks, statistical comparison tests and robustness studies belong to each user's research and are out of scope, as in scikit-learn.

## Common pitfalls and how the library handles them

| Pitfall | Decision in `moeforecast` |
|---|---|
| Expert ids built as `day0_week1` in one place and `0_1` in another, so lookups miss and every prediction falls back to the mean of all experts. | `ExpertKey` is a tuple of regime labels, used by the selector, the gating and the forecaster. Names are only for display (`format_key`). Test: `test_experts_are_routed_not_averaged`. |
| Experts and classifiers trained on all data, including the later test period. | `MoEForecaster.fit` receives only the training series; every component is fitted inside it. Evaluation means fitting on `train` and predicting on the full series (`temporal_train_test_split`). Test: `test_training_does_not_see_test_period`. |
| Hidden selection logic, e.g. picking a configuration by position in a ranking instead of by its score. | No hidden selection logic: the user picks the gating rule and its parameters explicitly. Gating tuning (roadmap) will select the minimum of a stated metric. |
| Reporting the score of the last candidate evaluated instead of the selected one. | `ExpertSelector.results_` keeps every candidate's score with a `selected` flag; the selected model is the one refitted. |
| Hyperparameters saved under one name (`reg__alpha`) and read under another (`alpha`), so they are silently dropped or applied to another model family. | Experts are scikit-learn estimators passed and cloned whole, so their parameters travel with them. (HPO on the roadmap will use the same estimator objects.) |
| Tuning one gating rule (e.g. centroid distance softmax) and deploying another (e.g. classifier confidence). | A single gating object, `MoEForecaster.gating`, is used both to evaluate and to predict. |
| Clustering raw profiles while the documentation and metadata say "z-normalized". | `ProfileClusterer(normalize=None | "zscore" | "mean")` is an explicit parameter, stored on the object. |
| Overriding the number of clusters by hand and reporting metrics of another configuration. | `evaluate(y)` returns candidate metrics without fitting final clusters. Users inspect the plots, choose `n_clusters`, then call `fit`; automatic selection requires explicit `auto_select=True`. |
| Classifier CV with shuffled stratified folds, mixing hours of the same day across folds. | `compare_classifiers` uses `TimeSeriesSplit` (with an optional gap). |
| Distance-based soft gating that compares feature vectors with centroids from another space. | The gating uses the classifiers' probabilities only. |
| Features such as bridge days computed differently for training and for forecasting. | One `FeatureBuilder.transform` for all uses. Holiday features come from the `holidays` package for any country. |
| Fourier terms counted from the first row of the training data. | The phase is anchored to 1970-01-01. Test: `test_fourier_terms_do_not_depend_on_series_start`. |
| MASE scale computed on non-contiguous test subsets. | `metrics.mase` uses the in-sample seasonal naive error of the contiguous training series (Hyndman & Koehler, 2006). |
| Synthetic-data fallbacks when an intermediate artifact is missing. | Not applicable: one object, no intermediate files. |
| Range checks in the wrong unit and model-card claims that the code does not support. | The library makes no claims beyond what its methods compute. `predict_with_metadata` reports what the gating actually did. |

Statistical tests, timing studies, SHAP-based selection and event analyses belong to benchmarking, which is out of scope.

## Components and their contracts

- **`prepare_series`** returns a float Series on a regular grid (missing steps are NaN) with `index.freq` set, warns about gaps, and never imputes automatically. Every other component expects this.
- **`FeatureBuilder`** is a scikit-learn transformer. `transform(y)` returns features aligned with `y.index`. Rows without enough history contain NaN and are skipped by every consumer.
- **`ProfileClusterer`** validates and clusters complete daily, weekly or monthly profiles. Euclidean distance is built in; DTW is an optional extra. `labels_for(index)` maps timestamps to regime labels.
- **`RegimeClassifier`** wraps any classifier with `predict_proba`. The gating aligns probabilities by `classes_`, so a regime missing from training gets zero weight.
- **`ExpertSelector`** follows the rule *candidates allowed by complexity → forward-chaining CV → best by metric → refit on the whole regime*.
- **Gating** maps per-dimension probabilities to weights over the existing experts. Weights are re-normalized over the experts that exist, so a missing combination never routes to nothing.
- **`MoEForecaster`** ties the components together. It has no fallback "average of all experts": hard gating picks the most probable *existing* expert.
