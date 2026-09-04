# IDS-Agent Phase 1 — Baseline vs. Paper

Generated 2026-09-02. Paper figures from Li, Xiang, Bastian, Song, Bo Li — IDS-Agent, NeurIPS 2024 Workshop on Open-World Agents.

## Headline comparison

Comparing our **weighted_f1** against the paper's reported F1. Within 0.02 counts as matched.

| Benchmark | Paper | Ours | Best classifier | Δ | Verdict |
|---|---|---|---|---|---|
| cic_iot_2023 | 0.75 | 0.7408 | random_forest | -0.0092 | matched |
| zero-day | 0.61 | — | — | — | not run (no run had `holdout_classes` configured) |

## The same predictions, scored four ways

Which average is used decides whether the baseline looks reproduced. These are the same models as above.

| Dataset | Classifier | `weighted_f1` | `macro_f1` | `macro_f1_present_only` | `accuracy` |
|---|---|---|---|---|---|
| cic_iot_2023 | random_forest | 0.7408 | 0.6351 | — | 0.7473 |

## cic_iot_2023 — all classifiers

| classifier | n_train_used | subsampled | fit_seconds | predict_seconds | accuracy | balanced_accuracy | macro_f1 | weighted_f1 | macro_precision | macro_recall | benign_false_positive_rate | attack_detection_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random_forest | 901577 | False | 153.5400 | 5.4000 | 0.7473 | 0.6270 | 0.6351 | 0.7408 | 0.6796 | 0.6270 | 0.3192 | 0.9741 |
| mlp | 901577 | False | 1158.6600 | 0.2300 | 0.7347 | 0.6107 | 0.6175 | 0.7270 | 0.6839 | 0.6107 | 0.3324 | 0.9716 |
| decision_tree | 901577 | False | 24.7700 | 0.1000 | 0.7021 | 0.5955 | 0.5957 | 0.7021 | 0.5964 | 0.5955 | 0.5203 | 0.9793 |
| knn | 200000 | True | 0.0400 | 63.2100 | 0.6917 | 0.5745 | 0.5763 | 0.6888 | 0.5934 | 0.5745 | 0.3836 | 0.9578 |
| logistic_regression | 901577 | False | 398.7600 | 0.0400 | 0.6785 | 0.5488 | 0.5451 | 0.6644 | 0.6141 | 0.5488 | 0.4994 | 0.9692 |
| svm | 50000 | True | 29.3500 | 871.3600 | 0.6550 | 0.5297 | 0.5243 | 0.6396 | 0.5876 | 0.5297 | 0.3974 | 0.9639 |

## Caveats

1. **The paper's averaging method is inferred, not confirmed.** It reports an F1 without saying whether it is macro, weighted or micro. On this data the gap between them reaches 0.17, so the comparison metric below is an assumption. Check the paper before claiming the baseline is reproduced.
2. **Some classifiers trained on a capped subset**, so their scores are not directly comparable with the rest: `knn` on cic_iot_2023 (200,000 rows), `svm` on cic_iot_2023 (50,000 rows). Kernel SVC is quadratic-to-cubic in training rows and KNN pays `n_train x n_test` at predict time; neither finishes on the full split. See `training.max_train_rows` in config/models.yaml.
