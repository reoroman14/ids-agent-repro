# IDS-Agent Phase 1 — Baseline vs. Paper

Generated 2026-09-02. Paper figures from Li, Xiang, Bastian, Song, Bo Li — IDS-Agent, NeurIPS 2024 Workshop on Open-World Agents.

## Headline comparison

Comparing our **weighted_f1** against the paper's reported F1. Within 0.02 counts as matched.

| Benchmark | Paper | Ours | Best classifier | Δ | Verdict |
|---|---|---|---|---|---|
| aci_iot_2023 | 0.97 | 0.9995 | random_forest | 0.0295 | above paper |
| zero-day | 0.61 | — | — | — | not run (no run had `holdout_classes` configured) |

## The same predictions, scored four ways

Which average is used decides whether the baseline looks reproduced. These are the same models as above.

| Dataset | Classifier | `weighted_f1` | `macro_f1` | `macro_f1_present_only` | `accuracy` |
|---|---|---|---|---|---|
| aci_iot_2023 | random_forest | 0.9995 | 0.9062 | 0.9886 | 0.9995 |

## aci_iot_2023 — all classifiers

| classifier | n_train_used | subsampled | fit_seconds | predict_seconds | accuracy | balanced_accuracy | macro_f1 | macro_f1_present_only | weighted_f1 | macro_precision | macro_recall | benign_false_positive_rate | attack_detection_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random_forest | 861989 | False | 95.2100 | 1.5900 | 0.9995 | 0.9822 | 0.9062 | 0.9886 | 0.9995 | 0.9131 | 0.9003 | 0.0005 | 0.9996 |
| decision_tree | 861989 | False | 22.9400 | 0.0800 | 0.9994 | 0.9846 | 0.9047 | 0.9869 | 0.9994 | 0.9070 | 0.9025 | 0.0010 | 0.9996 |
| mlp | 861989 | False | 802.8500 | 0.2800 | 0.9951 | 0.9510 | 0.8768 | 0.9565 | 0.9951 | 0.8852 | 0.8717 | 0.0140 | 0.9989 |
| knn | 200000 | True | 0.1300 | 84.9800 | 0.9928 | 0.9173 | 0.8474 | 0.9244 | 0.9927 | 0.8622 | 0.8409 | 0.0105 | 0.9969 |
| logistic_regression | 861989 | False | 146.6000 | 0.0800 | 0.9828 | 0.9027 | 0.8307 | 0.9062 | 0.9826 | 0.8677 | 0.8275 | 0.0416 | 0.9937 |
| svm | 50000 | True | 43.2300 | 276.7700 | 0.9741 | 0.8647 | 0.7883 | 0.8599 | 0.9737 | 0.7894 | 0.7927 | 0.0495 | 0.9935 |

## Caveats

1. **The paper's averaging method is inferred, not confirmed.** It reports an F1 without saying whether it is macro, weighted or micro. On this data the gap between them reaches 0.17, so the comparison metric below is an assumption. Check the paper before claiming the baseline is reproduced.
2. **Some classifiers trained on a capped subset**, so their scores are not directly comparable with the rest: `knn` on aci_iot_2023 (200,000 rows), `svm` on aci_iot_2023 (50,000 rows). Kernel SVC is quadratic-to-cubic in training rows and KNN pays `n_train x n_test` at predict time; neither finishes on the full split. See `training.max_train_rows` in config/models.yaml.
