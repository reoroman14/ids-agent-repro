"""
Train and evaluate each classifier on ACI-IoT'23 and CIC-IoT'23.

Target baseline (original IDS-Agent paper, for comparison):
- ACI-IoT'23: F1 ~ 0.97
- CIC-IoT'23: F1 ~ 0.75
- Zero-day attack recall ~ 0.61

Scoring is delegated to ``src/evaluation/metrics.py`` so classifier-only
baselines and later full-agent runs are scored by the exact same code.

Two properties this module is careful about:

Nothing is silently subsampled.
    Kernel SVC and KNN cannot run on ~900k training rows in useful time (see
    `training.max_train_rows` in config/models.yaml). Where a cap applies, the
    subsample is stratified, drawn with the split seed, and reported as
    ``n_train_used`` alongside a ``subsampled`` flag, so a row of the results
    table can never quietly mean something different from its neighbours.

Metrics are computed over the frozen class list.
    Not over the classes present in a given split — see the note in
    ``metrics.py`` about why that matters once ``holdout_classes`` is in play.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from ..data import loader as data_loader
from ..data import preprocessing as prep
from ..evaluation import metrics as metrics_mod
from . import registry

DEFAULT_MODELS_CONFIG = "config/models.yaml"

#: Columns of the results table, in display order.
RESULT_COLUMNS = [
    "classifier",
    "n_train_used",
    "subsampled",
    "fit_seconds",
    "predict_seconds",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    # Reported next to macro_f1 because on ACI they differ a lot: a class routed
    # train-only has zero test support, so plain macro averages a structural 0
    # into the mean. Both belong in the table rather than one being chosen.
    "macro_f1_present_only",
    "weighted_f1",
    "macro_precision",
    "macro_recall",
    "benign_false_positive_rate",
    "attack_detection_rate",
    "zero_day_recall",
]


def load_models_config(path: str = DEFAULT_MODELS_CONFIG) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Subsampling
# ---------------------------------------------------------------------------

def _stratified_subsample(
    X: pd.DataFrame, y: np.ndarray, max_rows: int, seed: int
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Take at most ``max_rows`` rows, preserving class proportions.

    Every class keeps at least one row even when its proportional share rounds
    to zero — dropping a rare class entirely would change which classes the
    model can predict, which is a different experiment, not a smaller one.
    """
    n = len(y)
    if max_rows is None or n <= max_rows:
        return X, y

    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) > max_rows:
        raise ValueError(
            f"max_train_rows={max_rows} is smaller than the number of classes "
            f"({len(classes)}); every class must keep at least one row."
        )

    quota = np.maximum(1, np.floor(counts / n * max_rows).astype(int))
    quota = np.minimum(quota, counts)

    # Distribute the rounding remainder to the largest classes.
    remainder = max_rows - int(quota.sum())
    if remainder > 0:
        headroom = counts - quota
        for i in np.argsort(-counts):
            if remainder <= 0:
                break
            take = min(remainder, int(headroom[i]))
            quota[i] += take
            remainder -= take

    picked: List[np.ndarray] = []
    for cls, k in zip(classes, quota):
        idx = np.flatnonzero(y == cls)
        picked.append(rng.choice(idx, size=int(k), replace=False) if k < len(idx) else idx)

    keep = np.sort(np.concatenate(picked))
    return X.iloc[keep], y[keep]


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train_classifier(
    name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    *,
    seed: int = 42,
    n_jobs: Optional[int] = None,
    max_train_rows: Optional[int] = None,
) -> Any:
    """
    Build and fit one classifier.

    Returns the fitted estimator. Details of the run that a caller cannot
    otherwise recover — rows actually used, whether a cap applied, fit time —
    are attached as ``model.fit_info_`` so ``run_all`` can report them without
    a second return value changing the stub's signature.
    """
    canonical = registry.resolve_name(name)
    model = registry.build_classifier(canonical, params, seed=seed, n_jobs=n_jobs)

    X_fit, y_fit = _stratified_subsample(X_train, y_train, max_train_rows, seed)
    subsampled = len(y_fit) != len(y_train)
    if subsampled:
        warnings.warn(
            f"{canonical}: training on {len(y_fit):,} of {len(y_train):,} rows "
            f"(max_train_rows={max_train_rows}). Its scores are not directly "
            f"comparable with classifiers trained on the full split.",
            stacklevel=2,
        )
    elif registry.spec(canonical).superlinear_in_rows and len(y_fit) > 250_000:
        warnings.warn(
            f"{canonical} scales superlinearly in training rows and has no "
            f"max_train_rows cap; {len(y_fit):,} rows may take a very long "
            f"time. {registry.spec(canonical).note}",
            stacklevel=2,
        )

    started = time.time()
    model.fit(X_fit, y_fit)
    elapsed = time.time() - started

    model.fit_info_ = {
        "classifier": canonical,
        "n_train_available": int(len(y_train)),
        "n_train_used": int(len(y_fit)),
        "subsampled": bool(subsampled),
        "max_train_rows": max_train_rows,
        "fit_seconds": float(elapsed),
        "seed": seed,
        "classes_seen": int(len(np.unique(y_fit))),
    }
    return model


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate(
    model: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    *,
    class_names: Optional[Sequence[str]] = None,
    holdout_classes: Optional[Sequence[str]] = None,
    include_confusion: bool = True,
) -> Dict[str, Any]:
    """
    Score a fitted model on a test split.

    ``class_names`` should be ``Splits.classes_`` — the frozen class list — so
    that every class gets a per-class row even when the model can never predict
    it, and macro averages are taken over a fixed denominator.
    """
    started = time.time()
    y_pred = model.predict(X_test)
    predict_seconds = time.time() - started

    result = metrics_mod.compute_metrics(
        y_test,
        y_pred,
        class_names=class_names,
        holdout_classes=holdout_classes,
        include_confusion=include_confusion,
    )
    result["predict_seconds"] = float(predict_seconds)
    result["score_kind"] = registry.score_kind(model)
    if hasattr(model, "fit_info_"):
        result["fit_info"] = dict(model.fit_info_)
    return result


# ---------------------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------------------

def run_all(
    dataset_name: str,
    config: Optional[Dict[str, Any]] = None,
    models_config: Optional[Dict[str, Any]] = None,
    *,
    splits: Optional[prep.Splits] = None,
    classifiers: Optional[Sequence[str]] = None,
    target: str = "label",
    output_dir: Optional[str] = None,
    include_high_leakage: bool = True,
    use_cache: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Train and evaluate every configured classifier on one dataset.

    Returns a results table, one row per classifier, sorted by macro F1.

    Pass ``splits`` to reuse an already-prepared :class:`~src.data.preprocessing.Splits`
    (the loaders take minutes on CIC); otherwise the dataset is loaded and
    prepared here. With ``output_dir`` set, the table is written as
    ``results.csv`` and each classifier's full metrics — per-class breakdown and
    confusion matrix included — as ``<classifier>.json``.

    A classifier that fails is recorded with its error rather than aborting the
    sweep: a five-hour run should not be lost to one estimator's convergence
    failure.
    """
    config = data_loader.load_config() if config is None else config
    models_config = load_models_config() if models_config is None else models_config

    training_cfg = models_config.get("training", {}) or {}
    n_jobs = training_cfg.get("n_jobs")
    caps = training_cfg.get("max_train_rows", {}) or {}
    seed = int(config.get("split", {}).get("seed", 42))
    holdout = list(config.get("split", {}).get("holdout_classes") or [])

    names = list(
        classifiers
        or training_cfg.get("classifiers")
        or registry.available_classifiers()
    )
    names = [registry.resolve_name(n) for n in names]

    if splits is None:
        df = data_loader.load_dataset(dataset_name, config, use_cache=use_cache)
        splits = prep.prepare_splits(
            df,
            config,
            models_config,
            target=target,
            include_high_leakage=include_high_leakage,
        )

    if verbose:
        print(splits.summary())
        if splits.report is not None:
            print(splits.report.summary())
        print()

    rows: List[Dict[str, Any]] = []
    details: Dict[str, Dict[str, Any]] = {}

    for name in names:
        if verbose:
            print(f"--- {name} ---", flush=True)
        try:
            model = train_classifier(
                name,
                splits.X_train,
                splits.y_train,
                models_config.get(name, {}),
                seed=seed,
                n_jobs=n_jobs,
                max_train_rows=caps.get(name),
            )
            result = evaluate(
                model,
                splits.X_test,
                splits.y_test,
                class_names=splits.classes_,
                holdout_classes=holdout,
            )
        except Exception as exc:  # noqa: BLE001 - one failure must not end the sweep
            warnings.warn(f"{name} failed: {type(exc).__name__}: {exc}", stacklevel=2)
            rows.append({"classifier": name, "error": f"{type(exc).__name__}: {exc}"})
            continue

        details[name] = result
        info = result.get("fit_info", {})
        rows.append(
            {
                "classifier": name,
                "n_train_used": info.get("n_train_used"),
                "subsampled": info.get("subsampled"),
                "fit_seconds": round(info.get("fit_seconds", float("nan")), 2),
                "predict_seconds": round(result["predict_seconds"], 2),
                **{
                    key: result[key]
                    for key in (
                        "accuracy",
                        "balanced_accuracy",
                        "macro_f1",
                        "weighted_f1",
                        "macro_precision",
                        "macro_recall",
                    )
                },
                **{
                    key: result[key]
                    for key in ("macro_f1_present_only",)
                    if key in result
                },
                **{
                    key: result[key]
                    for key in (
                        "benign_false_positive_rate",
                        "attack_detection_rate",
                        "zero_day_recall",
                    )
                    if key in result
                },
            }
        )
        if verbose:
            print(metrics_mod.summarize(result), flush=True)
            print()

    table = pd.DataFrame(rows)
    ordered = [c for c in RESULT_COLUMNS if c in table.columns]
    extra = [c for c in table.columns if c not in ordered]
    table = table[ordered + extra]
    if "macro_f1" in table.columns:
        table = table.sort_values("macro_f1", ascending=False, na_position="last")
    table = table.reset_index(drop=True)

    table.attrs["dataset"] = dataset_name
    table.attrs["target"] = target
    table.attrs["details"] = details

    if output_dir:
        _write_results(table, details, splits, output_dir, dataset_name, target)

    return table


def _write_results(
    table: pd.DataFrame,
    details: Dict[str, Dict[str, Any]],
    splits: prep.Splits,
    output_dir: str,
    dataset_name: str,
    target: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    table.to_csv(os.path.join(output_dir, "results.csv"), index=False)

    for name, result in details.items():
        with open(os.path.join(output_dir, f"{name}.json"), "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)

    run = {
        "dataset": dataset_name,
        "target": target,
        "n_features": len(splits.feature_names),
        "feature_names": splits.feature_names,
        "classes": splits.classes_,
        "n_train": int(len(splits.y_train)),
        "n_val": int(len(splits.y_val)),
        "n_test": int(len(splits.y_test)),
        "scaler": getattr(splits.numeric_transform, "scaler_kind", None),
    }
    if splits.report is not None:
        run["split"] = {
            "seed": splits.report.seed,
            "ratios": list(splits.report.ratios),
            "rare_classes": splits.report.rare_classes,
            "holdout_classes": splits.report.holdout_classes,
            "classes_absent_from_test": splits.report.classes_absent_from_test,
        }
    with open(os.path.join(output_dir, "run.json"), "w", encoding="utf-8") as fh:
        json.dump(run, fh, indent=2)


__all__ = [
    "RESULT_COLUMNS",
    "evaluate",
    "load_models_config",
    "run_all",
    "train_classifier",
]
