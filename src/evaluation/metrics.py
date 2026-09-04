"""
Standard metrics: F1, recall, precision, per-class breakdown.
Kept separate from train_eval.py so both classifier-only baselines
and full-agent runs share the exact same scoring code.

Two conventions here matter for comparing against the IDS-Agent paper.

`labels` is always passed explicitly.
    Scores are computed over the full frozen class list, not over the classes
    that happen to appear in ``y_true``/``y_pred``. With ``holdout_classes``
    set, the held-out class exists in the test set but never in training, so a
    classifier can never emit it; letting sklearn infer the label set would
    silently drop that class from the per-class report and quietly inflate
    macro-F1 by shrinking its denominator.

Zero-day recall is a detection question, not a classification one.
    A model cannot name a class it never trained on, so the held-out class's
    per-class recall is 0 by construction and says nothing. What the paper's
    ~0.61 measures is whether held-out attack traffic is flagged as *an attack
    at all*. ``zero_day_recall`` is therefore the fraction of held-out-class
    rows predicted as any non-benign class.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)

BENIGN = "Benign"


def _benign_index(class_names: Sequence[str]) -> Optional[int]:
    for i, name in enumerate(class_names):
        if str(name) == BENIGN:
            return i
    return None


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class_names: Optional[Sequence[str]] = None,
    labels: Optional[Sequence[int]] = None,
    holdout_classes: Optional[Sequence[str]] = None,
    include_confusion: bool = True,
) -> Dict[str, Any]:
    """
    Score a set of predictions.

    ``class_names`` should be the frozen class list (``Splits.classes_``), so
    ``labels`` defaults to every class index rather than only those observed.

    Returns a flat dict of headline scores plus a ``per_class`` breakdown, and
    — when the label set includes ``Benign`` — the detection-oriented figures an
    IDS is actually judged on: the false-positive rate on benign traffic and
    the overall attack detection rate.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got {y_true.shape} "
            f"and {y_pred.shape}."
        )

    if class_names is None and labels is None:
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
        class_names = [str(i) for i in labels]
    elif labels is None:
        labels = list(range(len(class_names)))
    elif class_names is None:
        class_names = [str(i) for i in labels]

    labels = list(labels)
    class_names = [str(c) for c in class_names]

    with warnings.catch_warnings():
        # balanced_accuracy_score warns "y_pred contains classes not in y_true"
        # whenever a model predicts a class that has no test support. On ACI a
        # train-only rare class makes that the expected case on every run, so
        # the warning is noise about a deliberate design decision rather than a
        # signal. Scoped to this call so nothing else is muted.
        warnings.filterwarnings(
            "ignore", message="y_pred contains classes not in y_true"
        )
        balanced = float(balanced_accuracy_score(y_true, y_pred))

    out: Dict[str, Any] = {
        "n_samples": int(y_true.size),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": balanced,
    }

    for average in ("macro", "weighted", "micro"):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=average, zero_division=0
        )
        out[f"{average}_precision"] = float(p)
        out[f"{average}_recall"] = float(r)
        out[f"{average}_f1"] = float(f)

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    out["per_class"] = {
        class_names[i]: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(s[i]),
        }
        for i in range(len(labels))
    }

    # Macro scores over only the classes actually present in y_true. With a
    # train-only rare class the plain macro average includes a class with zero
    # support, which drags the mean toward 0 for a reason unrelated to model
    # quality. Both are reported so the comparison is explicit.
    present = [i for i in range(len(labels)) if s[i] > 0]
    if present and len(present) < len(labels):
        out["macro_f1_present_only"] = float(np.mean([f[i] for i in present]))
        out["macro_recall_present_only"] = float(np.mean([r[i] for i in present]))
        out["n_classes_present"] = len(present)
        out["n_classes_total"] = len(labels)

    benign = _benign_index(class_names)
    if benign is not None:
        is_benign_true = y_true == benign
        n_benign = int(is_benign_true.sum())
        n_attack = int((~is_benign_true).sum())
        if n_benign:
            # Benign traffic predicted as some attack class.
            out["benign_false_positive_rate"] = float(
                (y_pred[is_benign_true] != benign).sum() / n_benign
            )
        if n_attack:
            # Attack traffic flagged as an attack, right class or not.
            out["attack_detection_rate"] = float(
                (y_pred[~is_benign_true] != benign).sum() / n_attack
            )

    if holdout_classes:
        out["zero_day"] = _zero_day_metrics(
            y_true, y_pred, class_names, holdout_classes, benign
        )
        out["zero_day_recall"] = out["zero_day"]["recall"]

    if include_confusion:
        out["confusion_matrix"] = confusion_matrix(
            y_true, y_pred, labels=labels
        ).tolist()
        out["confusion_labels"] = class_names

    return out


def _zero_day_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    holdout_classes: Sequence[str],
    benign: Optional[int],
) -> Dict[str, Any]:
    """
    Recall on classes withheld from training.

    Measured as "flagged as an attack at all", because a multi-class model has
    no output for a class it never saw. Reported per held-out class as well, so
    a single easy holdout cannot hide a hard one behind the average.
    """
    name_to_index = {name: i for i, name in enumerate(class_names)}
    unknown = [c for c in holdout_classes if c not in name_to_index]
    if unknown:
        raise ValueError(
            f"holdout_classes {unknown} are not in class_names; pass the frozen "
            f"class list so held-out classes can be scored."
        )
    if benign is None:
        raise ValueError(
            "zero-day recall needs a 'Benign' class in class_names to define "
            "what counts as being flagged as an attack."
        )

    indices = [name_to_index[c] for c in holdout_classes]
    mask = np.isin(y_true, indices)
    n = int(mask.sum())
    result: Dict[str, Any] = {
        "classes": list(holdout_classes),
        "n_samples": n,
        "recall": float((y_pred[mask] != benign).sum() / n) if n else float("nan"),
        "per_class": {},
    }
    for name, idx in zip(holdout_classes, indices):
        m = y_true == idx
        k = int(m.sum())
        result["per_class"][name] = {
            "n_samples": k,
            "recall": float((y_pred[m] != benign).sum() / k) if k else float("nan"),
        }
    return result


def zero_day_tradeoff(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    class_names: Sequence[str],
    holdout_classes: Sequence[str],
    model_classes: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """
    Trade-off curve between catching zero-day attacks and false-alarming on
    benign traffic.

    Why this exists: a single zero-day recall number is not interpretable.
    Measured across 27 leave-out runs, recall correlated with the benign
    false-alarm rate at r = 0.843 — a model that simply calls everything an
    attack scores 1.0 and is useless. The pair has to move together, so this
    reports the whole curve rather than one point on it.

    Framed as the detection question it actually is: **positives are the
    withheld zero-day rows, negatives are benign rows**, and the score is
    ``1 - P(Benign)``. Known attack classes are excluded — they are neither
    the thing being detected nor the thing being false-alarmed on, and folding
    them in would let good performance on familiar attacks disguise poor
    performance on novel ones.

    ``scores`` should come from ``registry.predict_scores``, which returns
    probabilities where available and a softmax over the decision function
    otherwise. The curve only depends on the ranking, so either works; the
    thresholds are not comparable across the two kinds.

    ``model_classes`` must be the estimator's ``classes_`` whenever the model
    was trained on a subset of ``class_names`` — which is always true here,
    since withholding a class is the entire point. A held-out class produces no
    score column, so the columns are NOT aligned with ``class_names``, and
    assuming they are would read the wrong column as Benign in any case where
    the counts happened to coincide.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    class_names = [str(c) for c in class_names]

    if scores.ndim != 2 or scores.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"scores must be (n_samples, n_classes); got {scores.shape} for "
            f"{y_true.shape[0]} samples."
        )

    benign = _benign_index(class_names)
    if benign is None:
        raise ValueError(
            "the trade-off curve needs a 'Benign' class in class_names to "
            "define the false-alarm axis."
        )

    # Locate the Benign column in the score matrix, which is indexed by the
    # classes the model was fitted on, not by the frozen class list.
    if model_classes is None:
        if scores.shape[1] != len(class_names):
            raise ValueError(
                f"scores has {scores.shape[1]} columns but {len(class_names)} "
                f"class names were given. Pass model_classes=model.classes_ — "
                f"a model trained with a held-out class emits no column for it."
            )
        benign_column = benign
    else:
        model_classes = np.asarray(model_classes)
        if scores.shape[1] != len(model_classes):
            raise ValueError(
                f"scores has {scores.shape[1]} columns but model_classes has "
                f"{len(model_classes)} entries."
            )
        found = np.flatnonzero(model_classes == benign)
        if found.size == 0:
            raise ValueError(
                "the model was never trained on the Benign class, so it cannot "
                "score anything as benign and the false-alarm axis is undefined."
            )
        benign_column = int(found[0])

    name_to_index = {name: i for i, name in enumerate(class_names)}
    unknown = [c for c in holdout_classes if c not in name_to_index]
    if unknown:
        raise ValueError(f"holdout_classes {unknown} are not in class_names.")
    held = [name_to_index[c] for c in holdout_classes]

    attack_score = 1.0 - scores[:, benign_column]
    is_benign = y_true == benign
    is_zero_day = np.isin(y_true, held)

    n_benign, n_zero_day = int(is_benign.sum()), int(is_zero_day.sum())
    if not n_benign or not n_zero_day:
        raise ValueError(
            f"need both benign and zero-day rows to build a curve; got "
            f"{n_benign} benign and {n_zero_day} zero-day."
        )

    mask = is_benign | is_zero_day
    fpr, recall, thresholds = roc_curve(is_zero_day[mask].astype(int), attack_score[mask])

    return {
        "fpr": fpr.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
        "auc": float(auc(fpr, recall)),
        "n_benign": n_benign,
        "n_zero_day": n_zero_day,
        "holdout_classes": list(holdout_classes),
    }


def recall_at_fpr(curve: Dict[str, Any], target_fpr: float) -> float:
    """Zero-day recall achievable at a given benign false-alarm rate."""
    return float(np.interp(target_fpr, curve["fpr"], curve["recall"]))


def fpr_at_recall(curve: Dict[str, Any], target_recall: float) -> float:
    """
    Benign false-alarm rate required to reach a given zero-day recall.

    This is what turns a bare published recall figure into something
    interpretable: it says what the number costs.
    """
    return float(np.interp(target_recall, curve["recall"], curve["fpr"]))


def tradeoff_operating_points(
    curve: Dict[str, Any],
    fprs: Sequence[float] = (0.001, 0.01, 0.05, 0.10),
) -> Dict[str, float]:
    """Zero-day recall at a few plausible operational false-alarm budgets."""
    return {f"recall_at_fpr_{f:g}": recall_at_fpr(curve, f) for f in fprs}


def summarize_tradeoff(
    curve: Dict[str, Any], *, title: str = "", reference_recall: float = 0.61
) -> str:
    """Human-readable rendering, including what a reference recall would cost."""
    lines = [title] if title else []
    lines += [
        f"  held out           : {', '.join(curve['holdout_classes'])}",
        f"  zero-day / benign  : {curve['n_zero_day']:,} / {curve['n_benign']:,} rows",
        f"  AUC                : {curve['auc']:.4f}",
    ]
    for name, value in tradeoff_operating_points(curve).items():
        budget = name.rsplit("_", 1)[-1]
        lines.append(f"  recall @ {budget:>5} FPR : {value:.4f}")
    cost = fpr_at_recall(curve, reference_recall)
    lines.append(
        f"  recall {reference_recall:.2f} costs  : {cost:.4f} benign false-alarm rate"
    )
    return "\n".join(lines)


def summarize(metrics: Dict[str, Any], *, title: str = "") -> str:
    """Compact human-readable rendering of :func:`compute_metrics` output."""
    lines = [title] if title else []
    lines += [
        f"  samples            : {metrics['n_samples']:,}",
        f"  accuracy           : {metrics['accuracy']:.4f}",
        f"  balanced accuracy  : {metrics['balanced_accuracy']:.4f}",
        f"  macro    P/R/F1    : {metrics['macro_precision']:.4f} / "
        f"{metrics['macro_recall']:.4f} / {metrics['macro_f1']:.4f}",
        f"  weighted P/R/F1    : {metrics['weighted_precision']:.4f} / "
        f"{metrics['weighted_recall']:.4f} / {metrics['weighted_f1']:.4f}",
    ]
    if "macro_f1_present_only" in metrics:
        lines.append(
            f"  macro F1 (present) : {metrics['macro_f1_present_only']:.4f} "
            f"over {metrics['n_classes_present']}/{metrics['n_classes_total']} classes"
        )
    if "benign_false_positive_rate" in metrics:
        lines.append(f"  benign FPR         : {metrics['benign_false_positive_rate']:.4f}")
    if "attack_detection_rate" in metrics:
        lines.append(f"  attack detection   : {metrics['attack_detection_rate']:.4f}")
    if "zero_day_recall" in metrics:
        zd = metrics["zero_day"]
        lines.append(
            f"  zero-day recall    : {metrics['zero_day_recall']:.4f} "
            f"over {zd['n_samples']:,} rows of {', '.join(zd['classes'])}"
        )
    return "\n".join(lines)


def worst_classes(metrics: Dict[str, Any], n: int = 5) -> list:
    """The n lowest-F1 classes that have support. Useful for the agent's context."""
    rows = [
        (name, m["f1"], m["support"])
        for name, m in metrics.get("per_class", {}).items()
        if m["support"] > 0
    ]
    return sorted(rows, key=lambda r: r[1])[:n]


__all__ = [
    "compute_metrics",
    "fpr_at_recall",
    "recall_at_fpr",
    "summarize",
    "summarize_tradeoff",
    "tradeoff_operating_points",
    "worst_classes",
    "zero_day_tradeoff",
]
