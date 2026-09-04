"""
Registry of the six classifiers used by IDS-Agent as tools:
RF, KNN, LR, DT, MLP, SVM.

Each classifier is wrapped behind the same interface (fit/predict/
predict_proba) so the agent's classification tool can call any of
them interchangeably.

Canonical names match the keys in config/models.yaml
(``random_forest``, ``knn``, ``logistic_regression``, ``decision_tree``,
``mlp``, ``svm``); the short forms ``rf`` / ``lr`` / ``dt`` are accepted as
aliases everywhere a name is taken.

Two things this module handles so callers do not have to:

Reproducibility.
    ``random_state`` is injected into every estimator that accepts one, from
    the single seed in config/datasets.yaml. Forgetting it on one estimator is
    the kind of thing that makes a run un-reproducible in a way nobody notices
    until the numbers move.

A uniform score interface.
    ``SVC`` is the odd one out: it has no calibrated ``predict_proba`` unless
    built with ``probability=True``, which fits an internal 5-fold Platt
    calibration and is several times slower. That parameter is also deprecated
    in scikit-learn 1.9 and removed in 1.11, so new code wanting real SVM
    probabilities should wrap the estimator in
    ``CalibratedClassifierCV(SVC(), ensemble=False)`` rather than set it.

    Rather than pay that cost on every run, :func:`predict_scores` returns
    calibrated probabilities when available and a softmax over
    ``decision_function`` otherwise, so the agent tool always receives an
    (n_samples, n_classes) score matrix. :func:`score_kind` reports which of
    the two it got — they rank identically but are not interchangeable as
    confidences, and the agent should not present a softmaxed margin as a
    probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


@dataclass(frozen=True)
class ClassifierSpec:
    """How to build one classifier, and what it costs."""

    name: str
    factory: Callable[..., Any]
    scale_sensitive: bool
    #: Rough cost driver, used only for the warnings in train_eval.
    superlinear_in_rows: bool = False
    #: False where the estimator still accepts n_jobs but ignores it, so the
    #: injection below does not pass a parameter that only emits a warning.
    inject_n_jobs: bool = True
    note: str = ""


CLASSIFIER_SPECS: Dict[str, ClassifierSpec] = {
    "random_forest": ClassifierSpec(
        "random_forest", RandomForestClassifier, scale_sensitive=False,
        note="Scale-invariant; parallelises well over n_jobs.",
    ),
    "knn": ClassifierSpec(
        "knn", KNeighborsClassifier, scale_sensitive=True, superlinear_in_rows=True,
        note="Cost is at predict time: every test row is compared against every "
             "stored training row. At 75 features no tree index helps, so this "
             "is effectively brute force.",
    ),
    "logistic_regression": ClassifierSpec(
        "logistic_regression", LogisticRegression, scale_sensitive=True,
        inject_n_jobs=False,
        note="Needs scaled inputs to converge within max_iter. Still accepts "
             "n_jobs but has ignored it since scikit-learn 1.8 (removed in "
             "1.10), so it is not injected — passing it only raises a warning.",
    ),
    "decision_tree": ClassifierSpec(
        "decision_tree", DecisionTreeClassifier, scale_sensitive=False,
        note="Scale-invariant and fast; an unbounded max_depth will overfit "
             "but is the paper's default.",
    ),
    "mlp": ClassifierSpec(
        "mlp", MLPClassifier, scale_sensitive=True,
        note="Needs scaled inputs. Single-threaded through BLAS only.",
    ),
    "svm": ClassifierSpec(
        "svm", SVC, scale_sensitive=True, superlinear_in_rows=True,
        note="Kernel SVC is between quadratic and cubic in training rows and "
             "single-threaded. It is the one classifier here that will not "
             "finish on a million rows.",
    ),
}

ALIASES: Dict[str, str] = {
    "rf": "random_forest",
    "forest": "random_forest",
    "lr": "logistic_regression",
    "logreg": "logistic_regression",
    "dt": "decision_tree",
    "tree": "decision_tree",
    "nn": "mlp",
    "neural_network": "mlp",
    "svc": "svm",
}

#: Short-name view, matching the shape the agent's tool layer expects.
CLASSIFIER_REGISTRY: Dict[str, Callable[..., Any]] = {
    "rf": RandomForestClassifier,
    "knn": KNeighborsClassifier,
    "lr": LogisticRegression,
    "dt": DecisionTreeClassifier,
    "mlp": MLPClassifier,
    "svm": SVC,
}


def available_classifiers() -> List[str]:
    """Canonical names, in the order config/models.yaml lists them."""
    return list(CLASSIFIER_SPECS)


def resolve_name(name: str) -> str:
    """Canonical name for a name or alias, or raise with the valid options."""
    key = str(name).strip().lower()
    if key in CLASSIFIER_SPECS:
        return key
    if key in ALIASES:
        return ALIASES[key]
    raise KeyError(
        f"Unknown classifier {name!r}. Expected one of "
        f"{sorted(CLASSIFIER_SPECS)} or an alias from {sorted(ALIASES)}."
    )


def spec(name: str) -> ClassifierSpec:
    return CLASSIFIER_SPECS[resolve_name(name)]


def build_classifier(
    name: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    seed: Optional[int] = None,
    n_jobs: Optional[int] = None,
) -> Any:
    """
    Build an unfitted estimator from its config/models.yaml parameters.

    Unknown parameter names raise rather than being ignored. sklearn accepts
    only its own keyword names, so a typo like ``n_estimator`` in models.yaml
    would otherwise either explode deep inside sklearn or, worse, be silently
    dropped by a permissive wrapper and change the model without any signal.

    ``seed`` and ``n_jobs`` are injected only into estimators that accept them,
    and never override a value given explicitly in ``params``.
    """
    canonical = resolve_name(name)
    estimator_cls = CLASSIFIER_SPECS[canonical].factory
    params = dict(params or {})

    valid = set(estimator_cls().get_params())
    unknown = sorted(set(params) - valid)
    if unknown:
        raise ValueError(
            f"Unknown parameter(s) for {canonical}: {unknown}. "
            f"Valid parameters: {sorted(valid)}. Fix the `{canonical}` block in "
            f"config/models.yaml."
        )

    if seed is not None and "random_state" in valid and "random_state" not in params:
        params["random_state"] = seed
    if (
        n_jobs is not None
        and CLASSIFIER_SPECS[canonical].inject_n_jobs
        and "n_jobs" in valid
        and "n_jobs" not in params
    ):
        params["n_jobs"] = n_jobs

    return estimator_cls(**params)


def build_from_config(
    name: str,
    models_config: Dict[str, Any],
    *,
    seed: Optional[int] = None,
    n_jobs: Optional[int] = None,
) -> Any:
    """Build a classifier using its parameter block from config/models.yaml."""
    canonical = resolve_name(name)
    params = models_config.get(canonical, {}) or {}
    if not isinstance(params, dict):
        raise TypeError(
            f"config/models.yaml `{canonical}` must be a mapping of parameters, "
            f"got {type(params).__name__}."
        )
    return build_classifier(canonical, params, seed=seed, n_jobs=n_jobs)


# ---------------------------------------------------------------------------
# Uniform scoring interface
# ---------------------------------------------------------------------------

def supports_proba(model: Any) -> bool:
    """True when the fitted model exposes genuine calibrated probabilities."""
    if not hasattr(model, "predict_proba"):
        return False
    # SVC always defines predict_proba but raises unless probability=True.
    if isinstance(model, SVC):
        return bool(getattr(model, "probability", False))
    return True


def score_kind(model: Any) -> str:
    """``'proba'`` or ``'decision_function'`` — what :func:`predict_scores` returns."""
    if supports_proba(model):
        return "proba"
    if hasattr(model, "decision_function"):
        return "decision_function"
    return "none"


def predict_scores(model: Any, X) -> np.ndarray:
    """
    An (n_samples, n_classes) score matrix for any of the six classifiers.

    Returns ``predict_proba`` where available. For an ``SVC`` built without
    ``probability=True`` it returns a softmax over ``decision_function``, which
    preserves the ranking but is NOT a calibrated probability — check
    :func:`score_kind` before treating the values as confidences.
    """
    kind = score_kind(model)
    if kind == "proba":
        return np.asarray(model.predict_proba(X))
    if kind == "decision_function":
        scores = np.asarray(model.decision_function(X))
        if scores.ndim == 1:  # binary: decision_function returns one column
            scores = np.column_stack([-scores, scores])
        shifted = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)
    raise AttributeError(
        f"{type(model).__name__} exposes neither predict_proba nor "
        f"decision_function, so it cannot produce class scores."
    )


__all__ = [
    "ALIASES",
    "CLASSIFIER_REGISTRY",
    "CLASSIFIER_SPECS",
    "ClassifierSpec",
    "available_classifiers",
    "build_classifier",
    "build_from_config",
    "predict_scores",
    "resolve_name",
    "score_kind",
    "spec",
    "supports_proba",
]
