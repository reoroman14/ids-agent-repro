"""
Common interface for Phase 2 attacks.

Every attack answers the same question — given one flow that the defender
currently classifies correctly, can the attacker change it so the defender is
wrong, without spending more than a stated budget and without breaking the
attack's own function?

The budget is explicit because an evasion result without one is meaningless:
any sample can be misclassified if you are allowed to change it arbitrarily.
``AttackBudget`` records both how far features may move and which features may
move at all.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class AttackBudget:
    """
    What the attacker is allowed to do.

    Parameters
    ----------
    epsilon
        Maximum change per feature, in standard deviations of the training
        data. The features are already standardised, so 1.0 means "one standard
        deviation", which is a large and visible change in flow terms.
    max_features
        How many features may be changed at once. A real attacker rewriting
        every field of every packet is less plausible than one adjusting a few.
    max_cost
        The highest ``manipulation_cost`` the attacker will accept, from
        ``feature_schema``. ``"low"`` is the strict, realistic setting.
    max_queries
        Query limit for search-based attacks, modelling an attacker who can
        probe the detector but not indefinitely.
    """

    epsilon: float = 0.5
    max_features: Optional[int] = None
    max_cost: str = "low"
    max_queries: int = 100

    def describe(self) -> str:
        parts = [f"eps={self.epsilon} sd", f"cost<={self.max_cost}"]
        if self.max_features:
            parts.append(f"<={self.max_features} features")
        parts.append(f"<={self.max_queries} queries")
        return ", ".join(parts)


@dataclass
class AttackResult:
    """Outcome of attacking one flow."""

    success: bool
    original_prediction: str = ""
    adversarial_prediction: str = ""
    true_label: str = ""
    #: The perturbed feature row, when the attack produced one.
    adversarial_row: Optional[pd.Series] = None
    perturbation: Dict[str, float] = field(default_factory=dict)
    n_features_changed: int = 0
    l2_norm: float = 0.0
    linf_norm: float = 0.0
    queries_used: int = 0
    notes: str = ""

    #: True when the attack turned an attack flow into an apparently benign one.
    evaded_to_benign: bool = False

    def summary(self) -> str:
        if not self.success:
            return (
                f"no evasion within budget "
                f"({self.queries_used} queries, still {self.original_prediction})"
            )
        return (
            f"{self.original_prediction} -> {self.adversarial_prediction}"
            f"{' (BENIGN)' if self.evaded_to_benign else ''}"
            f" by changing {self.n_features_changed} features "
            f"(L-inf {self.linf_norm:.3f} sd, {self.queries_used} queries)"
        )


class BaseAttack(abc.ABC):
    """Interface every Phase 2 attack implements."""

    name: str = ""

    def __init__(self, budget: Optional[AttackBudget] = None) -> None:
        self.budget = budget or AttackBudget()
        if not self.name:
            raise ValueError(f"{type(self).__name__} must define a name.")

    @abc.abstractmethod
    def attack(self, row: pd.Series, true_label: str, **kwargs: Any) -> AttackResult:
        """Attempt to make the defender misjudge ``row``."""

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} budget=({self.budget.describe()})>"


def summarise_results(
    results: Sequence[AttackResult], *, title: str = ""
) -> str:
    """Aggregate a batch of attack results into a readable report."""
    if not results:
        return f"{title}: no results"
    n = len(results)
    wins = [r for r in results if r.success]
    benign = [r for r in wins if r.evaded_to_benign]
    lines = [
        title or "attack results",
        f"  flows attacked        : {n}",
        f"  evasions              : {len(wins)} ({len(wins)/n:.1%})",
        f"  evaded to Benign      : {len(benign)} ({len(benign)/n:.1%})",
    ]
    if wins:
        lines += [
            f"  median features moved : "
            f"{int(np.median([r.n_features_changed for r in wins]))}",
            f"  median L-inf (sd)     : "
            f"{float(np.median([r.linf_norm for r in wins])):.3f}",
            f"  median queries        : "
            f"{int(np.median([r.queries_used for r in wins]))}",
        ]
    return "\n".join(lines)


__all__ = ["AttackBudget", "AttackResult", "BaseAttack", "summarise_results"]
