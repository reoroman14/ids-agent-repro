"""
Feature-space evasion: perturb a flow so the detector misjudges it.

Two attacks, both black-box — they use only the detector's output scores, never
its internals, which is the realistic position for someone probing a deployed
IDS.

``RandomNoiseEvasion``
    Random perturbations within the budget. The baseline that any real attack
    must beat: if random noise evades as often as a directed search, the
    detector is simply unstable and the "attack" is not doing any work.

``GreedyEvasion``
    Coordinate-wise search. Repeatedly tries the single feature change that
    most reduces confidence in the true class. Cheap, query-limited, and close
    to what an attacker with probe access would actually do.

Both respect ``manipulation_cost`` from the feature schema, so they only touch
features an attacker could change while still mounting the attack. Evasions
found by rewriting the transport protocol are excluded by construction, since
those describe a different flow rather than a disguised one.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..data import feature_schema as fs
from .base_attack import AttackBudget, AttackResult, BaseAttack

BENIGN = "Benign"

#: A scorer maps a batch of rows to (n_rows, n_classes) scores.
Scorer = Callable[[pd.DataFrame], np.ndarray]


class _FeatureSpaceAttack(BaseAttack):
    """Shared machinery: which columns may move, and how results are scored."""

    def __init__(
        self,
        scorer: Scorer,
        class_names: Sequence[str],
        feature_names: Sequence[str],
        *,
        dataset: str,
        budget: Optional[AttackBudget] = None,
        seed: int = 42,
    ) -> None:
        super().__init__(budget)
        self.scorer = scorer
        self.class_names = [str(c) for c in class_names]
        self.feature_names = list(feature_names)
        self.dataset = dataset
        self.rng = np.random.default_rng(seed)

        allowed = set(
            fs.attacker_controllable_columns(dataset, max_cost=self.budget.max_cost)
        )
        self.mutable = [c for c in self.feature_names if c in allowed]
        if not self.mutable:
            raise ValueError(
                f"no mutable features for {dataset} at max_cost="
                f"{self.budget.max_cost!r}; the attack has nothing to change."
            )

    # -- helpers -----------------------------------------------------------

    def _predict(self, rows: pd.DataFrame) -> List[str]:
        scores = np.asarray(self.scorer(rows))
        return [self.class_names[int(i)] for i in scores.argmax(axis=1)]

    def _true_class_score(self, rows: pd.DataFrame, true_index: int) -> np.ndarray:
        return np.asarray(self.scorer(rows))[:, true_index]

    def _result(
        self,
        row: pd.Series,
        adversarial: pd.Series,
        true_label: str,
        original: str,
        queries: int,
        notes: str = "",
    ) -> AttackResult:
        delta = {
            c: float(adversarial[c] - row[c])
            for c in self.mutable
            if abs(float(adversarial[c] - row[c])) > 1e-12
        }
        prediction = self._predict(adversarial.to_frame().T)[0]
        values = np.array(list(delta.values())) if delta else np.array([0.0])
        return AttackResult(
            success=prediction != original,
            original_prediction=original,
            adversarial_prediction=prediction,
            true_label=true_label,
            adversarial_row=adversarial,
            perturbation=delta,
            n_features_changed=len(delta),
            l2_norm=float(np.linalg.norm(values)),
            linf_norm=float(np.abs(values).max()),
            queries_used=queries,
            evaded_to_benign=(prediction == BENIGN and original != BENIGN),
            notes=notes,
        )


class RandomNoiseEvasion(_FeatureSpaceAttack):
    """
    Random perturbation within the budget — the control condition.

    Included because an evasion rate is uninterpretable on its own. If random
    noise flips 30% of predictions, a directed attack achieving 35% has
    demonstrated almost nothing about the attacker, and a great deal about the
    detector's stability.
    """

    name = "random_noise"

    def attack(self, row: pd.Series, true_label: str, **kwargs: Any) -> AttackResult:
        original = self._predict(row.to_frame().T)[0]
        k = min(self.budget.max_features or len(self.mutable), len(self.mutable))

        best: Optional[pd.Series] = None
        for query in range(1, self.budget.max_queries + 1):
            candidate = row.copy()
            columns = self.rng.choice(self.mutable, size=k, replace=False)
            noise = self.rng.uniform(
                -self.budget.epsilon, self.budget.epsilon, size=k
            )
            for column, value in zip(columns, noise):
                candidate[column] = float(candidate[column]) + float(value)
            if self._predict(candidate.to_frame().T)[0] != original:
                return self._result(row, candidate, true_label, original, query)
            best = candidate
        return self._result(
            row, best if best is not None else row.copy(), true_label, original,
            self.budget.max_queries, notes="budget exhausted",
        )


class GreedyEvasion(_FeatureSpaceAttack):
    """
    Greedy coordinate descent on the true class's score.

    Each round, every mutable feature is tried at plus and minus a step, and
    the single change that most reduces confidence in the true class is kept.
    Continues until the prediction flips or the query budget runs out.

    This is a weak attack by adversarial-ML standards — no gradients, no
    surrogate model — and that is deliberate. The question here is whether a
    modestly resourced attacker can evade the detector, not whether an
    unlimited one can. An unlimited one always can.
    """

    name = "greedy"

    def __init__(self, *args: Any, n_steps: int = 4, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.n_steps = n_steps

    def attack(self, row: pd.Series, true_label: str, **kwargs: Any) -> AttackResult:
        original = self._predict(row.to_frame().T)[0]
        if true_label not in self.class_names:
            return AttackResult(
                success=False, original_prediction=original, true_label=true_label,
                notes="true label not in the class list",
            )
        true_index = self.class_names.index(true_label)

        current = row.copy()
        moved: Dict[str, float] = {}
        queries = 0
        step = self.budget.epsilon / self.n_steps
        max_features = self.budget.max_features or len(self.mutable)

        while queries < self.budget.max_queries:
            candidates, meta = [], []
            for column in self.mutable:
                if column not in moved and len(moved) >= max_features:
                    continue
                for direction in (+1.0, -1.0):
                    shift = moved.get(column, 0.0) + direction * step
                    if abs(shift) > self.budget.epsilon + 1e-12:
                        continue
                    candidate = current.copy()
                    candidate[column] = float(row[column]) + shift
                    candidates.append(candidate)
                    meta.append((column, shift))
            if not candidates:
                break

            batch = pd.DataFrame(candidates)
            remaining = self.budget.max_queries - queries
            if len(batch) > remaining:
                batch, meta = batch.iloc[:remaining], meta[:remaining]
            queries += len(batch)

            scores = np.asarray(self.scorer(batch))
            predictions = scores.argmax(axis=1)

            flipped = np.flatnonzero(predictions != self.class_names.index(original))
            if flipped.size:
                # Prefer a flip that lands on Benign — the attacker's real goal.
                benign_index = (
                    self.class_names.index(BENIGN)
                    if BENIGN in self.class_names else -1
                )
                to_benign = [i for i in flipped if predictions[i] == benign_index]
                chosen = int(to_benign[0] if to_benign else flipped[0])
                return self._result(
                    row, batch.iloc[chosen], true_label, original, queries
                )

            best = int(scores[:, true_index].argmin())
            if scores[best, true_index] >= self._true_class_score(
                current.to_frame().T, true_index
            )[0]:
                break  # no single move helps any more
            column, shift = meta[best]
            moved[column] = shift
            current = batch.iloc[best].copy()

        return self._result(
            row, current, true_label, original, queries, notes="budget exhausted"
        )


def build_scorer(model: Any) -> Scorer:
    """Wrap a fitted classifier as a scorer over the frozen class list."""
    from ..classifiers import registry

    model_classes = np.asarray(getattr(model, "classes_", []))

    def scorer(rows: pd.DataFrame) -> np.ndarray:
        scores = registry.predict_scores(model, rows)
        if model_classes.size == 0:
            return scores
        # Re-index onto the full class list, so a model that never saw a class
        # scores it zero rather than shifting every column left.
        full = np.zeros((len(rows), int(model_classes.max()) + 1))
        for position, class_index in enumerate(model_classes):
            full[:, int(class_index)] = scores[:, position]
        return full

    return scorer


__all__ = ["GreedyEvasion", "RandomNoiseEvasion", "build_scorer"]
