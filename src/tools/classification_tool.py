"""
Wraps src/classifiers/registry.py as an agent tool: given flow
features, returns a classification (benign/attack + category) from
a chosen classifier.

The tool holds pre-fitted models. Fitting on demand inside an agent step would
make every run's cost depend on the agent's choices and would let a prompt
influence training — so models are fitted once, outside the loop, and the tool
only ever calls predict.

Its output carries ``Provenance.MODEL``: the numbers are produced by our own
classifiers. The *features* fed in are attacker-influenceable, which is exactly
what Phase 2 manipulates, but the computation over them is not.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..classifiers import registry
from .base_tool import BaseTool, Provenance, ToolResult

DEFAULT_TOP_K = 3


class ClassificationTool(BaseTool):
    """
    Run one of the fitted classifiers over the flow currently under analysis.

    Parameters
    ----------
    models
        Mapping of classifier name to a fitted estimator.
    feature_names
        Column order the models were fitted on. Inputs are reindexed to this
        order, because a silently reordered feature vector produces confident
        nonsense rather than an error.
    class_names
        Frozen class list, so predicted indices become names consistently.
    accuracy_hints
        Optional per-classifier headline score from the baseline run. Included
        in the rendered text so the model can weigh a confident opinion from a
        weak classifier appropriately — the system prompt asks it to do that,
        and without this it has no basis to.
    """

    name = "classify_flow"
    description = (
        "Classify the flow under analysis with one of the trained intrusion "
        "detection models. Returns the predicted class and the most likely "
        "alternatives with their scores. Call it with different classifiers to "
        "compare their opinions."
    )
    provenance = Provenance.MODEL

    def __init__(
        self,
        models: Mapping[str, Any],
        feature_names: Sequence[str],
        class_names: Sequence[str],
        *,
        accuracy_hints: Optional[Mapping[str, float]] = None,
        default_classifier: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        if not models:
            raise ValueError("ClassificationTool needs at least one fitted model.")
        self.models = dict(models)
        self.feature_names = list(feature_names)
        self.class_names = [str(c) for c in class_names]
        self.accuracy_hints = dict(accuracy_hints or {})
        self.top_k = top_k
        self.default_classifier = default_classifier or next(iter(self.models))

        available = sorted(self.models)
        self.input_schema = {
            "type": "object",
            "properties": {
                "classifier": {
                    "type": "string",
                    "enum": available,
                    "description": (
                        f"Which trained model to use. Defaults to "
                        f"{self.default_classifier}."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": "How many candidate classes to return.",
                },
            },
            "required": [],
        }
        self.description = f"{self.description} Available classifiers: {available}."
        super().__init__()
        self._current: Optional[pd.DataFrame] = None

    # -- flow under analysis ----------------------------------------------

    def set_flow(self, record: Mapping[str, Any]) -> None:
        """
        Set the flow this tool will classify.

        The loop sets this once per record; the model chooses only *which*
        classifier to run, never *what data* to run it on. That keeps the
        prompt from being able to redirect the tool onto other traffic.
        """
        row = {name: record.get(name, np.nan) for name in self.feature_names}
        self._current = pd.DataFrame([row], columns=self.feature_names)

    def run(self, **kwargs: Any) -> ToolResult:
        if self._current is None:
            return ToolResult.failure(
                self.name,
                "no flow is loaded; the agent loop must call set_flow() first",
                provenance=self.provenance,
            )

        classifier = kwargs.get("classifier") or self.default_classifier
        canonical = str(classifier).strip().lower()
        if canonical not in self.models:
            try:
                canonical = registry.resolve_name(canonical)
            except KeyError:
                canonical = ""
        if canonical not in self.models:
            return ToolResult.failure(
                self.name,
                f"unknown classifier {classifier!r}; available: {sorted(self.models)}",
                provenance=self.provenance,
            )

        top_k = int(kwargs.get("top_k") or self.top_k)
        top_k = max(1, min(top_k, len(self.class_names)))

        model = self.models[canonical]
        scores = registry.predict_scores(model, self._current)[0]
        kind = registry.score_kind(model)

        order = np.argsort(scores)[::-1][:top_k]
        model_classes = getattr(model, "classes_", np.arange(len(self.class_names)))
        candidates: List[Dict[str, Any]] = []
        for rank, position in enumerate(order, 1):
            class_index = int(model_classes[position])
            candidates.append(
                {
                    "rank": rank,
                    "class": self.class_names[class_index],
                    "score": float(scores[position]),
                }
            )

        top = candidates[0]
        data = {
            "classifier": canonical,
            "prediction": top["class"],
            "is_attack": top["class"] != "Benign",
            "score_kind": kind,
            "candidates": candidates,
            "classifier_accuracy": self.accuracy_hints.get(canonical),
        }
        return ToolResult(
            tool=self.name,
            ok=True,
            data=data,
            rendered=self._render(data),
            provenance=self.provenance,
        )

    def _render(self, data: Dict[str, Any]) -> str:
        lines = [
            f"classifier: {data['classifier']}"
            + (
                f" (baseline weighted F1 {data['classifier_accuracy']:.4f})"
                if data.get("classifier_accuracy") is not None
                else ""
            ),
            f"prediction: {data['prediction']}"
            f" ({'attack' if data['is_attack'] else 'benign'})",
            "candidates:",
        ]
        for c in data["candidates"]:
            lines.append(f"  {c['rank']}. {c['class']}  score {c['score']:.4f}")
        if data["score_kind"] != "proba":
            lines.append(
                "note: this model exposes no calibrated probability; scores are a "
                "normalised decision margin and rank classes but are not "
                "probabilities."
            )
        return "\n".join(lines)


__all__ = ["ClassificationTool"]
