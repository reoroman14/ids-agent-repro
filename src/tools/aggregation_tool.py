"""
Aggregates results across classifiers/tools into a single
explainable verdict, matching IDS-Agent's results-aggregation step.

The tool runs every available classifier on the current flow and combines
them, rather than reading back whatever the agent happened to mention in its
reasoning. Two reasons: the aggregate is then deterministic and reproducible
given the flow, and it cannot be steered by a prompt that misreports an earlier
observation — which matters once Phase 2 starts manipulating what the agent
believes it saw.

"Explainable" is taken literally. The rendered output states each classifier's
vote, how much they agreed, and where they disagreed, because an aggregate that
hides a three-two split behind a single label is worse than no aggregate.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..classifiers import registry
from .base_tool import BaseTool, Provenance, ToolResult

METHODS = ("weighted_mean", "mean_score", "majority")
DEFAULT_METHOD = "weighted_mean"
DEFAULT_TOP_K = 3


class AggregationTool(BaseTool):
    """
    Combine every classifier's opinion of the current flow into one verdict.

    Parameters
    ----------
    models
        Mapping of classifier name to fitted estimator.
    feature_names, class_names
        As for :class:`~src.tools.classification_tool.ClassificationTool`.
    accuracy_hints
        Per-classifier headline score, used as the weight for
        ``weighted_mean``. Without them that method falls back to equal
        weights, which is ``mean_score``.
    """

    name = "aggregate_classifiers"
    description = (
        "Run every trained model on this flow and combine their opinions into "
        "one verdict, reporting how much they agreed and where they differed. "
        "Use this to settle a disagreement between individual classifiers."
    )
    provenance = Provenance.MODEL

    def __init__(
        self,
        models: Mapping[str, Any],
        feature_names: Sequence[str],
        class_names: Sequence[str],
        *,
        accuracy_hints: Optional[Mapping[str, float]] = None,
        method: str = DEFAULT_METHOD,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        if not models:
            raise ValueError("AggregationTool needs at least one fitted model.")
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
        self.models = dict(models)
        self.feature_names = list(feature_names)
        self.class_names = [str(c) for c in class_names]
        self.accuracy_hints = dict(accuracy_hints or {})
        self.method = method
        self.top_k = top_k

        self.input_schema = {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "enum": list(METHODS),
                    "description": (
                        "How to combine: weighted_mean (by each model's known "
                        "accuracy), mean_score (equal weight), or majority "
                        "(one vote each)."
                    ),
                }
            },
            "required": [],
        }
        super().__init__()
        self._current: Optional[pd.DataFrame] = None

    def set_flow(self, record: Mapping[str, Any]) -> None:
        row = {name: record.get(name, np.nan) for name in self.feature_names}
        self._current = pd.DataFrame([row], columns=self.feature_names)

    def run(self, **kwargs: Any) -> ToolResult:
        if self._current is None:
            return ToolResult.failure(
                self.name,
                "no flow is loaded; the agent loop must call set_flow() first",
                provenance=self.provenance,
            )
        method = str(kwargs.get("method") or self.method)
        if method not in METHODS:
            return ToolResult.failure(
                self.name,
                f"unknown method {method!r}; expected one of {list(METHODS)}",
                provenance=self.provenance,
            )

        per_model, totals, weights = [], np.zeros(len(self.class_names)), 0.0
        for name, model in self.models.items():
            try:
                scores = registry.predict_scores(model, self._current)[0]
            except Exception as exc:  # noqa: BLE001 - one bad model must not sink it
                per_model.append({"classifier": name, "error": str(exc)})
                continue

            # Score columns follow the classes the model was fitted on, which is
            # not the frozen list whenever a class was withheld or unseen.
            aligned = np.zeros(len(self.class_names))
            model_classes = getattr(model, "classes_", np.arange(len(scores)))
            for position, class_index in enumerate(model_classes):
                if 0 <= int(class_index) < len(aligned):
                    aligned[int(class_index)] = scores[position]

            weight = float(self.accuracy_hints.get(name, 1.0)) if method == "weighted_mean" else 1.0
            top = int(np.argmax(aligned))
            per_model.append(
                {
                    "classifier": name,
                    "prediction": self.class_names[top],
                    "confidence": float(aligned[top]),
                    "weight": weight,
                    "score_kind": registry.score_kind(model),
                }
            )
            if method == "majority":
                totals[top] += weight
            else:
                totals += weight * aligned
            weights += weight

        votes = [m for m in per_model if "prediction" in m]
        if not votes:
            return ToolResult.failure(
                self.name, "every classifier failed on this flow",
                provenance=self.provenance,
            )

        combined = totals / weights if weights else totals
        order = np.argsort(combined)[::-1][: self.top_k]
        candidates = [
            {"rank": rank, "class": self.class_names[int(i)], "score": float(combined[i])}
            for rank, i in enumerate(order, 1)
        ]

        counts = Counter(m["prediction"] for m in votes)
        winner, winner_votes = counts.most_common(1)[0]
        dissent = sorted(
            {m["prediction"] for m in votes} - {candidates[0]["class"]}
        )

        data = {
            "method": method,
            "verdict": candidates[0]["class"],
            "is_attack": candidates[0]["class"] != "Benign",
            "agreement": winner_votes / len(votes),
            "n_classifiers": len(votes),
            "majority_class": winner,
            "unanimous": len(counts) == 1,
            "dissenting_classes": dissent,
            "candidates": candidates,
            "per_classifier": per_model,
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
            f"aggregate verdict: {data['verdict']} "
            f"({'attack' if data['is_attack'] else 'benign'}) "
            f"by {data['method']}",
            f"agreement: {data['agreement']:.0%} of {data['n_classifiers']} "
            f"classifiers voted {data['majority_class']}"
            + ("  [unanimous]" if data["unanimous"] else ""),
            "individual votes:",
        ]
        for entry in data["per_classifier"]:
            if "error" in entry:
                lines.append(f"  {entry['classifier']}: failed ({entry['error']})")
                continue
            lines.append(
                f"  {entry['classifier']}: {entry['prediction']} "
                f"(confidence {entry['confidence']:.3f}, weight {entry['weight']:.3f})"
            )
        if data["dissenting_classes"]:
            lines.append(
                "disagreement: also proposed "
                + ", ".join(data["dissenting_classes"])
                + " — treat this verdict as contested."
            )
        lines.append("combined ranking:")
        for c in data["candidates"]:
            lines.append(f"  {c['rank']}. {c['class']}  {c['score']:.4f}")
        return "\n".join(lines)


__all__ = ["METHODS", "AggregationTool"]
