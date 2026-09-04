"""
Generates a side-by-side comparison report against the original
IDS-Agent paper's reported numbers:
  ACI-IoT'23  F1 ~ 0.97
  CIC-IoT'23  F1 ~ 0.75
  Zero-day recall ~ 0.61

One caveat governs the whole module: **the paper states an F1 but not which
average it is**, and on this data that ambiguity is worth up to 0.17. A
train-only rare class contributes a structural zero to a macro average, so on
ACI the same Random Forest scores 0.995 weighted and 0.827 macro. A comparison
that silently picks one of those can claim either a clear reproduction or a
clear failure from identical predictions.

The report therefore always prints every variant next to the headline
comparison, and labels the compared metric explicitly as an assumption rather
than a fact. ``PAPER_BASELINE`` records that the averaging method is inferred;
if the paper is checked and the answer is known, set ``averaging_confirmed``
and the caveat disappears from the output.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

#: Figures reported by the original paper, with what we do and do not know.
PAPER_BASELINE: Dict[str, Any] = {
    "source": (
        "Li, Xiang, Bastian, Song, Bo Li — IDS-Agent, NeurIPS 2024 Workshop on "
        "Open-World Agents"
    ),
    "averaging_confirmed": False,
    "targets": {
        "aci_iot_2023": {"metric": "f1", "value": 0.97},
        "cic_iot_2023": {"metric": "f1", "value": 0.75},
    },
    "zero_day_recall": 0.61,
}

#: Which of our metrics stands in for the paper's unqualified "F1".
DEFAULT_COMPARISON_METRIC = "weighted_f1"

#: Printed alongside it, so the choice above is always visible as a choice.
METRIC_VARIANTS = ("weighted_f1", "macro_f1", "macro_f1_present_only", "accuracy")

#: How close counts as reproducing the figure.
DEFAULT_TOLERANCE = 0.02

ResultsInput = Union[pd.DataFrame, Dict[str, Any]]


@dataclass
class Comparison:
    """One benchmark, ours against the paper's."""

    dataset: str
    metric: str
    paper_value: Optional[float]
    our_value: Optional[float]
    our_classifier: Optional[str] = None
    variants: Dict[str, float] = field(default_factory=dict)
    note: str = ""

    @property
    def delta(self) -> Optional[float]:
        if self.paper_value is None or self.our_value is None:
            return None
        return self.our_value - self.paper_value

    def verdict(self, tolerance: float = DEFAULT_TOLERANCE) -> str:
        d = self.delta
        if d is None:
            return "not run"
        if abs(d) <= tolerance:
            return "matched"
        return "above paper" if d > 0 else "below paper"


# ---------------------------------------------------------------------------
# Normalizing the input
# ---------------------------------------------------------------------------

def _as_table(value: ResultsInput) -> pd.DataFrame:
    """Accept a results table, or a dict carrying one under 'table'."""
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, dict):
        if isinstance(value.get("table"), pd.DataFrame):
            return value["table"]
        return pd.DataFrame(value.get("rows", value))
    raise TypeError(
        f"Expected a results DataFrame from run_all() or a dict containing one, "
        f"got {type(value).__name__}."
    )


def _successful(table: pd.DataFrame) -> pd.DataFrame:
    """Rows for classifiers that actually produced scores."""
    if "error" in table.columns:
        table = table[table["error"].isna()]
    return table


def _best_row(table: pd.DataFrame, metric: str) -> Optional[pd.Series]:
    ok = _successful(table)
    if ok.empty or metric not in ok.columns or ok[metric].isna().all():
        return None
    return ok.loc[ok[metric].idxmax()]


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_to_paper(
    results: Dict[str, ResultsInput],
    *,
    metric: str = DEFAULT_COMPARISON_METRIC,
    baseline: Optional[Dict[str, Any]] = None,
) -> List[Comparison]:
    """
    Build one :class:`Comparison` per dataset, plus one for zero-day recall.

    The best classifier is chosen by ``metric`` — the same metric being
    compared — so the report never quietly picks a winner on one measure and
    reports it on another.
    """
    baseline = baseline or PAPER_BASELINE
    out: List[Comparison] = []

    for dataset, table in results.items():
        table = _as_table(table)
        target = baseline["targets"].get(dataset, {})
        best = _best_row(table, metric)

        variants = {}
        if best is not None:
            variants = {
                v: float(best[v])
                for v in METRIC_VARIANTS
                if v in best.index and pd.notna(best[v])
            }

        out.append(
            Comparison(
                dataset=dataset,
                metric=metric,
                paper_value=target.get("value"),
                our_value=float(best[metric]) if best is not None else None,
                our_classifier=str(best["classifier"]) if best is not None else None,
                variants=variants,
                note="" if target else "no paper figure for this dataset",
            )
        )

    zero_day = _zero_day_comparison(results, baseline)
    if zero_day is not None:
        out.append(zero_day)
    return out


def _zero_day_comparison(
    results: Dict[str, ResultsInput], baseline: Dict[str, Any]
) -> Optional[Comparison]:
    """Best zero-day recall across every dataset that ran a holdout."""
    best_value, best_label = None, None
    for dataset, table in results.items():
        table = _successful(_as_table(table))
        if "zero_day_recall" not in table.columns or table["zero_day_recall"].isna().all():
            continue
        idx = table["zero_day_recall"].idxmax()
        value = float(table.loc[idx, "zero_day_recall"])
        if best_value is None or value > best_value:
            best_value = value
            best_label = f"{table.loc[idx, 'classifier']} on {dataset}"

    target = baseline.get("zero_day_recall")
    if best_value is None:
        return Comparison(
            dataset="zero-day",
            metric="zero_day_recall",
            paper_value=target,
            our_value=None,
            note="no run had `holdout_classes` configured",
        )
    return Comparison(
        dataset="zero-day",
        metric="zero_day_recall",
        paper_value=target,
        our_value=best_value,
        our_classifier=best_label,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _fmt(value: Optional[float], places: int = 4) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:.{places}f}"


def _markdown_table(rows: Sequence[Sequence[Any]], header: Sequence[str]) -> str:
    lines = [
        "| " + " | ".join(str(h) for h in header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def _cell(value: Any) -> str:
    """Render one DataFrame cell: floats to 4dp, NaN to an em dash."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _dataframe_markdown(df: pd.DataFrame) -> str:
    """
    Render a DataFrame as a Markdown table.

    Hand-rolled rather than ``DataFrame.to_markdown``, which requires the
    optional ``tabulate`` package — not worth a new dependency for one table.
    """
    return _markdown_table(
        [[_cell(v) for v in row] for row in df.itertuples(index=False)],
        list(df.columns),
    )


def _caveats(results: Dict[str, ResultsInput], baseline: Dict[str, Any]) -> List[str]:
    """Everything a reader needs before believing the headline table."""
    notes: List[str] = []

    if not baseline.get("averaging_confirmed", False):
        notes.append(
            "**The paper's averaging method is inferred, not confirmed.** It "
            "reports an F1 without saying whether it is macro, weighted or "
            "micro. On this data the gap between them reaches 0.17, so the "
            "comparison metric below is an assumption. Check the paper before "
            "claiming the baseline is reproduced."
        )

    subsampled: List[str] = []
    for dataset, table in results.items():
        table = _successful(_as_table(table))
        if "subsampled" not in table.columns:
            continue
        for _, row in table[table["subsampled"] == True].iterrows():  # noqa: E712
            subsampled.append(
                f"`{row['classifier']}` on {dataset} "
                f"({int(row['n_train_used']):,} rows)"
            )
    if subsampled:
        notes.append(
            "**Some classifiers trained on a capped subset**, so their scores "
            "are not directly comparable with the rest: "
            + ", ".join(subsampled)
            + ". Kernel SVC is quadratic-to-cubic in training rows and KNN pays "
            "`n_train x n_test` at predict time; neither finishes on the full "
            "split. See `training.max_train_rows` in config/models.yaml."
        )

    for dataset, table in results.items():
        raw = table if isinstance(table, dict) else {}
        absent = raw.get("classes_absent_from_test")
        if absent:
            notes.append(
                f"**{dataset}: classes absent from the test split** — "
                + ", ".join(f"`{c}`" for c in absent)
                + ". They are trained on but never scored, which is why macro "
                "and weighted F1 diverge."
            )

    failed: List[str] = []
    for dataset, table in results.items():
        table = _as_table(table)
        if "error" in table.columns:
            for _, row in table[table["error"].notna()].iterrows():
                failed.append(f"`{row['classifier']}` on {dataset}: {row['error']}")
    if failed:
        notes.append("**Classifiers that failed to run:** " + "; ".join(failed) + ".")

    return notes


def generate_report(
    results: Dict[str, ResultsInput],
    output_path: Optional[str] = None,
    *,
    metric: str = DEFAULT_COMPARISON_METRIC,
    baseline: Optional[Dict[str, Any]] = None,
    tolerance: float = DEFAULT_TOLERANCE,
    title: str = "IDS-Agent Phase 1 — Baseline vs. Paper",
) -> str:
    """
    Render the comparison as Markdown, and write it to ``output_path`` if given.

    ``results`` maps a dataset name to the table returned by
    :func:`src.classifiers.train_eval.run_all`. Returns the Markdown text.
    """
    baseline = baseline or PAPER_BASELINE
    comparisons = compare_to_paper(results, metric=metric, baseline=baseline)

    parts: List[str] = [
        f"# {title}",
        "",
        f"Generated {date.today().isoformat()}. Paper figures from "
        f"{baseline['source']}.",
        "",
        "## Headline comparison",
        "",
        f"Comparing our **{metric}** against the paper's reported F1. "
        f"Within {tolerance:g} counts as matched.",
        "",
    ]

    rows = []
    for c in comparisons:
        rows.append(
            [
                c.dataset,
                _fmt(c.paper_value, 2),
                _fmt(c.our_value),
                c.our_classifier or "—",
                _fmt(c.delta, 4) if c.delta is not None else "—",
                c.verdict(tolerance) + (f" ({c.note})" if c.note else ""),
            ]
        )
    parts.append(
        _markdown_table(
            rows, ["Benchmark", "Paper", "Ours", "Best classifier", "Δ", "Verdict"]
        )
    )
    parts.append("")

    # The variant table is the point of this report, not an appendix.
    variant_rows = []
    for c in comparisons:
        if not c.variants:
            continue
        variant_rows.append(
            [c.dataset, c.our_classifier or "—"]
            + [_fmt(c.variants.get(v)) for v in METRIC_VARIANTS]
        )
    if variant_rows:
        parts += [
            "## The same predictions, scored four ways",
            "",
            "Which average is used decides whether the baseline looks "
            "reproduced. These are the same models as above.",
            "",
            _markdown_table(
                variant_rows,
                ["Dataset", "Classifier"] + [f"`{v}`" for v in METRIC_VARIANTS],
            ),
            "",
        ]

    for dataset, table in results.items():
        table = _as_table(table)
        parts += [f"## {dataset} — all classifiers", ""]
        display = [c for c in table.columns if c not in ("details",)]
        parts += [_dataframe_markdown(table[display]), ""]

    notes = _caveats(results, baseline)
    if notes:
        parts += ["## Caveats", ""]
        parts += [f"{i}. {n}" for i, n in enumerate(notes, 1)]
        parts.append("")

    text = "\n".join(parts)

    if output_path:
        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(text)

    return text


def summarize_comparisons(
    comparisons: Sequence[Comparison], tolerance: float = DEFAULT_TOLERANCE
) -> str:
    """One-line-per-benchmark rendering, for terminal output."""
    lines = []
    for c in comparisons:
        lines.append(
            f"  {c.dataset:16} paper {_fmt(c.paper_value, 2):>6}  "
            f"ours {_fmt(c.our_value):>6}  {c.verdict(tolerance)}"
            + (f"  [{c.our_classifier}]" if c.our_classifier else "")
        )
    return "\n".join(lines)


__all__ = [
    "Comparison",
    "DEFAULT_COMPARISON_METRIC",
    "DEFAULT_TOLERANCE",
    "METRIC_VARIANTS",
    "PAPER_BASELINE",
    "compare_to_paper",
    "generate_report",
    "summarize_comparisons",
]
