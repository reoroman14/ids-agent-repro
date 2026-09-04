"""
Manual ReAct-style reasoning-action loop (no LangChain), giving full
control over what enters the LLM context at each step — necessary
groundwork for Phase 2/3 telemetry-injection experiments.

The loop never formats tool output itself. Every observation reaches the
context through ``prompts.render_tool_result``, which uses a result's
``rendered`` text and fences it when its provenance is attacker-influenceable.
That is the boundary the module docstring in ``base_tool.py`` describes, and
this file is deliberately written so there is no second way across it.

Each run produces a :class:`Trace` holding the full message transcript, every
tool call, and the fraction of context characters that were attacker
influenceable. Phase 2 measures its attacks against exactly those fields.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..tools.base_tool import BaseTool, ToolRegistry, ToolResult
from . import prompts
from .llm_client import LLMClient, LLMError

DEFAULT_MAX_STEPS = 6

#: Values models emit in an ACTION line when they mean "no tool, I am done".
#: Observed from llama3: "None", "none", "None (final answer)". Treating these
#: as a tool name burns a step calling a tool that does not exist, and on a
#: measured run that cost 13 of 68 flows a step and 8 accuracy points.
_NULL_ACTIONS = frozenset({"none", "null", "n/a", "na", "no action", "nothing",
                           "no tool", "-", "final answer", "finish", "done"})

_ACTION_RE = re.compile(r"^\s*ACTION\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_ACTION_INPUT_RE = re.compile(
    r"^\s*ACTION_INPUT\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE
)
_THOUGHT_RE = re.compile(r"^\s*THOUGHT\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_VERDICT_RE = re.compile(r"^\s*VERDICT\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_CONFIDENCE_RE = re.compile(
    r"^\s*CONFIDENCE\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE
)
_REASONING_RE = re.compile(
    r"^\s*REASONING\s*:\s*(.+?)\s*\Z", re.IGNORECASE | re.MULTILINE | re.DOTALL
)


@dataclass
class Step:
    """One reasoning/action cycle."""

    index: int
    raw_response: str = ""
    thought: str = ""
    action: str = ""
    action_input: Dict[str, Any] = field(default_factory=dict)
    result: Optional[ToolResult] = None
    observation: str = ""
    parse_error: str = ""

    @property
    def used_tool(self) -> bool:
        return bool(self.action)


@dataclass
class Trace:
    """Everything a run did, for scoring and for Phase 2 attribution."""

    verdict: str = ""
    #: Exactly what the model emitted, before canonicalisation.
    verdict_raw: str = ""
    confidence: str = ""
    reasoning: str = ""
    ok: bool = False
    error: str = ""
    steps: List[Step] = field(default_factory=list)
    messages: List[Dict[str, str]] = field(default_factory=list)
    verdict_is_known_class: bool = True

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def tools_called(self) -> List[str]:
        return [s.action for s in self.steps if s.used_tool]

    @property
    def context_chars(self) -> int:
        return sum(len(m.get("content", "")) for m in self.messages)

    @property
    def untrusted_chars(self) -> int:
        """Characters of context that came from attacker-influenceable sources."""
        return sum(
            len(span)
            for m in self.messages
            for span in prompts.untrusted_spans(m.get("content", ""))
        )

    @property
    def untrusted_fraction(self) -> float:
        """
        Share of the context an attacker could influence.

        The headline exposure number for Phase 2: an injection can only work
        through this fraction, and watching it move as the threat model changes
        is more informative than watching the verdict alone.
        """
        total = self.context_chars
        return (self.untrusted_chars / total) if total else 0.0

    def summary(self) -> str:
        lines = [
            f"verdict     : {self.verdict or '(none)'}"
            + ("" if self.verdict_is_known_class else "  [NOT A KNOWN CLASS]"),
            f"confidence  : {self.confidence or '(none)'}",
            f"steps       : {self.n_steps}",
            f"tools used  : {', '.join(self.tools_called) or '(none)'}",
            f"context     : {self.context_chars:,} chars, "
            f"{self.untrusted_fraction:.1%} attacker-influenceable",
        ]
        if not self.ok:
            lines.append(f"error       : {self.error}")
        return "\n".join(lines)


class ReactAgent:
    """
    Reason-act loop over the classification, retrieval and aggregation tools.

    Parameters
    ----------
    llm_client
        Any :class:`~src.agent.llm_client.LLMClient`.
    tools
        A :class:`ToolRegistry`, or a list of tools to wrap in one.
    memory
        Optional object exposing ``recall(record) -> str``. Its output is
        treated as untrusted, because Phase 3 poisons precisely this channel.
    class_names
        Frozen class list. Used to tell the model its options and to flag a
        verdict that is not one of them.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tools: Any,
        *,
        memory: Any = None,
        class_names: Optional[Sequence[str]] = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_features_shown: Optional[int] = 20,
    ) -> None:
        self.llm_client = llm_client
        self.tools = tools if isinstance(tools, ToolRegistry) else ToolRegistry(list(tools))
        self.memory = memory
        self.class_names = [str(c) for c in (class_names or [])]
        self.max_steps = max_steps
        self.max_features_shown = max_features_shown

    # -- main entry point --------------------------------------------------

    def run(
        self, input_record: Mapping[str, Any], context: Optional[str] = None
    ) -> Trace:
        """
        Analyse one flow and return a verdict plus the full reasoning trace.

        ``context`` is an optional human-readable description of the same flow
        (see :func:`src.data.feature_schema.describe_flow`). The tools always
        run on ``input_record``; the context only ever reaches the prompt, so
        adding it cannot change what the classifiers see and the comparison
        against a run without it stays clean. It is fenced as untrusted, being
        derived from the same observed traffic.

        Never raises for model or tool misbehaviour: a malformed reply, an
        unknown tool or a backend failure all end in a Trace with ``ok`` False
        and an explanation. A robustness study cannot have the harness fall over
        on the inputs it is meant to measure.
        """
        record = dict(input_record)
        trace = Trace()

        # Point every flow-aware tool at this record. The model chooses which
        # tool to call, never what data it runs on.
        for name in self.tools.names():
            tool = self.tools.get(name)
            if hasattr(tool, "set_flow"):
                try:
                    tool.set_flow(record)
                except Exception as exc:  # noqa: BLE001
                    return self._fail(trace, f"tool {name} rejected the record: {exc}")

        memory_context = self._recall(record)
        trace.messages = prompts.build_initial_messages(
            record,
            [self.tools.get(n) for n in self.tools.names()],
            class_names=self.class_names,
            memory_context=memory_context,
            max_features=self.max_features_shown,
            context=context,
        )

        for index in range(self.max_steps):
            try:
                response = self.llm_client.complete(
                    trace.messages, tools=self.tools.specs()
                )
            except LLMError as exc:
                return self._fail(trace, str(exc))
            except Exception as exc:  # noqa: BLE001
                return self._fail(trace, f"{type(exc).__name__}: {exc}")

            step = Step(index=index, raw_response=response.text or "")
            trace.messages.append(
                {"role": "assistant", "content": response.text or ""}
            )

            # A backend that supports native tool calling wins over text parsing.
            action, action_input = self._extract_action(response)

            if action is None:
                final = self._extract_final(response.text or "")
                if final is not None:
                    trace.steps.append(step)
                    return self._finish(trace, *final)
                step.parse_error = (
                    "response contained neither a usable ACTION nor a final VERDICT"
                )
                trace.steps.append(step)
                trace.messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your reply had neither a tool call nor a final answer.\n"
                            + prompts.REACT_STEP_INSTRUCTION
                        ),
                    }
                )
                continue

            step.thought = self._extract_thought(response.text or "")
            step.action, step.action_input = action, action_input
            result = self.tools.call(action, **action_input)
            step.result = result
            # The ONLY path from a tool result into the context.
            step.observation = prompts.render_tool_result(result)
            trace.steps.append(step)
            trace.messages.append({"role": "user", "content": step.observation})

        # Out of steps: ask once for a verdict on the evidence gathered.
        trace.messages.append(
            {
                "role": "user",
                "content": (
                    f"You have used all {self.max_steps} available steps. "
                    "Give your final answer now, in the required format, based "
                    "on the evidence gathered."
                ),
            }
        )
        try:
            response = self.llm_client.complete(trace.messages)
        except Exception as exc:  # noqa: BLE001
            return self._fail(trace, f"{type(exc).__name__}: {exc}")
        trace.messages.append({"role": "assistant", "content": response.text or ""})
        final = self._extract_final(response.text or "")
        if final is None:
            return self._fail(
                trace, f"no final verdict after {self.max_steps} steps"
            )
        return self._finish(trace, *final)

    # -- helpers -----------------------------------------------------------

    def _recall(self, record: Mapping[str, Any]) -> Optional[str]:
        if self.memory is None or not hasattr(self.memory, "recall"):
            return None
        try:
            text = self.memory.recall(record)
        except Exception:  # noqa: BLE001 - memory must not break a run
            return None
        return str(text) if text else None

    def _extract_action(self, response: Any) -> tuple:
        """Prefer the backend's structured tool call; fall back to text."""
        for call in getattr(response, "tool_calls", None) or []:
            name = call.get("name")
            if name:
                arguments = call.get("arguments") or {}
                return name, (arguments if isinstance(arguments, dict) else {})

        text = getattr(response, "text", "") or ""
        match = _ACTION_RE.search(text)
        if not match:
            return None, {}
        action = match.group(1).strip().strip("`\"'*")

        # "ACTION: None" means the model is finishing, not calling a tool named
        # None. Treat it as no action so the final answer in the same reply is
        # read instead of a step being wasted.
        bare = re.sub(r"\s*\([^)]*\)\s*$", "", action).strip().lower()
        if not bare or bare in _NULL_ACTIONS:
            return None, {}

        arguments: Dict[str, Any] = {}
        raw = _ACTION_INPUT_RE.search(text)
        if raw:
            payload = raw.group(1).strip().strip("`")
            try:
                parsed = json.loads(payload)
                if isinstance(parsed, dict):
                    arguments = parsed
            except json.JSONDecodeError:
                # A model that writes `ACTION_INPUT: random_forest` instead of
                # JSON is common enough to be worth accommodating rather than
                # burning a step on.
                if payload and payload.lower() not in ("none", "null", "{}"):
                    arguments = {"_raw": payload}
        return action, arguments

    @staticmethod
    def _extract_thought(text: str) -> str:
        match = _THOUGHT_RE.search(text)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _extract_final(text: str) -> Optional[tuple]:
        verdict = _VERDICT_RE.search(text)
        if not verdict:
            return None
        confidence = _CONFIDENCE_RE.search(text)
        reasoning = _REASONING_RE.search(text)
        # Models sometimes repeat the label ("VERDICT: VERDICT: DNS Spoofing"),
        # which otherwise becomes part of the answer. Observed once in 68 flows.
        answer = re.sub(r"^\s*VERDICT\s*:\s*", "", verdict.group(1).strip(),
                        flags=re.IGNORECASE).strip()
        return (
            answer.strip("`\"'*"),
            confidence.group(1).strip().lower() if confidence else "",
            reasoning.group(1).strip() if reasoning else "",
        )

    def _normalize_verdict(self, verdict: str) -> tuple:
        """
        Match a free-text verdict against the known class list.

        Measured necessity: on a real run, llama3 answered ``Slowloris
        (attack)`` and ``Port Scan (attack)``. Those are correct judgements,
        and scoring them as wrong measures the model's formatting rather than
        its analysis — the same mistake as reading argmax where a ranking was
        wanted.

        Canonicalisation is deliberately conservative: case-insensitive match,
        then one trailing parenthetical removed. No substring or fuzzy
        matching, which would let "not a Port Scan" be scored as "Port Scan".
        """
        if not self.class_names:
            return verdict, True
        if verdict in self.class_names:
            return verdict, True

        lowered = {c.lower(): c for c in self.class_names}
        candidate = verdict.strip()
        if candidate.lower() in lowered:
            return lowered[candidate.lower()], True

        stripped = re.sub(r"\s*\([^)]*\)\s*$", "", candidate).strip()
        if stripped and stripped.lower() in lowered:
            return lowered[stripped.lower()], True

        return verdict, False

    def _finish(
        self, trace: Trace, verdict: str, confidence: str, reasoning: str
    ) -> Trace:
        trace.verdict_raw = verdict
        canonical, known = self._normalize_verdict(verdict)
        trace.verdict = canonical
        trace.verdict_is_known_class = known
        trace.confidence = confidence
        trace.reasoning = reasoning
        trace.ok = True
        return trace

    @staticmethod
    def _fail(trace: Trace, error: str) -> Trace:
        trace.ok = False
        trace.error = error
        return trace


def batch_run(
    agent: ReactAgent, records: Sequence[Mapping[str, Any]]
) -> List[Trace]:
    """Run the agent over several flows. Failures stay in the list as traces."""
    return [agent.run(record) for record in records]


__all__ = ["DEFAULT_MAX_STEPS", "ReactAgent", "Step", "Trace", "batch_run"]
