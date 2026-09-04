"""
Common interface every agent tool must implement: name, description,
and an explicit input/output schema (used both for LLM tool-calling
and, later, for tagging which inputs are attacker-reachable).

The important structure here is the split inside :class:`ToolResult` between
``data`` and ``rendered``.

``data``      structured, machine-readable output. The loop may compute with
              it, log it, and score against it. It NEVER goes into a prompt.
``rendered``  the exact text that may be inserted into the LLM context.

Phase 2 studies telemetry injection: an attacker who controls network traffic
controls some of what a tool observes, and therefore some of what reaches the
model's context. That attack only makes sense if there is a single, identifiable
place where observed data becomes prompt text. Collapsing these two fields —
letting the loop stringify ``data`` itself — would spread that boundary across
the codebase and make the Phase 2 experiment impossible to instrument.

``provenance`` records where a result's content came from, which is what makes
the attack surface enumerable rather than guessed at.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Provenance(str, Enum):
    """Where a tool result's content originated, and hence who can influence it."""

    #: Computed by our own models or code. An attacker can influence the input
    #: features but not the computation.
    MODEL = "model"

    #: Derived from observed network traffic. Directly attacker-influenceable —
    #: this is the Phase 2 injection surface.
    TELEMETRY = "telemetry"

    #: Static reference material (attack descriptions, documentation).
    KNOWLEDGE = "knowledge"

    #: Recalled from earlier sessions. The Phase 3 poisoning surface.
    MEMORY = "memory"


#: Provenances an attacker may be able to influence. Phase 2/3 target these.
UNTRUSTED_PROVENANCE = frozenset({Provenance.TELEMETRY, Provenance.MEMORY})


@dataclass
class ToolResult:
    """
    One tool invocation's output.

    Construct with both a structured ``data`` payload and the ``rendered`` text
    that represents it. Only ``rendered`` is allowed into an LLM prompt.
    """

    tool: str
    ok: bool
    data: Dict[str, Any] = field(default_factory=dict)
    rendered: str = ""
    provenance: Provenance = Provenance.MODEL
    error: Optional[str] = None

    @property
    def is_untrusted(self) -> bool:
        """True when an attacker may have influenced this content."""
        return self.provenance in UNTRUSTED_PROVENANCE

    @classmethod
    def failure(cls, tool: str, error: str, **kwargs: Any) -> "ToolResult":
        return cls(
            tool=tool,
            ok=False,
            error=error,
            rendered=f"Tool `{tool}` failed: {error}",
            **kwargs,
        )


class BaseTool(abc.ABC):
    """
    Interface every agent tool implements.

    Subclasses set ``name``, ``description`` and ``input_schema`` (a JSON Schema
    object, so it can be handed straight to an LLM tool-calling API), and
    implement :meth:`run`.
    """

    name: str = ""
    description: str = ""
    input_schema: Dict[str, Any] = {}
    provenance: Provenance = Provenance.MODEL

    def __init__(self) -> None:
        if not self.name:
            raise ValueError(f"{type(self).__name__} must define a name.")
        if not self.description:
            raise ValueError(f"{type(self).__name__} must define a description.")
        if not isinstance(self.input_schema, dict):
            raise TypeError(f"{type(self).__name__}.input_schema must be a dict.")

    @abc.abstractmethod
    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool. Must return a ToolResult, never raise for bad input."""

    # -- helpers ----------------------------------------------------------

    def validate_input(self, kwargs: Dict[str, Any]) -> Optional[str]:
        """
        Minimal check against ``input_schema``: required keys present, no
        unexpected keys. Returns an error string, or None when valid.

        Deliberately shallow — this guards against a model inventing argument
        names, not against adversarial payloads. Value-level trust is handled
        by ``provenance``, not by validation.
        """
        props = self.input_schema.get("properties", {})
        required = self.input_schema.get("required", [])
        missing = [k for k in required if k not in kwargs]
        if missing:
            return f"missing required argument(s): {missing}"
        if props:
            unexpected = [k for k in kwargs if k not in props]
            if unexpected:
                return (
                    f"unexpected argument(s): {unexpected}; "
                    f"expected any of {sorted(props)}"
                )
        return None

    def spec(self) -> Dict[str, Any]:
        """Tool-calling spec, for an LLM API or for prompts.TOOL_DESCRIPTIONS."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name!r}>"


class ToolRegistry:
    """Name-to-tool lookup with a dispatch that never raises into the loop."""

    def __init__(self, tools: Optional[List[BaseTool]] = None) -> None:
        self._tools: Dict[str, BaseTool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"a tool named {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools)

    def specs(self) -> List[Dict[str, Any]]:
        return [t.spec() for t in self._tools.values()]

    def call(self, name: str, **kwargs: Any) -> ToolResult:
        """
        Dispatch by name.

        An unknown tool or a raising tool becomes a failed ToolResult rather
        than an exception: a model that hallucinates a tool name should get a
        correctable message back, not crash the run.
        """
        tool = self.get(name)
        if tool is None:
            return ToolResult.failure(
                name, f"no such tool; available tools are {sorted(self._tools)}"
            )
        problem = tool.validate_input(kwargs)
        if problem is not None:
            return ToolResult.failure(name, problem)
        try:
            return tool.run(**kwargs)
        except Exception as exc:  # noqa: BLE001 - tools must not break the loop
            return ToolResult.failure(name, f"{type(exc).__name__}: {exc}")

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools


__all__ = [
    "BaseTool",
    "Provenance",
    "ToolRegistry",
    "ToolResult",
    "UNTRUSTED_PROVENANCE",
]
