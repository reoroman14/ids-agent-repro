"""
Wraps src/memory/store.py as an agent tool: given a query, retrieves
relevant past session context (knowledge/memory retrieval step of
the IDS-Agent loop).

Its results carry ``Provenance.MEMORY`` by default, which makes them untrusted
and therefore fenced when they enter the prompt. That default is deliberate:
anything recalled from a persistent store may have been written by an earlier
run that was itself manipulated, which is precisely Phase 3's threat model. A
store holding only fixed reference material can be marked
``Provenance.KNOWLEDGE`` explicitly — but the safe assumption is the default,
not the exception.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..memory.store import MemoryStore
from .base_tool import BaseTool, Provenance, ToolResult

DEFAULT_K = 3
MAX_K = 10
#: Snippets are truncated so a single long note cannot crowd out the analysis.
MAX_SNIPPET_CHARS = 400


class RetrievalTool(BaseTool):
    """
    Search stored notes for material relevant to the flow under analysis.

    Parameters
    ----------
    store
        Any :class:`MemoryStore`, or an object exposing ``search(query, k)``
        such as :class:`~src.memory.session_memory.CrossSessionMemory`.
    provenance
        ``MEMORY`` (default, untrusted) or ``KNOWLEDGE`` for a fixed corpus.
    default_query
        Used when the model calls the tool with no query — normally a summary
        of the current flow, set by the loop through :meth:`set_flow`.
    """

    name = "search_memory"
    description = (
        "Search notes from earlier analyses and reference material for context "
        "relevant to this flow. Call with a query describing what you want to "
        "know, or with no arguments to search on the current flow's features."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for. Defaults to the current flow.",
            },
            "k": {
                "type": "integer",
                "description": f"How many notes to return (max {MAX_K}).",
            },
        },
        "required": [],
    }

    def __init__(
        self,
        store: Any,
        *,
        provenance: Provenance = Provenance.MEMORY,
        default_k: int = DEFAULT_K,
        default_query: str = "",
    ) -> None:
        if store is None:
            raise ValueError("RetrievalTool needs a store to search.")
        self.store = store
        self.provenance = provenance
        self.default_k = default_k
        self.default_query = default_query
        super().__init__()

    def set_flow(self, record: Dict[str, Any]) -> None:
        """Loop hook: make the current flow the default query."""
        from ..memory.session_memory import summarize_record

        self.default_query = summarize_record(record)

    def _search(self, query: str, k: int) -> List[Any]:
        if hasattr(self.store, "search"):
            return list(self.store.search(query, k))
        if isinstance(self.store, MemoryStore) or hasattr(self.store, "query"):
            return list(self.store.query(query, k))
        raise AttributeError(
            f"{type(self.store).__name__} exposes neither search() nor query()"
        )

    def run(self, **kwargs: Any) -> ToolResult:
        query = str(kwargs.get("query") or self.default_query or "").strip()
        if not query:
            return ToolResult.failure(
                self.name,
                "no query given and no flow loaded to search on",
                provenance=self.provenance,
            )

        raw_k = kwargs.get("k") or self.default_k
        try:
            k = max(1, min(int(raw_k), MAX_K))
        except (TypeError, ValueError):
            k = self.default_k

        results = self._search(query, k)
        hits = [
            {
                "id": getattr(r, "id", ""),
                "text": self._truncate(getattr(r, "text", str(r))),
                "score": float(getattr(r, "score", 0.0)),
                "metadata": dict(getattr(r, "metadata", {}) or {}),
            }
            for r in results
        ]
        data = {"query": query, "n_hits": len(hits), "hits": hits,
                "source": self.provenance.value}
        return ToolResult(
            tool=self.name,
            ok=True,
            data=data,
            rendered=self._render(data),
            provenance=self.provenance,
        )

    @staticmethod
    def _truncate(text: str) -> str:
        text = str(text)
        if len(text) <= MAX_SNIPPET_CHARS:
            return text
        return text[: MAX_SNIPPET_CHARS - 1].rstrip() + "…"

    def _render(self, data: Dict[str, Any]) -> str:
        if not data["hits"]:
            return f"No stored notes matched: {data['query']}"
        lines = [f"{data['n_hits']} note(s) matching: {data['query']}"]
        for index, hit in enumerate(data["hits"], 1):
            verdict = hit["metadata"].get("verdict")
            tag = f" [previous verdict: {verdict}]" if verdict else ""
            lines.append(f"  {index}. (similarity {hit['score']:.2f}){tag} {hit['text']}")
        return "\n".join(lines)


__all__ = ["RetrievalTool"]
