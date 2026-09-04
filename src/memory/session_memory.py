"""
Session memory management.

Deliberately separates:
  - current-session memory (built and discarded within one run)
  - cross-session memory (persisted between runs)

This split is the future entry point for Phase 3's cross-session
memory poisoning threat surface. For Phase 1 both are just backed
by the same MemoryStore with no adversarial handling — the
separation only needs to exist structurally, not defensively, yet.

What the split buys, concretely: cross-session memory is the only channel
where something written during one analysis can influence a *later* one. That
is the whole of Phase 3's threat model. Keeping within-run scratch notes in a
separate object means an experiment can poison the persistent store without
disturbing the transient one, and can measure how a note crossed between them.

Both classes expose ``recall(record) -> str``, the interface
``ReactAgent`` expects, and both are treated as untrusted when rendered into a
prompt — see ``Provenance.MEMORY``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .store import DEFAULT_K, MemoryRecord, MemoryStore, build_store

DEFAULT_RECALL_K = 3
#: Notes below this similarity are not worth spending context on.
DEFAULT_MIN_SCORE = 0.05


def summarize_record(record: Mapping[str, Any], max_fields: int = 8) -> str:
    """Render a flow as a short query string for similarity search."""
    parts = []
    for key, value in list(record.items())[:max_fields]:
        parts.append(f"{key} {value}")
    return " ".join(parts)


@dataclass
class Note:
    """One thing worth remembering."""

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class SessionMemory:
    """
    Scratch memory for a single run. Cleared between records.

    Holds what the agent has already established about the flow in front of it,
    so a multi-step run does not re-derive the same fact. Nothing here survives
    the run, so nothing here can influence a later analysis — which is exactly
    why it is kept apart from the persistent store.
    """

    provenance = "session"

    def __init__(self, max_notes: int = 50) -> None:
        self.max_notes = max_notes
        self._notes: List[Note] = []

    def remember(self, text: str, **metadata: Any) -> None:
        if not str(text).strip():
            return
        self._notes.append(Note(text=str(text), metadata=dict(metadata)))
        if len(self._notes) > self.max_notes:
            self._notes = self._notes[-self.max_notes :]

    def notes(self) -> List[Note]:
        return list(self._notes)

    def recall(self, record: Optional[Mapping[str, Any]] = None) -> str:
        """Everything noted so far this run, most recent last."""
        return "\n".join(f"- {n.text}" for n in self._notes)

    def clear(self) -> None:
        self._notes = []

    def __len__(self) -> int:
        return len(self._notes)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<SessionMemory n={len(self._notes)}>"


class CrossSessionMemory:
    """
    Memory that persists between runs, backed by a :class:`MemoryStore`.

    This is the Phase 3 attack surface: anything written here can be retrieved
    while analysing a *different* flow, possibly in a different session. Phase 1
    adds no defences, per the module docstring — but the class records enough
    provenance on each note (which run wrote it, and what verdict it came from)
    that a later experiment can trace an influenced decision back to the note
    that caused it. Without that, a poisoning result is unattributable.
    """

    provenance = "cross_session"

    def __init__(
        self,
        store: Optional[MemoryStore] = None,
        *,
        backend: str = "tfidf",
        path: Optional[str] = None,
        k: int = DEFAULT_RECALL_K,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> None:
        self.store = store if store is not None else build_store(backend)
        self.path = path
        self.k = k
        self.min_score = min_score
        if path and os.path.exists(path):
            self.store.load(path)

    # -- writing -----------------------------------------------------------

    def remember(
        self,
        text: str,
        *,
        session_id: Optional[str] = None,
        verdict: Optional[str] = None,
        **metadata: Any,
    ) -> Optional[str]:
        """Persist one note. Returns its id, or None if the text was empty."""
        if not str(text).strip():
            return None
        meta = {"session_id": session_id, "verdict": verdict, **metadata}
        return self.store.add([str(text)], [meta])[0]

    def remember_verdict(
        self,
        record: Mapping[str, Any],
        verdict: str,
        *,
        reasoning: str = "",
        session_id: Optional[str] = None,
    ) -> Optional[str]:
        """Record the outcome of one analysis so later runs can recall it."""
        summary = summarize_record(record)
        text = f"A flow with {summary} was judged {verdict}."
        if reasoning:
            text += f" {reasoning}"
        return self.remember(
            text, session_id=session_id, verdict=verdict, kind="verdict"
        )

    # -- reading -----------------------------------------------------------

    def search(self, query: str, k: Optional[int] = None) -> List[MemoryRecord]:
        results = self.store.query(query, k or self.k)
        return [r for r in results if r.score >= self.min_score]

    def recall(self, record: Mapping[str, Any]) -> str:
        """
        Text to show the agent for this flow, or an empty string.

        Returned to the loop and rendered as untrusted, because a note may have
        been written by an earlier, possibly manipulated, run.
        """
        results = self.search(summarize_record(record))
        if not results:
            return ""
        return "\n".join(f"- {r.text} (similarity {r.score:.2f})" for r in results)

    # -- lifecycle ---------------------------------------------------------

    def persist(self, path: Optional[str] = None) -> str:
        target = path or self.path
        if not target:
            raise ValueError(
                "no path given and none configured; CrossSessionMemory cannot "
                "persist without one."
            )
        self.store.save(target)
        return target

    def clear(self) -> int:
        return self.store.clear()

    def __len__(self) -> int:
        return len(self.store)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<CrossSessionMemory n={len(self)} backend={self.store.backend!r}>"


class CombinedMemory:
    """
    Presents both memories to the agent through one ``recall``.

    Kept as a wrapper rather than merging the two stores, so the boundary the
    module docstring insists on survives contact with convenience. Each recalled
    line is labelled with which memory it came from, so a Phase 3 experiment can
    tell an injected persistent note from a legitimate within-run one.
    """

    def __init__(
        self, session: SessionMemory, cross_session: CrossSessionMemory
    ) -> None:
        self.session = session
        self.cross_session = cross_session

    def recall(self, record: Mapping[str, Any]) -> str:
        parts = []
        current = self.session.recall(record)
        if current:
            parts.append("From this session:\n" + current)
        prior = self.cross_session.recall(record)
        if prior:
            parts.append("From earlier sessions:\n" + prior)
        return "\n\n".join(parts)


__all__ = [
    "CombinedMemory",
    "CrossSessionMemory",
    "Note",
    "SessionMemory",
    "summarize_record",
]
