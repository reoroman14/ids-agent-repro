"""
Unified vector-store interface over FAISS or Chroma.

Backend is chosen via config; agent code should never import
FAISS/Chroma directly.

The default backend is ``tfidf``: an in-process store built on scikit-learn,
which is already a dependency. Phase 1 stores at most a few thousand short
notes, a scale at which an approximate index buys nothing, and a backend that
needs no embedding model keeps the memory layer runnable without a model server
— which matters because Phase 3's poisoning experiments must be reproducible
from a fixed corpus rather than from whatever an embedding API returns today.

``faiss`` is available for larger corpora and takes an ``embed_fn``, since
FAISS indexes vectors and has no opinion about where they come from.
"""

from __future__ import annotations

import abc
import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

DEFAULT_BACKEND = "tfidf"
DEFAULT_K = 5


@dataclass
class MemoryRecord:
    """One stored note, and its similarity score when returned from a query."""

    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "text": self.text, "metadata": self.metadata,
                "score": self.score}


class MemoryStore(abc.ABC):
    """Add, query and delete text notes. The only interface agent code sees."""

    backend: str = ""

    @abc.abstractmethod
    def add(
        self,
        texts: Sequence[str],
        metadata: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[str]:
        """Store notes and return their ids."""

    @abc.abstractmethod
    def query(self, text: str, k: int = DEFAULT_K) -> List[MemoryRecord]:
        """Return the k most similar notes, most similar first."""

    @abc.abstractmethod
    def delete(self, ids: Sequence[str]) -> int:
        """Remove notes by id. Returns how many were removed."""

    @abc.abstractmethod
    def all_records(self) -> List[MemoryRecord]:
        """Every stored note, in insertion order."""

    def clear(self) -> int:
        return self.delete([r.id for r in self.all_records()])

    def __len__(self) -> int:
        return len(self.all_records())

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} backend={self.backend!r} n={len(self)}>"

    # -- persistence -------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Write the corpus to JSON.

        Deliberately stores the source texts rather than an index: the index is
        rebuildable, and Phase 3 needs a poisoned corpus to be inspectable and
        diffable, which a binary index would prevent.
        """
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            "backend": self.backend,
            "records": [
                {"id": r.id, "text": r.text, "metadata": r.metadata}
                for r in self.all_records()
            ],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def load(self, path: str, *, replace: bool = True) -> int:
        """Read a corpus back. Returns the number of notes loaded."""
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        records = payload.get("records", [])
        if replace:
            self.clear()
        self.add(
            [r["text"] for r in records],
            [{**r.get("metadata", {}), "_id": r.get("id")} for r in records],
        )
        return len(records)


class TfidfMemoryStore(MemoryStore):
    """
    In-process store using TF-IDF vectors and cosine similarity.

    The vectoriser is refitted whenever the corpus changes. That is O(corpus)
    per write and would be wrong at scale, but at Phase 1 sizes it is instant
    and it keeps retrieval exactly reproducible across runs — an approximate
    index whose results shift with insertion order would make a poisoning
    result impossible to attribute.
    """

    backend = "tfidf"

    def __init__(self, **vectorizer_kwargs: Any) -> None:
        self._records: List[MemoryRecord] = []
        self._vectorizer = None
        self._matrix = None
        self._kwargs = {"stop_words": None, "lowercase": True, **vectorizer_kwargs}

    def add(
        self,
        texts: Sequence[str],
        metadata: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[str]:
        texts = list(texts)
        if metadata is not None and len(metadata) != len(texts):
            raise ValueError(
                f"metadata has {len(metadata)} entries for {len(texts)} texts."
            )
        ids: List[str] = []
        for index, text in enumerate(texts):
            meta = dict(metadata[index]) if metadata else {}
            record_id = str(meta.pop("_id", None) or uuid.uuid4())
            self._records.append(
                MemoryRecord(id=record_id, text=str(text), metadata=meta)
            )
            ids.append(record_id)
        self._reindex()
        return ids

    def query(self, text: str, k: int = DEFAULT_K) -> List[MemoryRecord]:
        if not self._records or not str(text).strip() or self._matrix is None:
            return []
        from sklearn.metrics.pairwise import linear_kernel

        try:
            vector = self._vectorizer.transform([str(text)])
        except Exception:  # noqa: BLE001 - unseen vocabulary only
            return []
        scores = linear_kernel(vector, self._matrix).ravel()
        order = scores.argsort()[::-1][: max(1, k)]
        return [
            MemoryRecord(
                id=self._records[i].id,
                text=self._records[i].text,
                metadata=dict(self._records[i].metadata),
                score=float(scores[i]),
            )
            for i in order
            if scores[i] > 0
        ]

    def delete(self, ids: Sequence[str]) -> int:
        wanted = set(ids)
        before = len(self._records)
        self._records = [r for r in self._records if r.id not in wanted]
        removed = before - len(self._records)
        if removed:
            self._reindex()
        return removed

    def all_records(self) -> List[MemoryRecord]:
        return list(self._records)

    def _reindex(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        if not self._records:
            self._vectorizer, self._matrix = None, None
            return
        self._vectorizer = TfidfVectorizer(**self._kwargs)
        try:
            self._matrix = self._vectorizer.fit_transform(
                [r.text for r in self._records]
            )
        except ValueError:
            # Every document was empty or pure stop words.
            self._vectorizer, self._matrix = None, None


class FaissMemoryStore(MemoryStore):
    """
    FAISS-backed store for corpora too large for the TF-IDF backend.

    Requires ``embed_fn``: a callable turning a list of strings into an
    (n, dim) array. FAISS indexes vectors and does not produce them, so the
    embedding choice stays with the caller rather than being hidden here.
    """

    backend = "faiss"

    def __init__(self, embed_fn: Callable[[Sequence[str]], Any], dim: Optional[int] = None):
        if embed_fn is None:
            raise ValueError(
                "the faiss backend needs an embed_fn; FAISS stores vectors but "
                "does not create them. Use backend 'tfidf' for a self-contained "
                "store."
            )
        self.embed_fn = embed_fn
        self._records: List[MemoryRecord] = []
        self._index = None
        self._dim = dim

    def _ensure_index(self, dim: int) -> None:
        if self._index is not None:
            return
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "faiss-cpu is not installed; `pip install faiss-cpu` or use "
                "backend 'tfidf'."
            ) from exc
        self._dim = dim
        self._index = faiss.IndexFlatIP(dim)

    def add(
        self,
        texts: Sequence[str],
        metadata: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[str]:
        import numpy as np

        texts = list(texts)
        if not texts:
            return []
        vectors = np.asarray(self.embed_fn(texts), dtype="float32")
        if vectors.ndim != 2 or len(vectors) != len(texts):
            raise ValueError(
                f"embed_fn must return one vector per text; got {vectors.shape} "
                f"for {len(texts)} texts."
            )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)

        self._ensure_index(vectors.shape[1])
        ids: List[str] = []
        for index, text in enumerate(texts):
            meta = dict(metadata[index]) if metadata else {}
            record_id = str(meta.pop("_id", None) or uuid.uuid4())
            self._records.append(MemoryRecord(id=record_id, text=str(text), metadata=meta))
            ids.append(record_id)
        self._index.add(vectors)
        return ids

    def query(self, text: str, k: int = DEFAULT_K) -> List[MemoryRecord]:
        import numpy as np

        if self._index is None or not self._records:
            return []
        vector = np.asarray(self.embed_fn([str(text)]), dtype="float32")
        vector = vector / np.maximum(np.linalg.norm(vector, axis=1, keepdims=True), 1e-12)
        scores, positions = self._index.search(vector, min(k, len(self._records)))
        out = []
        for score, position in zip(scores[0], positions[0]):
            if position < 0:
                continue
            record = self._records[int(position)]
            out.append(
                MemoryRecord(id=record.id, text=record.text,
                             metadata=dict(record.metadata), score=float(score))
            )
        return out

    def delete(self, ids: Sequence[str]) -> int:
        # IndexFlatIP has no stable per-id removal, so the index is rebuilt.
        wanted = set(ids)
        keep = [r for r in self._records if r.id not in wanted]
        removed = len(self._records) - len(keep)
        if removed:
            self._records, self._index = [], None
            if keep:
                self.add([r.text for r in keep],
                         [{**r.metadata, "_id": r.id} for r in keep])
        return removed

    def all_records(self) -> List[MemoryRecord]:
        return list(self._records)


_BACKENDS = {"tfidf": TfidfMemoryStore, "faiss": FaissMemoryStore}


def build_store(
    backend: str = DEFAULT_BACKEND, config: Optional[Dict[str, Any]] = None
) -> MemoryStore:
    """Build the configured store. Agent code calls this, never a backend."""
    key = str(backend or DEFAULT_BACKEND).lower()
    if key in ("chroma", "chromadb"):
        raise NotImplementedError(
            "the chroma backend is not implemented; requirements.txt lists it "
            "as an alternative to faiss-cpu. Use 'tfidf' or 'faiss'."
        )
    if key not in _BACKENDS:
        raise ValueError(
            f"unknown memory backend {key!r}; expected one of {sorted(_BACKENDS)}."
        )
    return _BACKENDS[key](**dict(config or {}))


__all__ = [
    "DEFAULT_K",
    "FaissMemoryStore",
    "MemoryRecord",
    "MemoryStore",
    "TfidfMemoryStore",
    "build_store",
]
