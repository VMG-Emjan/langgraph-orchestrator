"""Lightweight RAG layer: Chroma in-process vector store over ``data/docs``.

Two embedding backends, selected by the ``RAG_FAKE_EMBED`` env var:

* default  — Chroma's built-in ONNX MiniLM embedding (no torch, CPU-friendly,
             downloads the model once at first use). Used in the deployed app.
* fake     — a deterministic bag-of-words hash embedding. Zero downloads and
             zero network, used by tests and offline CI.

The store is an in-memory EphemeralClient: the corpus is tiny and re-ingested
at startup, so no persistence is needed (free-tier disks are ephemeral anyway).
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from pathlib import Path
from typing import List, TypedDict

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"

_FAKE_DIM = 256

_collection = None  # module-level singleton, built by ingest()


class RetrievedDoc(TypedDict):
    text: str
    source: str
    score: float


class HashEmbedding:
    """Deterministic bag-of-words hash embedding (tests / offline use)."""

    def name(self) -> str:  # chroma may ask for a name when persisting config
        return "hash-embed"

    def __call__(self, input: List[str]) -> List[List[float]]:  # noqa: A002 (chroma API name)
        return [self._embed(t) for t in input]

    @staticmethod
    def _embed(text: str) -> List[float]:
        vec = [0.0] * _FAKE_DIM
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.md5(token.encode()).digest()
            idx = int.from_bytes(digest[:4], "little") % _FAKE_DIM
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def _embedding_function():
    if os.environ.get("RAG_FAKE_EMBED"):
        return HashEmbedding()
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    return DefaultEmbeddingFunction()  # ONNX MiniLM-L6-v2


def _chunk(text: str) -> List[str]:
    """Split a markdown file into paragraph chunks."""
    parts = [p.strip() for p in re.split(r"\n\s*\n", text)]
    return [p for p in parts if len(p) >= 40]


def ingest(docs_dir: Path | None = None):
    """(Re)build the in-memory collection from the markdown corpus."""
    global _collection
    import chromadb

    docs_dir = docs_dir or DOCS_DIR
    client = chromadb.EphemeralClient()
    collection = client.create_collection(
        "docs", embedding_function=_embedding_function(), get_or_create=True
    )

    ids: List[str] = []
    texts: List[str] = []
    metas: List[dict] = []
    for path in sorted(docs_dir.glob("*.md")):
        for i, chunk in enumerate(_chunk(path.read_text(encoding="utf-8"))):
            ids.append(f"{path.stem}::{i}")
            texts.append(chunk)
            metas.append({"source": path.stem})
    if ids:
        collection.add(ids=ids, documents=texts, metadatas=metas)

    _collection = collection
    return collection


def reset() -> None:
    """Drop the singleton (tests switch embedding backends between runs)."""
    global _collection
    _collection = None


def count() -> int:
    return _collection.count() if _collection is not None else 0


def retrieve(query: str, k: int = 4) -> List[RetrievedDoc]:
    """Return the top-k chunks for a query with their source and distance."""
    collection = _collection if _collection is not None else ingest()
    res = collection.query(query_texts=[query], n_results=min(k, collection.count()))
    docs: List[RetrievedDoc] = []
    for text, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        docs.append(RetrievedDoc(text=text, source=str(meta["source"]), score=float(dist)))
    return docs
