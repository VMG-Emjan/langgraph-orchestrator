"""Tests for the Chroma store using the deterministic hash embedding.

RAG_FAKE_EMBED avoids the ONNX model download, so ingestion and retrieval run
fully offline while still exercising the real Chroma query path.
"""

from __future__ import annotations

import pytest

import src.rag as rag


@pytest.fixture(autouse=True)
def fake_embed(monkeypatch):
    monkeypatch.setenv("RAG_FAKE_EMBED", "1")
    rag.reset()
    yield
    rag.reset()


def test_ingest_indexes_corpus():
    collection = rag.ingest()
    assert collection.count() > 5  # several chunks across the doc set


def test_retrieve_finds_relevant_source():
    rag.ingest()
    docs = rag.retrieve("Helm chart kind cluster Kubernetes Terraform OIDC", k=3)
    assert len(docs) == 3
    assert any(d["source"] == "n8n-runner-deploy" for d in docs)


def test_retrieve_returns_text_source_score():
    rag.ingest()
    (doc, *_) = rag.retrieve("Piper TTS faster-whisper voice booking", k=1)
    assert doc["source"] == "voice-agent-poc"
    assert doc["text"] and isinstance(doc["score"], float)
