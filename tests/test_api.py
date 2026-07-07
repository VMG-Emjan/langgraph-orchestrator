"""FastAPI endpoint tests.

Runs the real app with the fake hash embedding and mocked LLMs, so /ask,
/eval, /trace and /orchestrate are exercised end-to-end with no API key.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import src.graph as orch_graph
import src.rag as rag
import src.rag_nodes as rag_nodes
from tests.test_rag_graph import FakeLLM


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_FAKE_EMBED", "1")
    monkeypatch.setenv("TRACE_DB", str(tmp_path / "traces.db"))
    rag.reset()

    # Every judge call approves; answer/rewrite are scripted.
    def fresh_llm(*a, **k):
        return FakeLLM(
            ['{"faithful": true, "relevance": 1, "score": 0.9, "reason": "ok"}'] * 10
        )

    monkeypatch.setattr(rag_nodes, "get_llm", fresh_llm)

    # Orchestrator planner/critic mocked like tests/test_orchestrator.py.
    def fake_planner(state):
        return {"plan": ["step-1", "step-2"], "trace": ["[planner] fake"]}

    def fake_critic(state):
        results = state.get("results", [])
        ok = all(r["tool_output"] for r in results)
        return {
            "approved": ok,
            "critique": "ok" if ok else "empty output",
            "retries": state.get("retries", 0) + 1,
            "trace": [f"[critic] approved={ok}"],
        }

    monkeypatch.setattr(orch_graph, "planner_node", fake_planner)
    monkeypatch.setattr(orch_graph, "critic_node", fake_critic)

    from app.main import create_app

    with TestClient(create_app()) as c:
        yield c
    rag.reset()


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["docs_indexed"] > 0


def test_ask_returns_answer_verdict_and_trace(client):
    res = client.post("/ask", json={"question": "What does rag-eval-lab measure?"})
    assert res.status_code == 200
    body = res.json()
    assert body["answer"]
    assert body["approved"] is True
    assert body["verdict"]["score"] == 0.9
    assert body["sources"]

    # Trace persisted and retrievable.
    trace = client.get(f"/trace/{body['trace_id']}")
    assert trace.status_code == 200
    assert trace.json()["kind"] == "ask"
    assert any("[judge]" in ln for ln in trace.json()["trace"])


def test_trace_404(client):
    assert client.get("/trace/nope").status_code == 404


def test_orchestrate_fail_first_retries_live(client):
    res = client.post("/orchestrate", json={"task": "demo task", "fail_first": True})
    assert res.status_code == 200
    body = res.json()
    assert body["approved"] is True
    assert body["retries"] == 2  # pass 1 rejected (empty output), pass 2 approved


def test_eval_golden_set(client):
    res = client.post("/eval")
    assert res.status_code == 200
    body = res.json()
    assert body["summary"]["questions"] == 3
    assert body["summary"]["all_approved"] is True
    assert 0.0 <= body["summary"]["retrieval_hit_rate"] <= 1.0
