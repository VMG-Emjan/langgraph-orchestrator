"""Tests for the RAG retrieve -> answer -> judge graph.

DeepSeek is never called: ``get_llm`` is monkeypatched in src.rag_nodes with a
scripted fake, and ``retrieve`` is replaced with a canned result, so the suite
is deterministic and needs no API key, no network and no embedding model.
"""

from __future__ import annotations

import src.rag_nodes as rag_nodes
from src.rag_graph import build_rag_graph
from src.rag_nodes import _parse_verdict, route_after_judge


class _Reply:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    """Scripted LLM: picks a reply by inspecting the system prompt."""

    def __init__(self, judge_replies):
        self.judge_replies = list(judge_replies)

    def invoke(self, messages):
        system = messages[0]["content"]
        if system == rag_nodes.ANSWER_SYSTEM:
            return _Reply("The answer is X [doc-a].")
        if system == rag_nodes.JUDGE_SYSTEM:
            return _Reply(self.judge_replies.pop(0))
        if system == rag_nodes.REWRITE_SYSTEM:
            return _Reply("rewritten query")
        raise AssertionError(f"unexpected system prompt: {system[:60]}")


def _fake_retrieve(query, k=4):
    return [{"text": f"chunk for {query}", "source": "doc-a", "score": 0.1}]


# --------------------------------------------------------------------------- #
# 1. conditional routing: approve ends, reject rewrites, cap ends
# --------------------------------------------------------------------------- #
def test_route_after_judge():
    assert route_after_judge({"approved": True, "retries": 1, "max_retries": 2}) == "end"
    assert route_after_judge({"approved": False, "retries": 1, "max_retries": 2}) == "rewrite"
    assert route_after_judge({"approved": False, "retries": 2, "max_retries": 2}) == "end"


# --------------------------------------------------------------------------- #
# 2. verdict parser tolerates fences and garbage
# --------------------------------------------------------------------------- #
def test_parse_verdict():
    good = '```json\n{"faithful": true, "relevance": 0.9, "score": 0.8, "reason": "ok"}\n```'
    v = _parse_verdict(good)
    assert v["faithful"] is True and v["score"] == 0.8

    bad = _parse_verdict("not json at all")
    assert bad["faithful"] is False and bad["score"] == 0.0


# --------------------------------------------------------------------------- #
# 3. approved on first pass: no rewrite, one retrieve
# --------------------------------------------------------------------------- #
def test_graph_approves_first_pass(monkeypatch):
    fake = FakeLLM(['{"faithful": true, "relevance": 1, "score": 0.9, "reason": "good"}'])
    monkeypatch.setattr(rag_nodes, "get_llm", lambda *a, **k: fake)
    monkeypatch.setattr(rag_nodes, "retrieve", _fake_retrieve)

    final = build_rag_graph().invoke(
        {"question": "q?", "max_retries": 2, "retries": 0},
        config={"configurable": {"thread_id": "t1"}},
    )
    assert final["approved"] is True
    assert final["retries"] == 1
    assert final["answer"] == "The answer is X [doc-a]."
    assert not any("[rewrite]" in ln for ln in final["trace"])


# --------------------------------------------------------------------------- #
# 4. judge rejects pass 1 -> rewrite -> retrieve again -> approve pass 2
# --------------------------------------------------------------------------- #
def test_graph_rewrites_then_approves(monkeypatch):
    fake = FakeLLM(
        [
            '{"faithful": false, "relevance": 0.2, "score": 0.2, "reason": "off-topic"}',
            '{"faithful": true, "relevance": 0.9, "score": 0.85, "reason": "grounded"}',
        ]
    )
    retrieve_calls = []

    def counting_retrieve(query, k=4):
        retrieve_calls.append(query)
        return _fake_retrieve(query, k)

    monkeypatch.setattr(rag_nodes, "get_llm", lambda *a, **k: fake)
    monkeypatch.setattr(rag_nodes, "retrieve", counting_retrieve)

    final = build_rag_graph().invoke(
        {"question": "q?", "max_retries": 2, "retries": 0},
        config={"configurable": {"thread_id": "t2"}},
    )
    assert final["approved"] is True
    assert final["retries"] == 2
    assert retrieve_calls == ["q?", "rewritten query"]  # loop-back happened
    assert any("[rewrite]" in ln for ln in final["trace"])


# --------------------------------------------------------------------------- #
# 5. retry cap: two rejections end the loop unapproved
# --------------------------------------------------------------------------- #
def test_graph_stops_at_retry_cap(monkeypatch):
    reject = '{"faithful": false, "relevance": 0.1, "score": 0.1, "reason": "bad"}'
    fake = FakeLLM([reject, reject])
    monkeypatch.setattr(rag_nodes, "get_llm", lambda *a, **k: fake)
    monkeypatch.setattr(rag_nodes, "retrieve", _fake_retrieve)

    final = build_rag_graph().invoke(
        {"question": "q?", "max_retries": 2, "retries": 0},
        config={"configurable": {"thread_id": "t3"}},
    )
    assert final["approved"] is False
    assert final["retries"] == 2
