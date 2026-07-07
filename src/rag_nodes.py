"""Nodes for the RAG retrieve -> answer -> judge loop.

answer_node and judge_node call the real DeepSeek LLM (via ``get_llm`` from
the orchestrator nodes). retrieve_node is deterministic over the local Chroma
store. rewrite_node reformulates the query when the judge rejects an answer,
which is what the conditional retry edge loops back into.

Tests monkeypatch ``get_llm`` and ``retrieve`` in this module's namespace, so
the suite runs with no API key and no embedding model download.
"""

from __future__ import annotations

import json

from .nodes import get_llm
from .rag import retrieve
from .rag_state import RagState, Verdict

APPROVAL_THRESHOLD = 0.6

ANSWER_SYSTEM = (
    "You are a RAG answering agent. Answer the question using ONLY the provided "
    "context chunks. Cite the source of every claim inline like [source-name]. "
    "If the context does not contain the answer, say so explicitly. Be concise."
)

JUDGE_SYSTEM = (
    "You are an evaluation judge for a RAG system. Given a question, retrieved "
    "context and an answer, decide whether the answer is faithful to the context "
    "(no hallucinated claims) and relevant to the question. Reply ONLY with JSON: "
    '{"faithful": true|false, "relevance": <0..1>, "score": <0..1>, '
    '"reason": "<short>"}. score is your overall quality judgment.'
)

REWRITE_SYSTEM = (
    "You rewrite search queries for a document retriever. Given a question, a "
    "failed query and the judge's rejection reason, produce ONE better search "
    "query. Reply with the query text only — no quotes, no prose."
)


def retrieve_node(state: RagState) -> dict:
    """Fetch top-k chunks for the current query (rewritten on retries)."""
    query = state.get("query") or state["question"]
    docs = retrieve(query, k=4)
    sources = [d["source"] for d in docs]
    return {
        "query": query,
        "docs": docs,
        "trace": [f"[retrieve] query={query!r} -> {len(docs)} chunks from {sources}"],
    }


def _context_block(state: RagState) -> str:
    return "\n\n".join(f"[{d['source']}]\n{d['text']}" for d in state.get("docs", []))


def answer_node(state: RagState) -> dict:
    """Generate a grounded, source-cited answer from the retrieved chunks."""
    llm = get_llm()
    user = f"Context:\n{_context_block(state)}\n\nQuestion: {state['question']}"
    reply = llm.invoke(
        [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": user}]
    )
    answer = reply.content.strip()
    return {"answer": answer, "trace": [f"[answer] {len(answer)} chars generated"]}


def _parse_verdict(raw: str) -> Verdict:
    text = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(text)
        return Verdict(
            faithful=bool(data.get("faithful", False)),
            relevance=float(data.get("relevance", 0.0)),
            score=float(data.get("score", 0.0)),
            reason=str(data.get("reason", "")),
        )
    except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
        return Verdict(faithful=False, relevance=0.0, score=0.0, reason=text[:200])


def judge_node(state: RagState) -> dict:
    """LLM judge: faithfulness + relevance verdict on the current answer."""
    llm = get_llm()
    user = (
        f"Question: {state['question']}\n\n"
        f"Context:\n{_context_block(state)}\n\n"
        f"Answer:\n{state.get('answer', '')}"
    )
    reply = llm.invoke(
        [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]
    )
    verdict = _parse_verdict(reply.content)
    retries = state.get("retries", 0) + 1
    approved = verdict["faithful"] and verdict["score"] >= APPROVAL_THRESHOLD
    return {
        "verdict": verdict,
        "approved": approved,
        "retries": retries,
        "trace": [
            f"[judge] pass={retries} approved={approved} "
            f"score={verdict['score']} faithful={verdict['faithful']} "
            f"reason={verdict['reason']!r}"
        ],
    }


def rewrite_node(state: RagState) -> dict:
    """Reformulate the retrieval query after a judge rejection."""
    llm = get_llm()
    reason = state.get("verdict", {}).get("reason", "")
    user = (
        f"Question: {state['question']}\n"
        f"Failed query: {state.get('query', '')}\n"
        f"Judge rejection: {reason}"
    )
    reply = llm.invoke(
        [{"role": "system", "content": REWRITE_SYSTEM}, {"role": "user", "content": user}]
    )
    query = reply.content.strip().strip('"')
    return {"query": query, "trace": [f"[rewrite] new query={query!r}"]}


def route_after_judge(state: RagState) -> str:
    """Conditional edge: END on approval or retry cap, else rewrite+retry."""
    if state.get("approved"):
        return "end"
    if state.get("retries", 0) >= state.get("max_retries", 2):
        return "end"
    return "rewrite"
