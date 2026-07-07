"""FastAPI service exposing the LangGraph RAG + orchestrator graphs.

Endpoints:
    GET  /health        liveness + indexed chunk count
    POST /ask           RAG: retrieve -> answer -> judge (DeepSeek), traced
    POST /eval          run the golden question set through the judge
    GET  /trace/{id}    node-by-node trace of a previous run (SQLite)
    POST /orchestrate   planner -> tool -> critic demo (fail_first retry loop)

DEEPSEEK_API_KEY comes from the environment (.env locally, platform secret in
deployment). No LangSmith / managed platform anywhere.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import rag
from src.graph import build_graph
from src.rag_graph import build_rag_graph

from .tracing import get_trace, save_trace

load_dotenv()

class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    max_retries: int = Field(default=2, ge=1, le=3)


class OrchestrateRequest(BaseModel):
    task: str = Field(min_length=3, max_length=500)
    fail_first: bool = True
    max_retries: int = Field(default=3, ge=1, le=5)


GOLDEN_SET = [
    {
        "question": "Which retrieval mode reached the best recall@5 in rag-eval-lab, "
        "and what was the number?",
        "expect_source": "rag-eval-lab",
    },
    {
        "question": "How does the fail_first flag demonstrate the orchestrator's "
        "retry loop-back?",
        "expect_source": "langgraph-orchestrator",
    },
    {
        "question": "How did n8n-runner-deploy prove Terraform against AWS without "
        "creating billable resources?",
        "expect_source": "n8n-runner-deploy",
    },
]


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        rag.ingest()
        yield

    app = FastAPI(
        title="langgraph-orchestrator live demo",
        description="RAG + multi-agent orchestration on open-source LangGraph, "
        "judged by DeepSeek. No managed platform, no LangSmith.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "docs_indexed": rag.count()}

    def _run_rag(question: str, max_retries: int, trace_id: str) -> dict:
        graph = build_rag_graph()
        final = graph.invoke(
            {"question": question, "max_retries": max_retries, "retries": 0},
            config={"configurable": {"thread_id": trace_id}},
        )
        return {
            "question": question,
            "answer": final.get("answer", ""),
            "sources": sorted({d["source"] for d in final.get("docs", [])}),
            "verdict": final.get("verdict", {}),
            "approved": final.get("approved", False),
            "retries": final.get("retries", 0),
            "trace": final.get("trace", []),
        }

    @app.post("/ask")
    def ask(req: AskRequest) -> dict:
        trace_id = uuid.uuid4().hex[:12]
        result = _run_rag(req.question, req.max_retries, trace_id)
        save_trace(trace_id, "ask", result)
        return {"trace_id": trace_id, **result}

    @app.post("/eval")
    def eval_golden() -> dict:
        trace_id = uuid.uuid4().hex[:12]
        results = []
        for item in GOLDEN_SET:
            run = _run_rag(item["question"], 2, f"{trace_id}-{len(results)}")
            results.append(
                {
                    "question": item["question"],
                    "expected_source": item["expect_source"],
                    "retrieval_hit": item["expect_source"] in run["sources"],
                    "score": run["verdict"].get("score", 0.0),
                    "faithful": run["verdict"].get("faithful", False),
                    "approved": run["approved"],
                    "answer": run["answer"],
                }
            )
        summary = {
            "questions": len(results),
            "retrieval_hit_rate": sum(r["retrieval_hit"] for r in results) / len(results),
            "mean_judge_score": round(sum(r["score"] for r in results) / len(results), 3),
            "all_approved": all(r["approved"] for r in results),
        }
        payload = {"summary": summary, "results": results}
        save_trace(trace_id, "eval", payload)
        return {"trace_id": trace_id, **payload}

    @app.get("/trace/{trace_id}")
    def trace(trace_id: str) -> dict:
        found = get_trace(trace_id)
        if found is None:
            raise HTTPException(status_code=404, detail="trace not found")
        return found

    @app.post("/orchestrate")
    def orchestrate(req: OrchestrateRequest) -> dict:
        trace_id = uuid.uuid4().hex[:12]
        graph = build_graph()
        final = graph.invoke(
            {
                "task": req.task,
                "fail_first": req.fail_first,
                "max_retries": req.max_retries,
                "retries": 0,
            },
            config={"configurable": {"thread_id": trace_id}},
        )
        result = {
            "task": req.task,
            "approved": final.get("approved", False),
            "retries": final.get("retries", 0),
            "plan": final.get("plan", []),
            "critique": final.get("critique", ""),
            "trace": final.get("trace", []),
        }
        save_trace(trace_id, "orchestrate", result)
        return {"trace_id": trace_id, **result}

    return app


app = create_app()
