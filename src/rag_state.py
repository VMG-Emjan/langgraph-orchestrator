"""Shared graph state for the RAG retrieve -> answer -> judge loop.

Only ``trace`` accumulates across passes (operator.add); everything else is
last-write-wins so a retry pass fully replaces stale docs/answers — the same
lesson the orchestrator state learned with ``results``.
"""

from __future__ import annotations

import operator
from typing import Annotated, List, TypedDict

from .rag import RetrievedDoc


class Verdict(TypedDict, total=False):
    """Structured judge output."""

    faithful: bool
    relevance: float
    score: float
    reason: str


class RagState(TypedDict, total=False):
    """State threaded through the RAG StateGraph.

    Fields:
        question:    Original user question (never mutated).
        query:       Current retrieval query; rewritten on judge rejection.
        docs:        Chunks retrieved for the current query.
        answer:      Generated answer grounded in ``docs``.
        verdict:     Judge JSON verdict for the current answer.
        approved:    Judge decision; True routes to END, False loops back.
        retries:     Completed retrieve->answer->judge passes.
        max_retries: Hard cap on loop-backs.
        trace:       Human-readable execution trace (append-only).
    """

    question: str
    query: str
    docs: List[RetrievedDoc]
    answer: str
    verdict: Verdict
    approved: bool
    retries: int
    max_retries: int
    trace: Annotated[List[str], operator.add]
