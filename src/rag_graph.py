"""LangGraph StateGraph wiring the RAG retrieve -> answer -> judge loop.

Same shape as the orchestrator graph: a conditional edge after the judge
either finishes (END) or routes through a query rewrite back to retrieval,
giving the RAG pipeline a self-correcting retry loop. A MemorySaver
checkpointer persists state per thread id.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .rag_nodes import answer_node, judge_node, retrieve_node, rewrite_node, route_after_judge
from .rag_state import RagState


def build_rag_graph(checkpointer: MemorySaver | None = None):
    """Compile and return the retrieve->answer->judge StateGraph.

    Topology:
        retrieve -> answer -> judge -> (end)     END
                                    -> (rewrite) rewrite -> retrieve
    """
    builder = StateGraph(RagState)

    builder.add_node("retrieve", retrieve_node)
    builder.add_node("answer", answer_node)
    builder.add_node("judge", judge_node)
    builder.add_node("rewrite", rewrite_node)

    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "answer")
    builder.add_edge("answer", "judge")
    builder.add_conditional_edges(
        "judge",
        route_after_judge,
        {"rewrite": "rewrite", "end": END},
    )
    builder.add_edge("rewrite", "retrieve")

    return builder.compile(checkpointer=checkpointer or MemorySaver())
