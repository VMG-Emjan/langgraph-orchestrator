"""LangGraph StateGraph wiring the planner -> tool -> critic loop.

The critic uses a conditional edge (``add_conditional_edges``) to either
finish (END) or route back to the planner for another pass, giving the
multi-agent orchestration its retry loop. A MemorySaver checkpointer
persists state across steps so a run is resumable by thread id.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .nodes import critic_node, planner_node, route_after_critic, tool_node
from .state import OrchestratorState


def build_graph(checkpointer: MemorySaver | None = None):
    """Compile and return the planner->tool->critic StateGraph.

    Topology:
        planner -> tool -> critic -> (retry) planner
                                   -> (end)   END
    """
    builder = StateGraph(OrchestratorState)

    builder.add_node("planner", planner_node)
    builder.add_node("tool", tool_node)
    builder.add_node("critic", critic_node)

    builder.set_entry_point("planner")
    builder.add_edge("planner", "tool")
    builder.add_edge("tool", "critic")
    builder.add_conditional_edges(
        "critic",
        route_after_critic,
        {"retry": "planner", "end": END},
    )

    return builder.compile(checkpointer=checkpointer or MemorySaver())
