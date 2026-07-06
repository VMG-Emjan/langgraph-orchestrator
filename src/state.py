"""Shared graph state for the LangGraph multi-agent orchestrator.

The planner -> tool -> critic loop reads and writes this single TypedDict
state object as it flows through the StateGraph.
"""

from __future__ import annotations

import operator
from typing import Annotated, List, TypedDict


class Step(TypedDict):
    """A single planned sub-task and the result the tool produced for it."""

    description: str
    tool_output: str


class OrchestratorState(TypedDict, total=False):
    """State threaded through every node of the StateGraph.

    Fields:
        task:          Original user request.
        plan:          Ordered sub-tasks produced by planner_node.
        results:       Tool outputs collected by tool_node (append-only).
        critique:      Latest critic_node verdict text.
        approved:      Critic decision; True routes to END, False loops back.
        retries:       How many planner->tool->critic passes have run.
        max_retries:   Hard cap so the loop can never run forever.
        trace:         Human-readable execution trace (append-only).
    """

    task: str
    plan: List[str]
    results: List[Step]  # overwritten each pass; latest pass wins
    critique: str
    approved: bool
    retries: int
    max_retries: int
    fail_first: bool  # demo: force one failed step on pass 1 to trigger a retry
    trace: Annotated[List[str], operator.add]
