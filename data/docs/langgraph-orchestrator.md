# LangGraph Multi-Agent Orchestrator

The langgraph-orchestrator project is a multi-agent orchestration proof built on the
open-source LangGraph Python library (no managed LangGraph Platform, no LangSmith).
It wires a planner -> tool -> critic loop as a StateGraph with conditional edges.

The planner agent breaks a task into ordered sub-tasks using the DeepSeek chat model.
The tool node executes each step deterministically so CI never needs an API key.
The critic agent judges the tool outputs and either approves the run or routes the
graph back to the planner through `add_conditional_edges`, forming a retry loop.

A MemorySaver checkpointer persists state per thread id, so every run is resumable.
The retry loop-back was proven live with a `fail_first` flag: the tool emits an empty
output on pass one, the critic flags it as `<NO_OUTPUT>` and rejects, the planner
produces a better plan, and pass two is approved.
