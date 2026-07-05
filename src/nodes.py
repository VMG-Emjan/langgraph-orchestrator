"""Nodes for the planner -> tool -> critic orchestration loop.

planner_node and critic_node call a real DeepSeek LLM (OpenAI-compatible).
tool_node is deterministic so runs are reproducible and CI needs no API key.

The LLM is created lazily via ``get_llm`` so tests can monkeypatch the
planner/critic entry points without ever constructing a real client.
"""

from __future__ import annotations

import json
import os
from typing import List

from .state import OrchestratorState, Step


def get_llm(temperature: float = 0.0):
    """Build a DeepSeek chat model (OpenAI-compatible endpoint).

    Reads DEEPSEEK_API_KEY from the environment. Imported lazily so that
    importing this module never requires the key or the network.
    """
    from langchain_deepseek import ChatDeepSeek

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set. Export it before running the example: "
            "export DEEPSEEK_API_KEY=sk-..."
        )
    return ChatDeepSeek(model="deepseek-chat", temperature=temperature, api_key=api_key)


# --------------------------------------------------------------------------- #
# planner
# --------------------------------------------------------------------------- #

PLANNER_SYSTEM = (
    "You are a planning agent. Break the user's task into 2-4 concrete, "
    "ordered sub-tasks. Reply ONLY with a JSON array of short strings, "
    "no prose, no markdown fences."
)


def _parse_plan(raw: str) -> List[str]:
    """Extract a list of steps from an LLM reply, tolerant of stray fences."""
    text = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x) for x in data if str(x).strip()]
    except json.JSONDecodeError:
        pass
    # Fallback: one step per non-empty line.
    return [ln.strip("-* ").strip() for ln in text.splitlines() if ln.strip()]


def planner_node(state: OrchestratorState) -> dict:
    """Turn the task (plus any prior critique) into an ordered plan."""
    task = state["task"]
    prior = state.get("critique", "")
    user = f"Task: {task}"
    if prior:
        user += f"\n\nThe previous attempt was rejected: {prior}\nProduce a better plan."

    llm = get_llm()
    reply = llm.invoke(
        [{"role": "system", "content": PLANNER_SYSTEM}, {"role": "user", "content": user}]
    )
    plan = _parse_plan(reply.content)
    return {
        "plan": plan,
        "trace": [f"[planner] produced {len(plan)} steps: {plan}"],
    }


# --------------------------------------------------------------------------- #
# tool (deterministic)
# --------------------------------------------------------------------------- #


def _run_tool(step: str) -> str:
    """Deterministic stand-in tool. Swap for a real API/search call as needed."""
    return f"executed<{step}>=OK(len={len(step)})"


def tool_node(state: OrchestratorState) -> dict:
    """Run the deterministic tool for every planned step."""
    results: List[Step] = []
    for step in state.get("plan", []):
        results.append(Step(description=step, tool_output=_run_tool(step)))
    return {
        "results": results,
        "trace": [f"[tool] executed {len(results)} step(s)"],
    }


# --------------------------------------------------------------------------- #
# critic
# --------------------------------------------------------------------------- #

CRITIC_SYSTEM = (
    "You are a critic agent. Given a task and the tool outputs, decide if the "
    'work is sufficient. Reply ONLY with JSON: {"approved": true|false, '
    '"reason": "<short>"}. Approve when every sub-task has a non-empty result.'
)


def _parse_critique(raw: str) -> tuple[bool, str]:
    text = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(text)
        return bool(data.get("approved", False)), str(data.get("reason", ""))
    except (json.JSONDecodeError, AttributeError):
        return ("approv" in text.lower() and "not" not in text.lower()), text


def critic_node(state: OrchestratorState) -> dict:
    """Judge the tool outputs; approve or send the loop back to the planner."""
    retries = state.get("retries", 0) + 1
    results = state.get("results", [])
    summary = "\n".join(f"- {r['description']}: {r['tool_output']}" for r in results)

    llm = get_llm()
    reply = llm.invoke(
        [
            {"role": "system", "content": CRITIC_SYSTEM},
            {"role": "user", "content": f"Task: {state['task']}\nOutputs:\n{summary}"},
        ]
    )
    approved, reason = _parse_critique(reply.content)
    return {
        "approved": approved,
        "critique": reason,
        "retries": retries,
        "trace": [f"[critic] pass={retries} approved={approved} reason={reason!r}"],
    }


def route_after_critic(state: OrchestratorState) -> str:
    """Conditional edge: END on approval or retry cap, else back to planner."""
    if state.get("approved"):
        return "end"
    if state.get("retries", 0) >= state.get("max_retries", 3):
        return "end"
    return "retry"
