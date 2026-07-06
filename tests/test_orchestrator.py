"""Tests for the planner -> tool -> critic orchestration graph.

The DeepSeek LLM is never called: planner_node and critic_node are
monkeypatched, so the whole suite is deterministic and runs in CI with
no DEEPSEEK_API_KEY.
"""

from __future__ import annotations

import src.nodes as nodes
from src.graph import build_graph
from src.nodes import route_after_critic, tool_node


# --------------------------------------------------------------------------- #
# 1. tool_node produces one deterministic result per planned step
# --------------------------------------------------------------------------- #
def test_tool_node_runs_every_step():
    state = {"task": "t", "plan": ["a", "bb", "ccc"]}
    out = tool_node(state)
    assert len(out["results"]) == 3
    assert out["results"][0]["description"] == "a"
    assert out["results"][0]["tool_output"] == "executed<a>=OK(len=1)"


# --------------------------------------------------------------------------- #
# 2. conditional routing: approve ends, reject retries, cap ends
# --------------------------------------------------------------------------- #
def test_route_after_critic():
    assert route_after_critic({"approved": True, "retries": 1, "max_retries": 3}) == "end"
    assert route_after_critic({"approved": False, "retries": 1, "max_retries": 3}) == "retry"
    assert route_after_critic({"approved": False, "retries": 3, "max_retries": 3}) == "end"


# --------------------------------------------------------------------------- #
# 3. full graph loops back once then approves (LLM nodes mocked)
# --------------------------------------------------------------------------- #
def test_graph_retries_then_approves(monkeypatch):
    calls = {"plan": 0, "critic": 0}

    def fake_planner(state):
        calls["plan"] += 1
        return {"plan": ["step-x", "step-y"], "trace": [f"[planner] call {calls['plan']}"]}

    def fake_critic(state):
        calls["critic"] += 1
        approved = calls["critic"] >= 2  # reject first pass, approve second
        return {
            "approved": approved,
            "critique": "needs work" if not approved else "ok",
            "retries": state.get("retries", 0) + 1,
            "trace": [f"[critic] call {calls['critic']} approved={approved}"],
        }

    monkeypatch.setattr(nodes, "planner_node", fake_planner)
    monkeypatch.setattr(nodes, "critic_node", fake_critic)
    # Rebind the names graph.py imported at module load.
    import src.graph as g
    monkeypatch.setattr(g, "planner_node", fake_planner)
    monkeypatch.setattr(g, "critic_node", fake_critic)

    graph = build_graph()
    final = graph.invoke(
        {"task": "demo", "max_retries": 3, "retries": 0},
        config={"configurable": {"thread_id": "test-retry"}},
    )

    assert final["approved"] is True
    assert calls["plan"] == 2  # planner ran twice => loop-back happened
    assert calls["critic"] == 2


# --------------------------------------------------------------------------- #
# 4. fail_first injects an empty result on pass 1 only
# --------------------------------------------------------------------------- #
def test_fail_first_injects_empty_on_first_pass_only():
    plan = ["a", "b"]
    first = tool_node({"plan": plan, "fail_first": True, "retries": 0})
    assert first["results"][0]["tool_output"] == ""  # step 0 failed
    assert first["results"][1]["tool_output"] != ""

    second = tool_node({"plan": plan, "fail_first": True, "retries": 1})
    assert all(r["tool_output"] for r in second["results"])  # recovered


# --------------------------------------------------------------------------- #
# 5. plan parser tolerates JSON fences and bullet fallbacks
# --------------------------------------------------------------------------- #
def test_parse_plan_variants():
    assert nodes._parse_plan('```json\n["a","b"]\n```') == ["a", "b"]
    assert nodes._parse_plan("- one\n- two") == ["one", "two"]
