"""Demonstrate a REAL planner -> tool -> critic loop-back against DeepSeek.

The tool is told to fail one step on the first pass (``fail_first=True``).
The real DeepSeek critic sees the empty result, rejects the work, and the
conditional edge routes back to the planner for a second pass — which then
succeeds and is approved.

    export DEEPSEEK_API_KEY=sk-...
    python examples/run_retry_example.py
"""

from __future__ import annotations

import datetime as _dt
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.graph import build_graph  # noqa: E402


def main() -> None:
    task = "Summarize why conditional routing matters in an agent graph."

    graph = build_graph()
    config = {"configurable": {"thread_id": "retry-demo-1"}}
    initial = {"task": task, "max_retries": 3, "retries": 0, "fail_first": True}

    final = graph.invoke(initial, config=config)

    trace = final.get("trace", [])
    logs_dir = pathlib.Path(__file__).resolve().parents[1] / "logs"
    logs_dir.mkdir(exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    (logs_dir / f"retry_{stamp}.log").write_text("\n".join(trace), encoding="utf-8")

    print("=" * 70)
    print("TASK:", task, "(fail_first=True → forces a retry)")
    print("=" * 70)
    print("EXECUTION TRACE (planner -> tool -> critic -> planner loop-back):")
    for line in trace:
        print(" ", line)
    print("-" * 70)
    print("APPROVED:", final.get("approved"), "| PASSES:", final.get("retries"))
    print("FINAL CRITIQUE:", final.get("critique"))


if __name__ == "__main__":
    main()
