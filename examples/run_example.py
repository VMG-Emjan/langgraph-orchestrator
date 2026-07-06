"""End-to-end run of the LangGraph orchestrator against a real DeepSeek LLM.

Usage:
    export DEEPSEEK_API_KEY=sk-...
    python examples/run_example.py

Writes an execution trace to logs/ and prints the final state.
"""

from __future__ import annotations

import datetime as _dt
import pathlib
import sys

# Ensure Unicode prints on any console, incl. Windows cp1254.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Make ``src`` importable when run as a plain script.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.graph import build_graph  # noqa: E402


def main() -> None:
    task = (
        "Research the benefits of LangGraph for multi-agent orchestration "
        "and outline a short blog post."
    )

    graph = build_graph()
    config = {"configurable": {"thread_id": "example-1"}}
    initial = {"task": task, "max_retries": 3, "retries": 0}

    final = graph.invoke(initial, config=config)

    trace = final.get("trace", [])
    logs_dir = pathlib.Path(__file__).resolve().parents[1] / "logs"
    logs_dir.mkdir(exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_path = logs_dir / f"run_{stamp}.log"
    log_path.write_text("\n".join(trace), encoding="utf-8")

    print("=" * 70)
    print("TASK:", task)
    print("=" * 70)
    print("PLAN:", final.get("plan"))
    print("-" * 70)
    print("EXECUTION TRACE (planner -> tool -> critic loop):")
    for line in trace:
        print(" ", line)
    print("-" * 70)
    print("APPROVED:", final.get("approved"), "| PASSES:", final.get("retries"))
    print("CRITIQUE:", final.get("critique"))
    print("Trace written to:", log_path)


if __name__ == "__main__":
    main()
