"""Render the compiled StateGraph to assets/graph.png (Mermaid).

    python examples/render_graph.py

Falls back to writing assets/graph.mmd if PNG rendering (which needs network
access to the Mermaid renderer) is unavailable, so the topology is always
captured somewhere.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.graph import build_graph  # noqa: E402


def main() -> None:
    assets = pathlib.Path(__file__).resolve().parents[1] / "assets"
    assets.mkdir(exist_ok=True)
    graph = build_graph().get_graph()

    mmd_path = assets / "graph.mmd"
    mmd_path.write_text(graph.draw_mermaid(), encoding="utf-8")
    print("Wrote", mmd_path)

    try:
        png = graph.draw_mermaid_png()
        png_path = assets / "graph.png"
        png_path.write_bytes(png)
        print("Wrote", png_path)
    except Exception as exc:  # noqa: BLE001 - rendering is best-effort
        print("PNG render skipped (needs network):", exc)
        print("Mermaid source is available at", mmd_path)


if __name__ == "__main__":
    main()
