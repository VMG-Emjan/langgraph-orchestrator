"""SQLite request tracing — the $0 stand-in for LangSmith.

One row per graph run: trace id, kind (ask/eval/orchestrate), timestamp and
the full JSON payload including the node-by-node trace. The DB lives on local
disk (ephemeral on free-tier hosts; traces reset on restart, which is fine
for a demo).
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

_DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "traces.db"


def _db_path() -> Path:
    return Path(os.environ.get("TRACE_DB", str(_DEFAULT_DB)))


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS traces ("
        "id TEXT PRIMARY KEY, kind TEXT NOT NULL, ts REAL NOT NULL, payload TEXT NOT NULL)"
    )
    return conn


def save_trace(trace_id: str, kind: str, payload: dict) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO traces (id, kind, ts, payload) VALUES (?, ?, ?, ?)",
            (trace_id, kind, time.time(), json.dumps(payload, ensure_ascii=False)),
        )


def get_trace(trace_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, kind, ts, payload FROM traces WHERE id = ?", (trace_id,)
        ).fetchone()
    if row is None:
        return None
    return {"trace_id": row[0], "kind": row[1], "ts": row[2], **json.loads(row[3])}
