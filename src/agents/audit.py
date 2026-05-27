"""SQLite-backed audit log of agent tool calls. Mirrors src.feedback.store."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AuditRow:
    timestamp: float
    trace_id: str
    user_id: str
    tool: str
    args_json: str
    status: str  # ok | blocked_pre | blocked_post | error
    violations: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    output_preview: str = ""
    error: str = ""


class AuditStore:
    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS agent_audit (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL NOT NULL,
        trace_id TEXT NOT NULL,
        user_id TEXT NOT NULL,
        tool TEXT NOT NULL,
        args_json TEXT NOT NULL,
        status TEXT NOT NULL,
        violations_json TEXT NOT NULL,
        latency_ms REAL NOT NULL,
        output_preview TEXT NOT NULL,
        error TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_agent_audit_ts ON agent_audit(timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_agent_audit_status ON agent_audit(status);
    """

    def __init__(self, db_path: str | Path = "data/agent_audit.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.executescript(self._SCHEMA)
        self._conn.commit()
        self._lock = threading.Lock()

    def record(self, row: AuditRow) -> int:
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO agent_audit
                   (timestamp, trace_id, user_id, tool, args_json, status,
                    violations_json, latency_ms, output_preview, error)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    row.timestamp,
                    row.trace_id,
                    row.user_id,
                    row.tool,
                    row.args_json,
                    row.status,
                    json.dumps(row.violations),
                    row.latency_ms,
                    row.output_preview[:500],
                    row.error[:500],
                ),
            )
            self._conn.commit()
            return cur.lastrowid or 0

    def recent(self, limit: int = 50, status: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if status:
                cur = self._conn.execute(
                    """SELECT timestamp, trace_id, user_id, tool, args_json, status,
                              violations_json, latency_ms, output_preview, error
                       FROM agent_audit WHERE status = ? ORDER BY timestamp DESC LIMIT ?""",
                    (status, limit),
                )
            else:
                cur = self._conn.execute(
                    """SELECT timestamp, trace_id, user_id, tool, args_json, status,
                              violations_json, latency_ms, output_preview, error
                       FROM agent_audit ORDER BY timestamp DESC LIMIT ?""",
                    (limit,),
                )
            rows = cur.fetchall()
        return [
            {
                "timestamp": r[0],
                "trace_id": r[1],
                "user_id": r[2],
                "tool": r[3],
                "args": json.loads(r[4]),
                "status": r[5],
                "violations": json.loads(r[6]),
                "latency_ms": r[7],
                "output_preview": r[8],
                "error": r[9],
            }
            for r in rows
        ]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            cur = self._conn.execute(
                """SELECT status, COUNT(*) FROM agent_audit
                   WHERE timestamp > ? GROUP BY status""",
                (time.time() - 24 * 3600,),
            )
            by_status = dict(cur.fetchall())
            cur = self._conn.execute(
                """SELECT tool, COUNT(*) FROM agent_audit
                   WHERE timestamp > ? GROUP BY tool ORDER BY 2 DESC""",
                (time.time() - 24 * 3600,),
            )
            by_tool = dict(cur.fetchall())
        return {"by_status_24h": by_status, "by_tool_24h": by_tool}

    def close(self) -> None:
        with self._lock:
            self._conn.close()
