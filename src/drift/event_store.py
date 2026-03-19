"""Persistent storage for drift events.

Every confirmed drift alert is written to the `drift_events` table with
the KS statistic, p-value, means, and the automated action taken
(e.g. "traffic_shifted", "webhook_sent", "logged_only").

The table is append-only — events are never updated. Downstream tooling
(Grafana SQLite plugin, or a simple query script) can read the history
for post-hoc analysis.
"""

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from src.monitoring.logging import logger


@dataclass
class DriftEvent:
    metric: str
    direction: str
    ks_statistic: float
    p_value: float
    reference_mean: float
    current_mean: float
    action_taken: str
    variant_affected: str = ""
    timestamp: float = 0.0


class DriftEventStore:
    def __init__(self, db_path: str = "data/drift_events.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS drift_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric TEXT NOT NULL,
                direction TEXT NOT NULL,
                ks_statistic REAL NOT NULL,
                p_value REAL NOT NULL,
                reference_mean REAL NOT NULL,
                current_mean REAL NOT NULL,
                action_taken TEXT NOT NULL,
                variant_affected TEXT DEFAULT '',
                timestamp REAL NOT NULL
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_drift_ts ON drift_events(timestamp)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_drift_metric ON drift_events(metric, direction)"
        )
        self._conn.commit()

    def record(self, event: DriftEvent) -> int:
        event.timestamp = event.timestamp or time.time()
        cursor = self._conn.execute(
            """INSERT INTO drift_events
               (metric, direction, ks_statistic, p_value, reference_mean,
                current_mean, action_taken, variant_affected, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event.metric, event.direction, event.ks_statistic,
                event.p_value, event.reference_mean, event.current_mean,
                event.action_taken, event.variant_affected, event.timestamp,
            ),
        )
        self._conn.commit()
        logger.info(
            f"Drift event persisted: {event.metric}/{event.direction} "
            f"action={event.action_taken}",
            extra={"guard_type": "drift"},
        )
        return cursor.lastrowid

    def get_recent(self, hours: float = 24.0, limit: int = 100) -> list[DriftEvent]:
        cutoff = time.time() - (hours * 3600)
        rows = self._conn.execute(
            """SELECT * FROM drift_events
               WHERE timestamp > ?
               ORDER BY timestamp DESC
               LIMIT ?""",
            (cutoff, limit),
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def get_by_metric(self, metric: str, direction: str, limit: int = 50) -> list[DriftEvent]:
        rows = self._conn.execute(
            """SELECT * FROM drift_events
               WHERE metric = ? AND direction = ?
               ORDER BY timestamp DESC LIMIT ?""",
            (metric, direction, limit),
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> DriftEvent:
        return DriftEvent(
            metric=row["metric"],
            direction=row["direction"],
            ks_statistic=row["ks_statistic"],
            p_value=row["p_value"],
            reference_mean=row["reference_mean"],
            current_mean=row["current_mean"],
            action_taken=row["action_taken"],
            variant_affected=row["variant_affected"],
            timestamp=row["timestamp"],
        )

    def close(self) -> None:
        self._conn.close()
