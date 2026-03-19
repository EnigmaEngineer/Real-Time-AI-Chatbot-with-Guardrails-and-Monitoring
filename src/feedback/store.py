"""Feedback collection and storage (SQLite-backed)."""

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from src.monitoring.metrics import FEEDBACK_COUNTER
from src.monitoring.logging import logger


@dataclass
class FeedbackEntry:
    conversation_id: str
    message_id: str
    user_id: str
    rating: int  # 1 = thumbs up, -1 = thumbs down
    experiment: str = ""
    variant: str = ""
    comment: str = ""
    timestamp: float = 0.0


class FeedbackStore:
    def __init__(self, config: dict):
        fb_cfg = config.get("feedback", {})
        db_path = fb_cfg.get("db_path", "data/feedback.db")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                rating INTEGER NOT NULL,
                experiment TEXT DEFAULT '',
                variant TEXT DEFAULT '',
                comment TEXT DEFAULT '',
                timestamp REAL NOT NULL
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_conv ON feedback(conversation_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_exp ON feedback(experiment, variant)"
        )
        self._conn.commit()

    def submit(self, entry: FeedbackEntry) -> int:
        entry.timestamp = entry.timestamp or time.time()
        cursor = self._conn.execute(
            """INSERT INTO feedback (conversation_id, message_id, user_id, rating, experiment, variant, comment, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.conversation_id,
                entry.message_id,
                entry.user_id,
                entry.rating,
                entry.experiment,
                entry.variant,
                entry.comment,
                entry.timestamp,
            ),
        )
        self._conn.commit()
        FEEDBACK_COUNTER.labels(
            rating="up" if entry.rating > 0 else "down",
            experiment=entry.experiment,
            variant=entry.variant,
        ).inc()
        logger.info(
            f"Feedback recorded: {entry.rating}",
            extra={"user_id": entry.user_id, "experiment": entry.experiment, "variant": entry.variant},
        )
        return cursor.lastrowid

    def get_by_experiment(self, experiment: str) -> list[FeedbackEntry]:
        rows = self._conn.execute(
            "SELECT * FROM feedback WHERE experiment = ? ORDER BY timestamp",
            (experiment,),
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def get_summary(self, experiment: str = "") -> dict:
        where = "WHERE experiment = ?" if experiment else ""
        params = (experiment,) if experiment else ()
        rows = self._conn.execute(
            f"""SELECT variant, 
                       COUNT(*) as total,
                       SUM(CASE WHEN rating > 0 THEN 1 ELSE 0 END) as positive,
                       SUM(CASE WHEN rating < 0 THEN 1 ELSE 0 END) as negative
                FROM feedback {where}
                GROUP BY variant""",
            params,
        ).fetchall()
        return {
            row["variant"]: {
                "total": row["total"],
                "positive": row["positive"],
                "negative": row["negative"],
                "approval_rate": row["positive"] / row["total"] if row["total"] > 0 else 0.0,
            }
            for row in rows
        }

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> FeedbackEntry:
        return FeedbackEntry(
            conversation_id=row["conversation_id"],
            message_id=row["message_id"],
            user_id=row["user_id"],
            rating=row["rating"],
            experiment=row["experiment"],
            variant=row["variant"],
            comment=row["comment"],
            timestamp=row["timestamp"],
        )

    def close(self) -> None:
        self._conn.close()
