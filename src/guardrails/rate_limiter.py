"""Per-user rate limiting tied to guardrail violations.

Tracks violations per user in a sliding time window. After exceeding the
configured threshold (default: 3 violations within 10 minutes), the user
is temporarily banned for a cooldown period. Bans are in-memory with TTL
eviction — no persistent state required, so a pod restart clears all bans
(which is the desired behavior during incidents).
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock

from src.monitoring.metrics import RATE_LIMIT_BANS
from src.monitoring.logging import logger


@dataclass(frozen=True)
class RateLimitConfig:
    max_violations: int = 3
    window_seconds: float = 600.0  # 10 minutes
    ban_duration_seconds: float = 900.0  # 15 minutes


@dataclass
class _UserRecord:
    violation_timestamps: list[float] = field(default_factory=list)
    banned_until: float = 0.0


class ViolationRateLimiter:
    """Thread-safe per-user violation tracker with temporary bans."""

    def __init__(self, config: dict):
        rl_cfg = config.get("guardrails", {}).get("rate_limiting", {})
        self._cfg = RateLimitConfig(
            max_violations=rl_cfg.get("max_violations", 3),
            window_seconds=rl_cfg.get("window_seconds", 600.0),
            ban_duration_seconds=rl_cfg.get("ban_duration_seconds", 900.0),
        )
        self._users: dict[str, _UserRecord] = defaultdict(_UserRecord)
        self._lock = Lock()

    def is_banned(self, user_id: str) -> bool:
        """Check whether a user is currently banned."""
        with self._lock:
            record = self._users.get(user_id)
            if record is None:
                return False
            if record.banned_until > time.monotonic():
                return True
            # Ban expired — clear it so the record doesn't linger
            if record.banned_until > 0:
                record.banned_until = 0.0
                record.violation_timestamps.clear()
            return False

    def record_violation(self, user_id: str) -> bool:
        """Record a guardrail violation for user_id.

        Returns True if this violation triggers a new ban.
        """
        now = time.monotonic()
        with self._lock:
            record = self._users[user_id]

            # Already banned — don't extend the ban for violations during it
            if record.banned_until > now:
                return False

            record.violation_timestamps.append(now)
            cutoff = now - self._cfg.window_seconds
            record.violation_timestamps = [
                ts for ts in record.violation_timestamps if ts > cutoff
            ]

            if len(record.violation_timestamps) >= self._cfg.max_violations:
                record.banned_until = now + self._cfg.ban_duration_seconds
                record.violation_timestamps.clear()
                RATE_LIMIT_BANS.inc()
                logger.warning(
                    f"User temp-banned after {self._cfg.max_violations} violations "
                    f"(ban duration: {self._cfg.ban_duration_seconds}s)",
                    extra={"user_id": user_id},
                )
                return True

        return False

    def get_status(self, user_id: str) -> dict:
        """Return current rate-limit status for a user (for debug/API)."""
        now = time.monotonic()
        with self._lock:
            record = self._users.get(user_id)
            if record is None:
                return {"banned": False, "violations_in_window": 0, "ban_remaining_seconds": 0}

            banned = record.banned_until > now
            cutoff = now - self._cfg.window_seconds
            active = [ts for ts in record.violation_timestamps if ts > cutoff]
            return {
                "banned": banned,
                "violations_in_window": len(active),
                "ban_remaining_seconds": round(max(0.0, record.banned_until - now), 1),
            }

    def evict_expired(self) -> int:
        """Purge records for users with no active ban and no recent violations.

        Call periodically (e.g. every 5 minutes) to prevent unbounded growth.
        Returns the number of evicted entries.
        """
        now = time.monotonic()
        cutoff = now - self._cfg.window_seconds
        evicted = 0
        with self._lock:
            stale_keys = [
                uid
                for uid, rec in self._users.items()
                if rec.banned_until <= now
                and all(ts <= cutoff for ts in rec.violation_timestamps)
            ]
            for key in stale_keys:
                del self._users[key]
                evicted += 1
        return evicted
