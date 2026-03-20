"""Per-API-key request rate limiting (requests per minute).

Uses a sliding-window counter: each key tracks request timestamps in a
deque with a 60-second TTL. When the count exceeds the configured RPM,
the request is rejected with 429 Too Many Requests.

Separate from ViolationRateLimiter (which bans after guardrail violations).
This limits raw throughput per API key to prevent abuse and protect the
LLM backend from overload.
"""

import time
from collections import defaultdict, deque
from threading import Lock

from src.monitoring.metrics import RPM_REJECTED
from src.monitoring.logging import logger


class RPMRateLimiter:
    """Thread-safe per-key RPM limiter with sliding window."""

    def __init__(self, config: dict):
        rpm_cfg = config.get("server", {}).get("rate_limit", {})
        self._rpm = rpm_cfg.get("rpm", 60)
        self._enabled = rpm_cfg.get("enabled", False) and self._rpm > 0
        self._windows: dict[str, deque[float]] = defaultdict(lambda: deque())
        self._lock = Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def allow(self, key: str) -> bool:
        """Return True if the request should proceed, False if rate-limited."""
        if not self._enabled:
            return True

        now = time.monotonic()
        cutoff = now - 60.0

        with self._lock:
            window = self._windows[key]

            # Evict timestamps older than 60 seconds
            while window and window[0] <= cutoff:
                window.popleft()

            if len(window) >= self._rpm:
                RPM_REJECTED.inc()
                logger.warning(
                    f"RPM limit exceeded ({self._rpm}/min)",
                    extra={"status": "rpm_limited"},
                )
                return False

            window.append(now)
            return True

    def get_remaining(self, key: str) -> int:
        """Return how many requests are left in the current window."""
        if not self._enabled:
            return self._rpm

        now = time.monotonic()
        cutoff = now - 60.0

        with self._lock:
            window = self._windows.get(key, deque())
            active = sum(1 for ts in window if ts > cutoff)
            return max(0, self._rpm - active)

    def evict_expired(self) -> int:
        """Purge keys with no recent requests. Call periodically."""
        now = time.monotonic()
        cutoff = now - 60.0
        evicted = 0
        with self._lock:
            stale = [k for k, w in self._windows.items() if not w or w[-1] <= cutoff]
            for k in stale:
                del self._windows[k]
                evicted += 1
        return evicted
