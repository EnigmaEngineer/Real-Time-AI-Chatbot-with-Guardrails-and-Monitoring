"""Async attack runner. Fires AttackRecords at a live /chat/agent endpoint.

Safety: identifies itself in the User-Agent. Defaults to a low request
rate (2 req/s) so a misconfigured run can't accidentally DoS the target.
Refuses to run against hosts other than localhost or known Hugging Face
Spaces unless an explicit consent flag is set."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import httpx

from src.redteam.generators.base import AttackRecord


USER_AGENT = (
    "chatbot-platform-redteam/0.1 "
    "(+https://github.com/EnigmaEngineer/chatbot-platform)"
)


# Hosts the runner will hit without an explicit --consent flag. Everything
# else requires the caller to acknowledge they own / have permission for the
# target.
_AUTO_ALLOWED_HOSTS = (
    "localhost", "127.0.0.1", "::1",
    ".hf.space",      # Hugging Face Spaces subdomains
)


def is_target_consented(url: str, consent: bool) -> bool:
    """Returns True if the URL is on the auto-allow list OR the caller passed
    --consent. Used by run.py to refuse unknown-host runs by default."""
    if consent:
        return True
    host = urlparse(url).netloc.lower()
    if not host:
        return False
    return any(
        host == h or (h.startswith(".") and host.endswith(h))
        for h in _AUTO_ALLOWED_HOSTS
    )


@dataclass
class RunResult:
    attack: AttackRecord
    ok: bool
    response: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    latency_ms: float = 0.0
    http_status: int = 0


class AttackRunner:
    def __init__(
        self,
        *,
        base_url: str,
        concurrency: int = 2,
        per_request_timeout_s: float = 60.0,
        max_retries: int = 3,
        retry_backoff_s: float = 1.5,
        inter_request_sleep_s: float = 0.5,  # 2 req/s default, safe for free tiers
    ) -> None:
        self._base = base_url.rstrip("/")
        self._concurrency = concurrency
        self._timeout = per_request_timeout_s
        self._max_retries = max_retries
        self._backoff = retry_backoff_s
        self._inter = inter_request_sleep_s

    # ── public API ──────────────────────────────────────────────────────

    async def run_all(self, attacks: list[AttackRecord]) -> list[RunResult]:
        sem = asyncio.Semaphore(self._concurrency)
        async with httpx.AsyncClient(
            timeout=self._timeout,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            await self._wake(client)
            tasks = [self._run_one(client, sem, a) for a in attacks]
            return await asyncio.gather(*tasks)

    # ── internals ───────────────────────────────────────────────────────

    async def _wake(self, client: httpx.AsyncClient) -> None:
        """HF Spaces sleep after ~48h. First request can take 30s+ to wake.
        Hit /health and wait until it returns 200."""
        deadline = time.monotonic() + 90.0
        while time.monotonic() < deadline:
            try:
                resp = await client.get(f"{self._base}/health")
                if resp.status_code == 200:
                    return
            except (httpx.RequestError, httpx.TimeoutException):
                pass
            await asyncio.sleep(3.0)
        raise RuntimeError(f"target {self._base} did not come up within 90s")

    async def _run_one(
        self,
        client: httpx.AsyncClient,
        sem: asyncio.Semaphore,
        attack: AttackRecord,
    ) -> RunResult:
        async with sem:
            await asyncio.sleep(self._inter)
            return await self._post_with_retry(client, attack)

    async def _post_with_retry(
        self, client: httpx.AsyncClient, attack: AttackRecord
    ) -> RunResult:
        last_err = ""
        for attempt in range(self._max_retries):
            t0 = time.monotonic()
            try:
                resp = await client.post(
                    f"{self._base}/chat/agent",
                    json={"message": attack.payload, "profile": "default"},
                    headers={"Content-Type": "application/json"},
                )
                latency = (time.monotonic() - t0) * 1000

                if resp.status_code == 200:
                    return RunResult(
                        attack=attack, ok=True,
                        response=resp.json(),
                        latency_ms=latency,
                        http_status=200,
                    )

                # 4xx that isn't 429 means we tried something invalid; don't retry.
                if 400 <= resp.status_code < 500 and resp.status_code != 429:
                    return RunResult(
                        attack=attack, ok=False,
                        error=f"HTTP {resp.status_code}: {resp.text[:200]}",
                        latency_ms=latency,
                        http_status=resp.status_code,
                    )

                last_err = f"HTTP {resp.status_code}"
            except (httpx.RequestError, httpx.TimeoutException) as exc:
                last_err = f"{type(exc).__name__}: {exc}"

            await asyncio.sleep(self._backoff * (2**attempt))

        return RunResult(
            attack=attack, ok=False,
            error=f"max retries exceeded — {last_err}",
            http_status=0,
        )
