"""Async LLM client with circuit breaker, retries, and mock mode."""

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator

import httpx

from src.monitoring.metrics import TOKEN_THROUGHPUT, ERROR_COUNTER, CIRCUIT_BREAKER_STATE
from src.monitoring.logging import logger


class CircuitState(Enum):
    CLOSED = 0
    OPEN = 1
    HALF_OPEN = 2


@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    _failure_count: int = 0
    _state: CircuitState = CircuitState.CLOSED
    _last_failure_time: float = 0.0

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._last_failure_time > self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
        return self._state

    def record_success(self) -> None:
        self._failure_count = 0
        self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN

    @property
    def is_call_allowed(self) -> bool:
        return self.state != CircuitState.OPEN


class MockLLMClient:
    """Deterministic mock for testing without a GPU. Recognizes the agent
    system prompt and emits ACTION/FINAL replies driven by keywords in the
    user message, so the demo works in mock mode."""

    async def generate(self, messages: list[dict], model: str, **kwargs) -> str:
        await asyncio.sleep(0.05)
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        if "ACTION:" in system and "FINAL:" in system:
            return self._agent_reply(messages)
        user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return f"Mock response to: {user_msg[:80]}"

    def _agent_reply(self, messages: list[dict]) -> str:
        import re as _re

        # First user turn is the actual query. Anything after is the loop.
        user_turns = [m for m in messages if m["role"] == "user"]
        if not user_turns:
            return "FINAL: I don't have anything to respond to."
        query = user_turns[0]["content"].lower()

        # Once we've seen any observation, wrap up.
        seen_obs = any(
            isinstance(m["content"], str) and m["content"].startswith("OBSERVATION")
            for m in messages
        )
        if seen_obs:
            last_obs = next(
                (m["content"] for m in reversed(messages)
                 if isinstance(m["content"], str) and m["content"].startswith("OBSERVATION")),
                "",
            )
            return f"FINAL: Based on the tool result, {last_obs[:200]}"

        if any(k in query for k in ("evil_tool", "shell_exec", "filesystem", "rm -rf")):
            return 'ACTION: {"tool": "evil_tool", "args": {"cmd": "whatever"}}'
        if "wikipedia" in query or " wiki " in f" {query} ":
            return ('ACTION: {"tool": "web_fetch", '
                    '"args": {"url": "https://en.wikipedia.org/wiki/Python_(programming_language)"}}')
        if "http://" in query or "https://" in query or "fetch " in query:
            url_match = _re.search(r"https?://\S+", query)
            url = url_match.group(0) if url_match else "https://example.com/"
            return f'ACTION: {{"tool": "web_fetch", "args": {{"url": "{url}"}}}}'
        if any(k in query for k in ("search docs", "find in", "lookup", "rag ")):
            q = query.replace("search docs", "").replace("find in", "").strip()[:100]
            return f'ACTION: {{"tool": "rag_search", "args": {{"query": "{q}", "top_k": 3}}}}'
        if any(k in query for k in ("calculate", "compute", "what is ")) or _re.search(
            r"\d+\s*[\+\-\*\/]\s*\d+", query
        ):
            expr_match = _re.search(r"[\d\.\+\-\*\/\(\)\s]+", query)
            expr = (expr_match.group(0).strip() if expr_match else "2+2")
            expr = expr.rstrip("+-*/. ") or "2+2"
            return f'ACTION: {{"tool": "calculator", "args": {{"expression": "{expr}"}}}}'

        return f"FINAL: {query.strip().capitalize()}. No tool needed."

    async def generate_stream(self, messages: list[dict], model: str, **kwargs) -> AsyncIterator[str]:
        response = await self.generate(messages, model, **kwargs)
        for word in response.split():
            yield word + " "
            await asyncio.sleep(0.02)


class LLMClient:
    def __init__(self, config: dict):
        llm_cfg = config["llm"]
        self.base_url = llm_cfg["base_url"].rstrip("/")
        self.api_key = llm_cfg.get("api_key", "")
        self.timeout = llm_cfg.get("timeout_seconds", 30)
        self.max_retries = llm_cfg.get("max_retries", 3)
        self.backoff_factor = llm_cfg.get("retry_backoff_factor", 0.5)
        self.default_model = llm_cfg["default_model"]
        self.fallback_model = llm_cfg.get("fallback_model", self.default_model)
        self.mock_mode = llm_cfg.get("mock_mode", False)

        cb_cfg = llm_cfg.get("circuit_breaker", {})
        self._breakers: dict[str, CircuitBreaker] = {}
        self._cb_failure_threshold = cb_cfg.get("failure_threshold", 5)
        self._cb_recovery_timeout = cb_cfg.get("recovery_timeout_seconds", 60)

        self._mock = MockLLMClient() if self.mock_mode else None
        self._client = httpx.AsyncClient(timeout=self.timeout) if not self.mock_mode else None

    def _get_breaker(self, model: str) -> CircuitBreaker:
        if model not in self._breakers:
            self._breakers[model] = CircuitBreaker(
                failure_threshold=self._cb_failure_threshold,
                recovery_timeout=self._cb_recovery_timeout,
            )
        return self._breakers[model]

    async def generate(self, messages: list[dict], model: str | None = None, **kwargs) -> str:
        model = model or self.default_model

        if self._mock:
            return await self._mock.generate(messages, model, **kwargs)

        breaker = self._get_breaker(model)
        if not breaker.is_call_allowed:
            CIRCUIT_BREAKER_STATE.labels(model=model).set(breaker.state.value)
            logger.warning("Circuit open, using fallback", extra={"model": model})
            if model != self.fallback_model:
                return await self.generate(messages, self.fallback_model, **kwargs)
            ERROR_COUNTER.labels(error_type="circuit_open").inc()
            raise RuntimeError("All LLM circuits open")

        last_exc = None
        for attempt in range(self.max_retries):
            try:
                response = await self._call_api(messages, model, stream=False, **kwargs)
                breaker.record_success()
                CIRCUIT_BREAKER_STATE.labels(model=model).set(0)
                content = response["choices"][0]["message"]["content"]
                usage = response.get("usage", {})
                TOKEN_THROUGHPUT.labels(direction="input", model=model).inc(usage.get("prompt_tokens", 0))
                TOKEN_THROUGHPUT.labels(direction="output", model=model).inc(usage.get("completion_tokens", 0))
                return content
            except (httpx.HTTPStatusError, httpx.ReadTimeout, httpx.ConnectError) as exc:
                last_exc = exc
                breaker.record_failure()
                CIRCUIT_BREAKER_STATE.labels(model=model).set(breaker.state.value)
                ERROR_COUNTER.labels(error_type=type(exc).__name__).inc()
                wait = self.backoff_factor * (2**attempt)
                logger.warning(
                    f"LLM call failed (attempt {attempt + 1}), retrying in {wait:.1f}s",
                    extra={"model": model},
                    exc_info=True,
                )
                await asyncio.sleep(wait)

        if model != self.fallback_model:
            logger.warning("Primary model exhausted retries, trying fallback", extra={"model": model})
            return await self.generate(messages, self.fallback_model, **kwargs)

        raise RuntimeError(f"LLM call failed after {self.max_retries} retries") from last_exc

    async def generate_stream(
        self, messages: list[dict], model: str | None = None, **kwargs
    ) -> AsyncIterator[str]:
        model = model or self.default_model

        if self._mock:
            async for chunk in self._mock.generate_stream(messages, model, **kwargs):
                yield chunk
            return

        breaker = self._get_breaker(model)
        if not breaker.is_call_allowed:
            CIRCUIT_BREAKER_STATE.labels(model=model).set(breaker.state.value)
            if model != self.fallback_model:
                async for chunk in self.generate_stream(messages, self.fallback_model, **kwargs):
                    yield chunk
                return
            raise RuntimeError("All LLM circuits open")

        try:
            async with self._client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": True,
                    **kwargs,
                },
                headers=self._headers(),
            ) as resp:
                resp.raise_for_status()
                breaker.record_success()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    import json

                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
        except Exception as exc:
            breaker.record_failure()
            ERROR_COUNTER.labels(error_type=type(exc).__name__).inc()
            raise

    async def _call_api(self, messages: list[dict], model: str, stream: bool = False, **kwargs) -> dict:
        payload = {"model": model, "messages": messages, "stream": stream, **kwargs}
        resp = await self._client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=self._headers(),
        )
        resp.raise_for_status()
        return resp.json()

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
