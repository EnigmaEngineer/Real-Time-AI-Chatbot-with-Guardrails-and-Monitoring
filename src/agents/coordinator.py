"""Agent coordinator: ReAct loop with hard caps on iterations, wall clock,
observation size, and per-tool timeout."""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from src.agents.audit import AuditRow, AuditStore
from src.agents.policy import PolicyEngine
from src.agents.prompts import render_system_prompt, render_tool_observation
from src.agents.tools.registry import ToolRegistry
from src.monitoring.logging import logger
from src.monitoring.metrics import (
    AGENT_BLOCKED,
    AGENT_ITERATIONS,
    AGENT_TOOL_CALLS,
    AGENT_TOOL_LATENCY,
)


DEFAULT_MAX_ITERATIONS = 5
DEFAULT_MAX_WALL_SECONDS = 15.0
DEFAULT_MAX_OBS_CHARS = 8000
DEFAULT_TOOL_TIMEOUT_S = 10.0


_ACTION_RE = re.compile(
    r"^\s*ACTION:\s*(\{.*\})\s*$", re.DOTALL | re.MULTILINE | re.IGNORECASE
)
_FINAL_RE = re.compile(r"^\s*FINAL:\s*(.+)\s*$", re.DOTALL | re.MULTILINE | re.IGNORECASE)


@dataclass
class ToolCallTrace:
    iteration: int
    tool: str
    args: dict[str, Any]
    status: str
    violations: list[str]
    latency_ms: float
    output_preview: str
    error: str = ""


@dataclass
class AgentResult:
    answer: str
    trace: list[ToolCallTrace] = field(default_factory=list)
    iterations_used: int = 0
    terminated_reason: str = "final"
    latency_ms: float = 0.0


class _LLMReplyParseError(Exception):
    pass


def _parse_reply(text: str) -> tuple[str, dict | str]:
    m = _ACTION_RE.search(text)
    if m:
        try:
            return "action", json.loads(m.group(1))
        except json.JSONDecodeError as exc:
            raise _LLMReplyParseError(f"ACTION json invalid: {exc}") from exc
    m = _FINAL_RE.search(text)
    if m:
        return "final", m.group(1).strip()
    raise _LLMReplyParseError(
        f"reply contained neither ACTION nor FINAL. Got: {text[:200]!r}"
    )


class AgentCoordinator:
    def __init__(
        self,
        *,
        llm,
        registry: ToolRegistry,
        policy: PolicyEngine,
        audit: AuditStore | None = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        max_wall_seconds: float = DEFAULT_MAX_WALL_SECONDS,
        max_obs_chars: int = DEFAULT_MAX_OBS_CHARS,
        tool_timeout_s: float = DEFAULT_TOOL_TIMEOUT_S,
        model: str | None = None,
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._policy = policy
        self._audit = audit
        self._max_iter = max_iterations
        self._max_wall = max_wall_seconds
        self._max_obs = max_obs_chars
        self._tool_timeout = tool_timeout_s
        self._model = model

    async def run(
        self,
        *,
        user_message: str,
        profile: str = "default",
        trace_id: str = "",
        user_id: str = "anonymous",
    ) -> AgentResult:
        start = time.monotonic()
        deadline = start + self._max_wall
        tools_block = self._registry.describe_for_prompt(profile)
        system_prompt = render_system_prompt(tools_block, self._max_iter)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        trace: list[ToolCallTrace] = []
        total_obs = 0

        for i in range(1, self._max_iter + 1):
            if time.monotonic() > deadline:
                AGENT_BLOCKED.labels(reason="timeout").inc()
                AGENT_ITERATIONS.observe(i - 1)
                return AgentResult(
                    answer="(agent timed out before producing a final answer)",
                    trace=trace, iterations_used=i - 1,
                    terminated_reason="timeout",
                    latency_ms=(time.monotonic() - start) * 1000,
                )

            try:
                reply = await self._llm.generate(messages, model=self._model)
            except RuntimeError as exc:
                logger.error("LLM failed during agent loop", extra={"error": str(exc)})
                AGENT_BLOCKED.labels(reason="llm_error").inc()
                AGENT_ITERATIONS.observe(i - 1)
                return AgentResult(
                    answer="(agent could not reach the language model)",
                    trace=trace, iterations_used=i - 1,
                    terminated_reason="llm_error",
                    latency_ms=(time.monotonic() - start) * 1000,
                )

            try:
                kind, payload = _parse_reply(reply)
            except _LLMReplyParseError as exc:
                # If the LLM drifts off-format, return what it said rather
                # than burn another iteration re-prompting it.
                logger.warning("Could not parse agent reply, treating as final",
                               extra={"detail": str(exc)})
                AGENT_ITERATIONS.observe(i)
                return AgentResult(
                    answer=reply.strip(),
                    trace=trace, iterations_used=i,
                    terminated_reason="parse_error",
                    latency_ms=(time.monotonic() - start) * 1000,
                )

            if kind == "final":
                AGENT_ITERATIONS.observe(i)
                return AgentResult(
                    answer=str(payload),
                    trace=trace, iterations_used=i,
                    terminated_reason="final",
                    latency_ms=(time.monotonic() - start) * 1000,
                )

            action = payload
            tool_name = action.get("tool", "")
            args = action.get("args", {}) or {}
            tool = self._registry.get(tool_name)
            row = AuditRow(
                timestamp=time.time(), trace_id=trace_id, user_id=user_id,
                tool=tool_name, args_json=json.dumps(args, default=str),
                status="ok", violations=[], latency_ms=0.0,
                output_preview="", error="",
            )

            if tool is None or not self._registry.is_allowed(profile, tool_name):
                row.status = "blocked_pre"
                row.violations = ["unknown_or_disallowed_tool"]
                row.error = f"tool {tool_name!r} not available in profile {profile!r}"
                AGENT_TOOL_CALLS.labels(tool=tool_name or "<unknown>", status="blocked_pre").inc()
                self._record_audit(row)
                trace.append(_trace_from_row(i, row))
                obs = render_tool_observation(
                    tool_name or "<unknown>", False, "", row.error
                )
                messages.extend(_action_and_obs(action, obs))
                total_obs += len(obs)
                if total_obs > self._max_obs:
                    AGENT_BLOCKED.labels(reason="obs_overflow").inc()
                    break
                continue

            pre = self._policy.guard_pre(tool, args, profile)
            if not pre.allowed:
                row.status = "blocked_pre"
                row.violations = pre.violations
                row.error = f"policy blocked: {', '.join(pre.violations)}"
                self._record_audit(row)
                trace.append(_trace_from_row(i, row))
                obs = render_tool_observation(tool_name, False, "", row.error)
                messages.extend(_action_and_obs(action, obs))
                total_obs += len(obs)
                continue

            t0 = time.monotonic()
            try:
                result = await asyncio.wait_for(
                    tool.run(pre.sanitized_args or args), timeout=self._tool_timeout
                )
            except asyncio.TimeoutError:
                latency = (time.monotonic() - t0) * 1000
                row.status = "error"
                row.violations = ["tool_timeout"]
                row.error = f"tool {tool_name} exceeded {self._tool_timeout}s timeout"
                row.latency_ms = latency
                AGENT_TOOL_CALLS.labels(tool=tool_name, status="error").inc()
                self._record_audit(row)
                trace.append(_trace_from_row(i, row))
                obs = render_tool_observation(tool_name, False, "", row.error)
                messages.extend(_action_and_obs(action, obs))
                continue
            except Exception as exc:
                latency = (time.monotonic() - t0) * 1000
                row.status = "error"
                row.violations = ["tool_exception"]
                row.error = f"{type(exc).__name__}: {exc}"
                row.latency_ms = latency
                AGENT_TOOL_CALLS.labels(tool=tool_name, status="error").inc()
                self._record_audit(row)
                trace.append(_trace_from_row(i, row))
                obs = render_tool_observation(tool_name, False, "", row.error)
                messages.extend(_action_and_obs(action, obs))
                continue

            latency = (time.monotonic() - t0) * 1000
            AGENT_TOOL_LATENCY.labels(tool=tool_name).observe(latency / 1000.0)
            row.latency_ms = latency

            post = self._policy.guard_post(tool, result)
            if not post.allowed:
                row.status = "blocked_post"
                row.violations = post.violations
                row.error = f"output blocked: {', '.join(post.violations)}"
                row.output_preview = (result.output or "")[:200]
                self._record_audit(row)
                trace.append(_trace_from_row(i, row))
                obs = render_tool_observation(tool_name, False, "", row.error)
                messages.extend(_action_and_obs(action, obs))
                continue

            row.status = "ok"
            row.output_preview = (post.sanitized_output or result.output)[:200]
            AGENT_TOOL_CALLS.labels(tool=tool_name, status="ok").inc()
            self._record_audit(row)
            trace.append(_trace_from_row(i, row))
            obs = render_tool_observation(
                tool_name, True, post.sanitized_output or result.output
            )
            messages.extend(_action_and_obs(action, obs))
            total_obs += len(obs)
            if total_obs > self._max_obs:
                AGENT_BLOCKED.labels(reason="obs_overflow").inc()
                break

        AGENT_BLOCKED.labels(reason="max_iterations").inc()
        AGENT_ITERATIONS.observe(self._max_iter)
        return AgentResult(
            answer="(agent reached the iteration cap without a final answer)",
            trace=trace, iterations_used=self._max_iter,
            terminated_reason="max_iterations",
            latency_ms=(time.monotonic() - start) * 1000,
        )

    def _record_audit(self, row: AuditRow) -> None:
        if self._audit is None:
            return
        try:
            self._audit.record(row)
        except Exception as exc:
            logger.warning("Audit log write failed", extra={"error": str(exc)})


def _action_and_obs(action: dict, obs: str) -> list[dict[str, str]]:
    return [
        {"role": "assistant", "content": f"ACTION: {json.dumps(action)}"},
        {"role": "user", "content": obs},
    ]


def _trace_from_row(iteration: int, row: AuditRow) -> ToolCallTrace:
    return ToolCallTrace(
        iteration=iteration,
        tool=row.tool,
        args=json.loads(row.args_json),
        status=row.status,
        violations=list(row.violations),
        latency_ms=row.latency_ms,
        output_preview=row.output_preview,
        error=row.error,
    )
