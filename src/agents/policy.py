"""Per-tool policy engine for the agent coordinator."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from src.agents.tools.base import Tool, ToolResult
from src.agents.tools.web_fetch import _host_allowed
from src.monitoring.logging import logger
from src.monitoring.metrics import (
    AGENT_POLICY_VIOLATIONS,
    AGENT_TOOL_CALLS,
)


MAX_ARG_LENGTH_CHARS = 2000


# Default injection patterns. Production patches these from config.yaml so
# the same rules apply here and in the InputGuard.
_DEFAULT_INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all prior",
    "disregard your system prompt",
    "you are now",
    "act as if you have no restrictions",
    "reveal your system prompt",
    "system prompt",
]

_PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),       # SSN
    re.compile(r"\b(?:\d[ -]*?){13,16}\b"),     # credit card
]


class PolicyViolation(Exception):
    def __init__(self, tool: str, policy: str, detail: str) -> None:
        super().__init__(f"[{tool}] {policy}: {detail}")
        self.tool = tool
        self.policy = policy
        self.detail = detail


@dataclass
class PolicyDecision:
    allowed: bool
    violations: list[str]
    sanitized_args: dict[str, Any] | None = None
    sanitized_output: str | None = None


class PolicyEngine:
    def __init__(self, injection_patterns: list[str] | None = None) -> None:
        patterns = injection_patterns or _DEFAULT_INJECTION_PATTERNS
        self._injection_re = re.compile(
            "|".join(re.escape(p) for p in patterns), re.IGNORECASE
        )

        self._pre_checks: dict[str, Callable[[Tool, dict], dict]] = {
            "max_arg_length": _max_arg_length,
            "no_pii_in_args": _no_pii_in_args,
            "url_allowlist": _url_allowlist,
        }
        self._post_checks: dict[str, Callable[[Tool, str], str]] = {
            "injection_scan_output": self._injection_scan_output,
        }

    # ── public API ──────────────────────────────────────────────────────

    def guard_pre(
        self, tool: Tool, args: dict[str, Any], profile: str
    ) -> PolicyDecision:
        violations: list[str] = []
        sanitized = dict(args)

        for policy_name in tool.policies:
            check = self._pre_checks.get(policy_name)
            if check is None:
                continue
            try:
                sanitized = check(tool, sanitized) or sanitized
            except PolicyViolation as exc:
                violations.append(exc.policy)
                AGENT_POLICY_VIOLATIONS.labels(tool=tool.name, policy=exc.policy).inc()
                logger.warning(
                    "Agent policy blocked tool call",
                    extra={
                        "tool": tool.name,
                        "policy": exc.policy,
                        "detail": exc.detail,
                        "profile": profile,
                    },
                )

        if violations:
            AGENT_TOOL_CALLS.labels(tool=tool.name, status="blocked_pre").inc()
            return PolicyDecision(allowed=False, violations=violations)
        return PolicyDecision(allowed=True, violations=[], sanitized_args=sanitized)

    def guard_post(self, tool: Tool, result: ToolResult) -> PolicyDecision:
        if not result.ok:
            return PolicyDecision(allowed=True, violations=[], sanitized_output=result.output)

        violations: list[str] = []
        sanitized = result.output

        for policy_name in tool.policies:
            check = self._post_checks.get(policy_name)
            if check is None:
                continue
            try:
                sanitized = check(tool, sanitized) or sanitized
            except PolicyViolation as exc:
                violations.append(exc.policy)
                AGENT_POLICY_VIOLATIONS.labels(tool=tool.name, policy=exc.policy).inc()
                logger.warning(
                    "Agent policy blocked tool output",
                    extra={"tool": tool.name, "policy": exc.policy, "detail": exc.detail},
                )

        if violations:
            AGENT_TOOL_CALLS.labels(tool=tool.name, status="blocked_post").inc()
            return PolicyDecision(allowed=False, violations=violations)
        return PolicyDecision(allowed=True, violations=[], sanitized_output=sanitized)

    # ── post-check that needs engine state ──────────────────────────────

    def _injection_scan_output(self, tool: Tool, output: str) -> str:
        """Redact known prompt-injection phrases from tool output."""
        if not isinstance(output, str):
            return output
        if self._injection_re.search(output):
            redacted = self._injection_re.sub("[REDACTED:INJECTION]", output)
            AGENT_POLICY_VIOLATIONS.labels(
                tool=tool.name, policy="injection_scan_output"
            ).inc()
            logger.warning(
                "Redacted injection-like content from tool output",
                extra={"tool": tool.name, "policy": "injection_scan_output"},
            )
            return redacted
        return output


# ── pre-call check helpers ──────────────────────────────────────────────


def _max_arg_length(tool: Tool, args: dict[str, Any]) -> dict[str, Any]:
    for k, v in args.items():
        if isinstance(v, str) and len(v) > MAX_ARG_LENGTH_CHARS:
            raise PolicyViolation(
                tool.name,
                "max_arg_length",
                f"arg {k!r} length {len(v)} > {MAX_ARG_LENGTH_CHARS}",
            )
    return args


def _no_pii_in_args(tool: Tool, args: dict[str, Any]) -> dict[str, Any]:
    blob = json.dumps(args, default=str)
    for pat in _PII_PATTERNS:
        if pat.search(blob):
            raise PolicyViolation(
                tool.name, "no_pii_in_args", "argument matches PII pattern"
            )
    return args


def _url_allowlist(tool: Tool, args: dict[str, Any]) -> dict[str, Any]:
    url = args.get("url", "")
    if not isinstance(url, str) or not _host_allowed(url):
        raise PolicyViolation(
            tool.name, "url_allowlist", f"host not allowed: {url!r}"
        )
    return args
