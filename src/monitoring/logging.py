"""Structured JSON logging with per-request trace context.

Every log line includes:
  trace_id         — 16-hex-char request identifier for correlation
  user_id_hash     — SHA-256 prefix of user_id (privacy-safe, grep-able)
  variant_name     — A/B test variant ("" if no experiment)
  guardrail_action — "pass" | "block" | "redact" | "rate_limit" | ""

These four fields are always present (empty string when not applicable)
so downstream log pipelines can parse a single, stable schema.
"""

import hashlib
import logging
import json
import sys
from contextvars import ContextVar
from uuid import uuid4

# ── Per-request context (async-safe via contextvars) ───────────────────────

trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")
user_id_hash_var: ContextVar[str] = ContextVar("user_id_hash", default="")
variant_name_var: ContextVar[str] = ContextVar("variant_name", default="")
guardrail_action_var: ContextVar[str] = ContextVar("guardrail_action", default="")


def get_trace_id() -> str:
    tid = trace_id_var.get()
    if not tid:
        tid = uuid4().hex[:16]
        trace_id_var.set(tid)
    return tid


def set_trace_id(tid: str) -> None:
    trace_id_var.set(tid)


def set_user_context(user_id: str) -> None:
    """Hash user_id for logs — keeps it grep-able across a session but not PII."""
    hashed = hashlib.sha256(user_id.encode()).hexdigest()[:12]
    user_id_hash_var.set(hashed)


def set_variant(name: str) -> None:
    variant_name_var.set(name)


def set_guardrail_action(action: str) -> None:
    guardrail_action_var.set(action)


def reset_request_context() -> None:
    """Clear all per-request context vars (call at start of each request)."""
    trace_id_var.set("")
    user_id_hash_var.set("")
    variant_name_var.set("")
    guardrail_action_var.set("")


# ── Backward-compatible aliases ────────────────────────────────────────────
# Existing code uses these names; keep them working.

def get_correlation_id() -> str:
    return get_trace_id()


def set_correlation_id(cid: str) -> None:
    set_trace_id(cid)


# ── JSON Formatter ─────────────────────────────────────────────────────────

class JSONFormatter(logging.Formatter):
    """Emits one JSON object per line — no multi-line stack traces in prod."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            # Always-present trace context
            "trace_id": trace_id_var.get(""),
            "user_id_hash": user_id_hash_var.get(""),
            "variant_name": variant_name_var.get(""),
            "guardrail_action": guardrail_action_var.get(""),
        }

        if record.exc_info and record.exc_info[0]:
            entry["exception"] = self.formatException(record.exc_info)

        # Structured extras from `logger.info("msg", extra={...})`
        _EXTRA_KEYS = (
            "model", "endpoint", "user_id", "experiment", "variant",
            "guard_type", "latency_ms", "confidence", "injection_score",
            "tokens_in", "tokens_out", "cost_usd", "status",
        )
        for key in _EXTRA_KEYS:
            val = getattr(record, key, None)
            if val is not None:
                entry[key] = val

        return json.dumps(entry, default=str)


# ── Logger setup ───────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO") -> logging.Logger:
    root = logging.getLogger("chatbot")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        root.addHandler(handler)
    return root


logger = setup_logging()
