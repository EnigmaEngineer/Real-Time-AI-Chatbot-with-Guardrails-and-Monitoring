"""Prometheus metrics for chatbot observability.

Instruments every code path that matters for SLO compliance:
  - Availability  SLO: 99.9% of requests return non-5xx  (error budget: 43.2s/12h)
  - Latency    SLO: 95% of requests complete under 800ms

Naming follows Prometheus conventions:
  chatbot_<subsystem>_<metric>_<unit>
"""

from prometheus_client import Counter, Histogram, Gauge, Info, Summary


# ── Request lifecycle ──────────────────────────────────────────────────────

REQUEST_LATENCY = Histogram(
    "chatbot_request_duration_seconds",
    "End-to-end request latency including guardrails + LLM",
    labelnames=["endpoint", "model", "status"],
    buckets=(0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0),
)

REQUEST_TOTAL = Counter(
    "chatbot_requests",
    "Total chat requests (success + failure), the denominator for violation-rate alerts",
    labelnames=["endpoint", "status"],
)

# ── SLO burn-rate indicators ───────────────────────────────────────────────
# These are the metrics that Prometheus alert rules evaluate directly.

SLO_LATENCY_VIOLATIONS = Counter(
    "chatbot_slo_latency_violations_total",
    "Requests exceeding the 800ms latency SLO target",
)

SLO_ERROR_TOTAL = Counter(
    "chatbot_slo_errors_total",
    "Requests counted as SLO failures (5xx, circuit-open fallbacks)",
)

# ── Token economics ───────────────────────────────────────────────────────

TOKEN_THROUGHPUT = Counter(
    "chatbot_tokens_total",
    "Total tokens processed",
    labelnames=["direction", "model"],
)

COST_PER_CONVERSATION = Histogram(
    "chatbot_cost_per_conversation_dollars",
    "Estimated cost in USD per conversation turn (input + output tokens × rate)",
    labelnames=["model"],
    buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5),
)

# ── Errors ─────────────────────────────────────────────────────────────────

ERROR_COUNTER = Counter(
    "chatbot_errors_total",
    "Total errors by type",
    labelnames=["error_type"],
)

# ── Guardrails ─────────────────────────────────────────────────────────────

GUARDRAIL_TRIGGERS = Counter(
    "chatbot_guardrail_triggers_total",
    "Guardrail trigger count",
    labelnames=["guard_type", "profile"],
)

PII_DETECTIONS = Counter(
    "chatbot_pii_detections_total",
    "PII entities detected by method and type",
    labelnames=["method", "entity_type"],
)

OUTPUT_CONFIDENCE = Histogram(
    "chatbot_output_confidence",
    "Distribution of output safety confidence scores",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

RATE_LIMIT_BANS = Counter(
    "chatbot_rate_limit_bans_total",
    "Number of temporary user bans triggered by violation rate limiting",
)

# ── Connections ────────────────────────────────────────────────────────────

ACTIVE_CONNECTIONS = Gauge(
    "chatbot_active_websocket_connections",
    "Number of active WebSocket connections",
)

# ── Feedback ───────────────────────────────────────────────────────────────

FEEDBACK_COUNTER = Counter(
    "chatbot_feedback_total",
    "User feedback events",
    labelnames=["rating", "experiment", "variant"],
)

# ── A/B testing ────────────────────────────────────────────────────────────

AB_ASSIGNMENT_COUNTER = Counter(
    "chatbot_ab_assignments_total",
    "A/B experiment assignments",
    labelnames=["experiment", "variant"],
)

AB_VARIANT_LATENCY = Histogram(
    "chatbot_ab_variant_duration_seconds",
    "Request latency broken down by A/B variant for performance comparison",
    labelnames=["experiment", "variant"],
    buckets=(0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 2.0, 5.0),
)

AB_VARIANT_FEEDBACK = Counter(
    "chatbot_ab_variant_feedback_total",
    "Feedback by variant (for Grafana A/B performance panel)",
    labelnames=["experiment", "variant", "rating"],
)

# ── Drift ──────────────────────────────────────────────────────────────────

DRIFT_SCORE = Gauge(
    "chatbot_drift_score",
    "Current drift KS-test statistic by metric",
    labelnames=["metric_name", "direction"],
)

DRIFT_P_VALUE = Gauge(
    "chatbot_drift_p_value",
    "Current drift KS-test p-value (alert fires when < 0.05)",
    labelnames=["metric_name", "direction"],
)

DRIFT_ACTIONS = Counter(
    "chatbot_drift_actions",
    "Automated actions taken in response to drift",
    labelnames=["action"],  # traffic_shifted, webhook_queued, logged_only
)

# ── Infrastructure ─────────────────────────────────────────────────────────

CIRCUIT_BREAKER_STATE = Gauge(
    "chatbot_circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=open, 2=half-open)",
    labelnames=["model"],
)

BUILD_INFO = Info("chatbot_build", "Build metadata")

RPM_REJECTED = Counter(
    "chatbot_rpm_rejected",
    "Requests rejected by per-API-key RPM rate limiting",
)


# ── Cost estimation helper ─────────────────────────────────────────────────
# Prices per 1K tokens.  Override via config if your provider differs.

_DEFAULT_COST_PER_1K: dict[str, dict[str, float]] = {
    "llama-3.1-8b":  {"input": 0.00010, "output": 0.00016},
    "llama-3.1-70b": {"input": 0.00059, "output": 0.00079},
}


def record_cost(model: str, input_tokens: int, output_tokens: int, cost_table: dict | None = None) -> float:
    """Estimate and record the dollar cost of a single conversation turn."""
    table = cost_table or _DEFAULT_COST_PER_1K
    rates = table.get(model, {"input": 0.0005, "output": 0.0007})
    cost = (input_tokens / 1000) * rates["input"] + (output_tokens / 1000) * rates["output"]
    COST_PER_CONVERSATION.labels(model=model).observe(cost)
    return cost
