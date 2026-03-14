"""SLO definitions and error-budget burn-rate calculation.

SLOs:
  1. Availability: 99.9% of requests return non-error responses
     - Error budget: 0.1% = 43.2 seconds per 12-hour window
  2. Latency:      95% of requests complete under 800ms
     - Burn rate > 1× means we exhaust the budget before the window ends

The burn-rate alerting strategy uses two windows (5m fast / 1h slow) to
balance sensitivity with noise.  See deploy/prometheus_alerts.yml for the
concrete rules that evaluate these metrics.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SLODefinition:
    name: str
    target: float          # e.g. 0.999
    window_hours: float    # e.g. 12
    metric_query: str      # PromQL for the error ratio
    description: str


AVAILABILITY_SLO = SLODefinition(
    name="availability",
    target=0.999,
    window_hours=12,
    metric_query=(
        "sum(rate(chatbot_slo_errors_total[%(window)s])) "
        "/ sum(rate(chatbot_requests_total[%(window)s]))"
    ),
    description="99.9% of requests return a non-error response",
)

LATENCY_SLO = SLODefinition(
    name="latency_p95",
    target=0.95,
    window_hours=12,
    metric_query=(
        "sum(rate(chatbot_slo_latency_violations_total[%(window)s])) "
        "/ sum(rate(chatbot_requests_total[%(window)s]))"
    ),
    description="95% of requests complete in under 800ms",
)

ALL_SLOS = [AVAILABILITY_SLO, LATENCY_SLO]

LATENCY_THRESHOLD_SECONDS = 0.8   # 800ms
ERROR_BUDGET_FRACTION = 1.0 - AVAILABILITY_SLO.target  # 0.001


def compute_burn_rate(error_ratio: float, slo: SLODefinition) -> float:
    """How fast we're burning the error budget.

    burn_rate = 1.0 means we'll exactly exhaust the budget over the window.
    burn_rate > 1.0 means we'll exhaust it early.
    """
    budget = 1.0 - slo.target
    if budget <= 0:
        return float("inf")
    return error_ratio / budget


def remaining_budget_seconds(error_ratio: float, slo: SLODefinition) -> float:
    """Seconds of budget remaining at the current error rate."""
    total_seconds = slo.window_hours * 3600
    budget_seconds = total_seconds * (1.0 - slo.target)
    consumed = total_seconds * error_ratio
    return max(0.0, budget_seconds - consumed)


def format_slo_report() -> str:
    """Human-readable SLO reference for README / runbooks."""
    lines = [
        "## Service Level Objectives",
        "",
    ]
    for slo in ALL_SLOS:
        budget_pct = (1.0 - slo.target) * 100
        budget_sec = slo.window_hours * 3600 * (1.0 - slo.target)
        lines.extend([
            f"### {slo.name}",
            "",
            f"**Target:** {slo.target * 100:.1f}%  ",
            f"**Window:** {slo.window_hours:.0f} hours  ",
            f"**Error budget:** {budget_pct:.2f}% = {budget_sec:.1f}s per window  ",
            f"**Description:** {slo.description}  ",
            "",
            f"**PromQL (error ratio):**",
            f"```",
            f"{slo.metric_query % {'window': '5m'}}",
            f"```",
            "",
        ])
    return "\n".join(lines)
