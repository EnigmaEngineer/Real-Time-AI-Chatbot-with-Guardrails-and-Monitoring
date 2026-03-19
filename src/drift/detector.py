"""Drift detection with automated remediation.

Tracks six signals across input and output:
  Input:  query_length, query_topic, query_sentiment
  Output: response_length, refusal_rate, response_sentiment

When the KS test detects statistically significant drift that exceeds
the standard-deviation threshold, the detector:
  1. Persists the event to the drift_events SQLite table
  2. Fires a Slack webhook (if configured)
  3. Optionally shifts A/B traffic away from the drifted variant toward
     the baseline by adjusting live weights on the ABRouter

The traffic shift is gradual: each drift event reduces the drifted variant's
weight by `traffic_shift_step` (default 0.10).  The shift is bounded — the
variant weight never drops below `traffic_shift_min_weight` (default 0.10)
so it's never fully removed, preserving some traffic for monitoring recovery.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field

import httpx

from src.monitoring.metrics import DRIFT_SCORE, DRIFT_P_VALUE, DRIFT_ACTIONS
from src.monitoring.logging import logger
from src.drift.classifiers import TopicClassifier, SentimentScorer
from src.drift.event_store import DriftEventStore, DriftEvent


@dataclass
class DriftAlert:
    metric: str
    direction: str
    ks_statistic: float
    p_value: float
    reference_mean: float
    current_mean: float
    action_taken: str = ""
    timestamp: float = field(default_factory=time.time)


class DriftDetector:
    def __init__(self, config: dict):
        drift_cfg = config.get("drift", {})
        self.window_size = drift_cfg.get("window_size", 1000)
        self.reference_size = drift_cfg.get("reference_window_size", 5000)
        self.alpha = drift_cfg.get("ks_test_alpha", 0.05)
        self.std_threshold = drift_cfg.get("alert_std_threshold", 2.0)
        self.webhook_url = drift_cfg.get("alerting", {}).get("webhook_url", "")

        # Traffic shifting config
        shift_cfg = drift_cfg.get("traffic_shifting", {})
        self.shift_enabled = shift_cfg.get("enabled", True)
        self.shift_step = shift_cfg.get("step", 0.10)
        self.shift_min_weight = shift_cfg.get("min_weight", 0.10)
        self.baseline_variant = shift_cfg.get("baseline_variant", "control")

        # Data windows
        self._reference: dict[str, deque] = {}
        self._current: dict[str, deque] = {}
        self._alerts: list[DriftAlert] = []

        # Classifiers for enriched signals
        custom_topics = drift_cfg.get("custom_topics", None)
        self._topic_clf = TopicClassifier(custom_topics)
        self._sentiment = SentimentScorer()

        # Event store
        db_path = drift_cfg.get("event_db_path", "data/drift_events.db")
        self._event_store = DriftEventStore(db_path)

        # Refusal tracking (rolling counters for refusal rate calculation)
        self._refusal_window: deque[int] = deque(maxlen=self.window_size)
        self._refusal_ref: deque[int] = deque(maxlen=self.reference_size)

    # ── Recording API (called from the request path) ──────────────────

    def record(self, metric_name: str, value: float, direction: str = "input") -> None:
        """Record a scalar metric value for drift tracking."""
        key = f"{direction}:{metric_name}"
        if key not in self._reference:
            self._reference[key] = deque(maxlen=self.reference_size)
            self._current[key] = deque(maxlen=self.window_size)
        self._reference[key].append(value)
        self._current[key].append(value)

    def record_input(self, text: str) -> None:
        """Record all input-side drift signals from a user message."""
        self.record("query_length", float(len(text)), "input")
        self.record("query_topic", float(self._topic_clf.classify_index(text)), "input")
        self.record("query_sentiment", self._sentiment.score(text), "input")

    def record_output(self, text: str, was_refused: bool) -> None:
        """Record all output-side drift signals from a model response."""
        self.record("response_length", float(len(text)), "output")
        self.record("response_sentiment", self._sentiment.score(text), "output")

        refusal_val = 1 if was_refused else 0
        self._refusal_ref.append(refusal_val)
        self._refusal_window.append(refusal_val)

        # Compute refusal rate as a proportion and track it as a continuous metric
        if len(self._refusal_window) >= 20:
            rate = sum(self._refusal_window) / len(self._refusal_window)
            self.record("refusal_rate", rate, "output")

    # ── Drift checking ────────────────────────────────────────────────

    def check_all(self, ab_router=None) -> list[DriftAlert]:
        """Run KS test on all tracked metrics. Returns list of triggered alerts.

        If `ab_router` is provided and traffic shifting is enabled, drifted
        variants will have their weights reduced automatically.
        """
        alerts = []
        for key in list(self._current.keys()):
            direction, metric = key.split(":", 1)
            ref = list(self._reference[key])
            cur = list(self._current[key])

            if len(cur) < 30 or len(ref) < 30:
                continue

            ks_stat, p_value = self._ks_test(ref, cur)
            DRIFT_SCORE.labels(metric_name=metric, direction=direction).set(ks_stat)
            DRIFT_P_VALUE.labels(metric_name=metric, direction=direction).set(p_value)

            if p_value >= self.alpha:
                continue

            ref_mean = sum(ref) / len(ref)
            cur_mean = sum(cur) / len(cur)
            ref_std = _std(ref, ref_mean)

            if ref_std <= 0 or abs(cur_mean - ref_mean) <= self.std_threshold * ref_std:
                continue

            # Drift confirmed — determine action
            actions: list[str] = []

            # 1. Persist to drift_events table
            event = DriftEvent(
                metric=metric, direction=direction,
                ks_statistic=ks_stat, p_value=p_value,
                reference_mean=ref_mean, current_mean=cur_mean,
                action_taken="",  # filled below
            )

            # 2. Traffic shifting
            shifted_variant = ""
            if self.shift_enabled and ab_router is not None:
                shifted_variant = self._shift_traffic(ab_router)
                if shifted_variant:
                    actions.append(f"traffic_shifted:{shifted_variant}")
                    event.variant_affected = shifted_variant

            # 3. Webhook (async will be called by the CronJob; here we just mark it)
            if self.webhook_url:
                actions.append("webhook_queued")

            action_str = ",".join(actions) if actions else "logged_only"
            event.action_taken = action_str
            self._event_store.record(event)
            DRIFT_ACTIONS.labels(action=action_str.split(":")[0] if ":" in action_str else action_str).inc()

            alert = DriftAlert(
                metric=metric, direction=direction,
                ks_statistic=ks_stat, p_value=p_value,
                reference_mean=ref_mean, current_mean=cur_mean,
                action_taken=action_str,
            )
            alerts.append(alert)
            self._alerts.append(alert)

            logger.warning(
                f"Drift detected: {metric} ({direction}) KS={ks_stat:.4f} "
                f"p={p_value:.4f} action={action_str}",
                extra={"guard_type": "drift"},
            )

        return alerts

    # ── Traffic shifting ──────────────────────────────────────────────

    def _shift_traffic(self, ab_router) -> str:
        """Reduce weight of non-baseline variants. Returns the variant name
        that was shifted, or empty string if no shift was needed.
        """
        shifted = ""
        for exp_name, exp in ab_router.experiments.items():
            variants = exp.get("variants", {})
            if len(variants) < 2:
                continue

            for vname, vcfg in variants.items():
                if vname == self.baseline_variant:
                    continue
                current_weight = vcfg.get("weight", 0.5)
                if current_weight <= self.shift_min_weight:
                    continue  # already at minimum

                new_weight = max(self.shift_min_weight, current_weight - self.shift_step)
                delta = current_weight - new_weight
                vcfg["weight"] = new_weight

                # Give the freed weight to the baseline
                if self.baseline_variant in variants:
                    variants[self.baseline_variant]["weight"] = min(
                        1.0, variants[self.baseline_variant].get("weight", 0.5) + delta,
                    )

                shifted = vname
                logger.warning(
                    f"Traffic shifted: {exp_name}/{vname} weight "
                    f"{current_weight:.2f} → {new_weight:.2f} "
                    f"(baseline {self.baseline_variant} absorbs +{delta:.2f})",
                    extra={"experiment": exp_name, "variant": vname},
                )

        return shifted

    # ── Slack webhook ─────────────────────────────────────────────────

    async def send_alerts(self, alerts: list[DriftAlert]) -> None:
        """Send drift alerts to Slack webhook. Called by the CronJob."""
        if not alerts or not self.webhook_url:
            return

        blocks = []
        for a in alerts:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*{a.direction}/{a.metric}*\n"
                        f"KS={a.ks_statistic:.4f}  p={a.p_value:.6f}\n"
                        f"Ref μ={a.reference_mean:.2f} → Current μ={a.current_mean:.2f}\n"
                        f"Action: `{a.action_taken}`"
                    ),
                },
            })

        payload = {
            "text": f"🚨 Drift Alert: {len(alerts)} metric(s) drifted",
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": "🚨 Drift Detection Alert"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": f"*{len(alerts)} metric(s)* exceeded drift threshold"}},
                *blocks,
            ],
        }

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(self.webhook_url, json=payload)
                resp.raise_for_status()
                logger.info(f"Drift alert sent to Slack ({len(alerts)} alerts)")
        except Exception as exc:
            logger.error(f"Failed to send Slack drift alert: {exc}")

    # ── Accessors ─────────────────────────────────────────────────────

    def get_alerts(self) -> list[DriftAlert]:
        return list(self._alerts)

    def get_recent_events(self, hours: float = 24.0) -> list[DriftEvent]:
        return self._event_store.get_recent(hours)

    @property
    def topic_classifier(self) -> TopicClassifier:
        return self._topic_clf

    @property
    def sentiment_scorer(self) -> SentimentScorer:
        return self._sentiment

    def close(self) -> None:
        self._event_store.close()

    # ── KS test ───────────────────────────────────────────────────────

    @staticmethod
    def _ks_test(sample_a: list[float], sample_b: list[float]) -> tuple[float, float]:
        """Two-sample Kolmogorov-Smirnov test. Returns (statistic, p_value)."""
        a_sorted = sorted(sample_a)
        b_sorted = sorted(sample_b)
        n_a, n_b = len(a_sorted), len(b_sorted)

        all_values = sorted(set(a_sorted + b_sorted))
        max_diff = 0.0

        for v in all_values:
            cdf_a = _bisect_right(a_sorted, v) / n_a
            cdf_b = _bisect_right(b_sorted, v) / n_b
            max_diff = max(max_diff, abs(cdf_a - cdf_b))

        n_eff = math.sqrt(n_a * n_b / (n_a + n_b))
        lambda_val = (n_eff + 0.12 + 0.11 / n_eff) * max_diff
        p_value = 2.0 * sum(
            ((-1.0) ** (k - 1)) * math.exp(-2.0 * k * k * lambda_val * lambda_val)
            for k in range(1, 101)
        )
        p_value = max(0.0, min(1.0, p_value))
        return max_diff, p_value


def _bisect_right(sorted_list: list[float], value: float) -> int:
    lo, hi = 0, len(sorted_list)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_list[mid] <= value:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _std(values: list[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)
