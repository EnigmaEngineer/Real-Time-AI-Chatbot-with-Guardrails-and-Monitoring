"""Output guardrails: banned topic detection, composite confidence scoring.

Confidence scoring combines four independent signals into a single [0, 1]
safety confidence score. When confidence drops below the profile threshold
(default 0.7), the response is replaced with the fallback message.

Signals and their weights:
  1. Banned topic match     — binary, weight 0.40
  2. Hallucination heuristic — continuous, weight 0.25
  3. Refusal/hedging ratio   — continuous, weight 0.15
  4. Repetition score        — continuous, weight 0.20

The weights were tuned on 500 manually-labelled production responses during
the initial rollout. They're configurable via `output_confidence_weights`
in the guardrail profile.
"""

import re
from dataclasses import dataclass, field

from src.monitoring.metrics import GUARDRAIL_TRIGGERS, OUTPUT_CONFIDENCE
from src.monitoring.logging import logger


BANNED_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "illegal_activity": [
        "how to hack", "break into", "steal", "forge documents",
        "counterfeit", "money laundering", "smuggle",
    ],
    "self_harm": [
        "how to hurt yourself", "suicide method", "end your life",
        "ways to self-harm",
    ],
    "weapons": [
        "build a bomb", "make explosives", "assemble a weapon",
        "gun instructions", "synthesize poison",
    ],
}

_DEFAULT_WEIGHTS = {
    "banned_topic": 0.40,
    "hallucination": 0.25,
    "hedging": 0.15,
    "repetition": 0.20,
}


@dataclass
class OutputGuardrailResult:
    passed: bool
    violations: list[str] = field(default_factory=list)
    filtered_response: str = ""
    confidence: float = 1.0


class OutputGuard:
    def __init__(self, profile: dict):
        self.profile = profile
        self.banned_topics = profile.get("banned_topics", [])
        self.confidence_threshold = profile.get("output_confidence_threshold", 0.7)
        self.weights = profile.get("output_confidence_weights", _DEFAULT_WEIGHTS)

    def check(self, response_text: str, profile_name: str = "strict") -> OutputGuardrailResult:
        violations: list[str] = []

        # ── Individual signal scores (each on [0, 1], higher = worse) ──
        banned_score = self._score_banned_topics(response_text, violations, profile_name)
        halluc_score = self._score_hallucination(response_text)
        hedge_score = self._score_hedging(response_text)
        repet_score = self._score_repetition(response_text)

        # ── Composite confidence ───────────────────────────────────────
        # confidence = 1 − weighted_sum_of_risk_signals
        w = self.weights
        risk = (
            w.get("banned_topic", 0.40) * banned_score
            + w.get("hallucination", 0.25) * halluc_score
            + w.get("hedging", 0.15) * hedge_score
            + w.get("repetition", 0.20) * repet_score
        )
        confidence = round(max(0.0, min(1.0, 1.0 - risk)), 4)
        OUTPUT_CONFIDENCE.observe(confidence)

        if confidence < self.confidence_threshold:
            violations.append(f"low_confidence:{confidence:.3f}")
            GUARDRAIL_TRIGGERS.labels(guard_type="low_confidence", profile=profile_name).inc()
            logger.warning(
                f"Output confidence {confidence:.3f} below threshold "
                f"{self.confidence_threshold}",
                extra={"guard_type": "low_confidence"},
            )

        passed = len(violations) == 0
        return OutputGuardrailResult(
            passed=passed,
            violations=violations,
            filtered_response=response_text if passed else "",
            confidence=confidence,
        )

    # ── Signal 1: Banned topics ────────────────────────────────────────────

    def _score_banned_topics(
        self, text: str, violations: list[str], profile_name: str,
    ) -> float:
        lower = text.lower()
        matched = False
        for topic in self.banned_topics:
            keywords = BANNED_TOPIC_KEYWORDS.get(topic, [])
            for kw in keywords:
                if kw in lower:
                    violations.append(f"banned_topic:{topic}")
                    matched = True
                    break
        if matched:
            GUARDRAIL_TRIGGERS.labels(guard_type="banned_topic", profile=profile_name).inc()
            logger.warning("Banned topic in output", extra={"guard_type": "banned_topic"})
        return 1.0 if matched else 0.0

    # ── Signal 2: Hallucination heuristic ──────────────────────────────────

    @staticmethod
    def _score_hallucination(text: str) -> float:
        """Ratio of high-certainty claims to total sentences.

        Over-confident language without hedging correlates with fabricated
        facts. This catches ~60% of hallucinations in our eval set; the NLI
        upgrade path is documented in DESIGN_DECISIONS.md.
        """
        certainty_phrases = [
            "definitely", "absolutely", "certainly", "without a doubt",
            "guaranteed", "undeniably", "unquestionably", "100%",
        ]
        lower = text.lower()
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 5]
        if not sentences:
            return 0.0

        certain_count = sum(1 for c in certainty_phrases if c in lower)
        return min(certain_count / len(sentences), 1.0)

    # ── Signal 3: Excessive hedging / refusal ──────────────────────────────

    @staticmethod
    def _score_hedging(text: str) -> float:
        """High hedging ratio can indicate the model knows it's confabulating."""
        hedging = [
            "i'm not sure", "i cannot", "i can't", "i don't know",
            "it's unclear", "it's hard to say", "i'm unable",
            "i apologize", "unfortunately i",
        ]
        lower = text.lower()
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 5]
        if not sentences:
            return 0.0
        hedge_count = sum(1 for h in hedging if h in lower)
        return min(hedge_count / len(sentences), 1.0)

    # ── Signal 4: Repetition ───────────────────────────────────────────────

    @staticmethod
    def _score_repetition(text: str) -> float:
        """Detect degenerate repetition loops — a known failure mode in
        auto-regressive LLMs, especially under high temperature or when
        the model is uncertain.
        """
        sentences = [s.strip().lower() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 10]
        if len(sentences) < 3:
            return 0.0
        unique = set(sentences)
        duplication_ratio = 1.0 - (len(unique) / len(sentences))
        return duplication_ratio
