"""Lightweight text classifiers for drift signal enrichment.

Two classifiers:
  1. TopicClassifier  — maps text to one of N topic buckets using TF-IDF-style
     keyword overlap.  Zero-shot: no training data or GPU required.
  2. SentimentScorer  — maps text to [-1, 1] using positive/negative word lists.

Both are intentionally simple.  The drift detector needs a *consistent*
signal, not a perfect one — if the classifier is wrong but *consistently*
wrong in the same way, the KS test still detects the shift.  A transformer-
based zero-shot classifier (e.g. facebook/bart-large-mnli) would be more
accurate but adds ~300ms per call and a 1.6GB model download.

Upgrade path: set `drift.topic_classifier` to "transformer" in config.yaml
and the detector will import and use transformers.pipeline("zero-shot-classification").
"""

from __future__ import annotations

import math
import re

from src.monitoring.logging import logger


# ── Topic classification ───────────────────────────────────────────────────
# Each topic is defined by a bag of high-signal keywords.  A text's topic is
# the one with the highest normalized overlap.

_TOPIC_KEYWORDS: dict[str, set[str]] = {
    "technical": {
        "code", "api", "error", "bug", "deploy", "server", "database", "python",
        "function", "class", "debug", "compile", "runtime", "docker", "kubernetes",
        "git", "sql", "query", "endpoint", "json", "http", "log", "test",
    },
    "creative": {
        "story", "poem", "write", "creative", "imagine", "fiction", "character",
        "plot", "narrative", "metaphor", "song", "lyrics", "art", "design",
        "draw", "painting", "novel", "essay", "draft", "compose",
    },
    "factual": {
        "what", "when", "where", "who", "how", "explain", "define", "history",
        "science", "math", "calculate", "formula", "theory", "fact", "capital",
        "population", "distance", "temperature", "currency", "country",
    },
    "conversational": {
        "hello", "hi", "hey", "thanks", "please", "help", "sorry", "bye",
        "good", "morning", "afternoon", "evening", "fine", "okay", "sure",
        "yes", "no", "maybe", "chat", "talk",
    },
    "business": {
        "revenue", "strategy", "market", "customer", "sales", "product",
        "meeting", "report", "budget", "forecast", "investor", "growth",
        "profit", "stakeholder", "roadmap", "kpi", "quarterly", "board",
    },
}


class TopicClassifier:
    def __init__(self, custom_topics: dict[str, list[str]] | None = None):
        self._topics = dict(_TOPIC_KEYWORDS)
        if custom_topics:
            for topic, words in custom_topics.items():
                self._topics[topic] = set(w.lower() for w in words)

    def classify(self, text: str) -> str:
        """Return the most likely topic label for the given text."""
        words = set(re.findall(r"[a-z]+", text.lower()))
        if not words:
            return "unknown"

        best_topic = "unknown"
        best_score = 0.0

        for topic, keywords in self._topics.items():
            overlap = len(words & keywords)
            if overlap == 0:
                continue
            # Normalize by sqrt of keyword set size to avoid bias toward large bags
            score = overlap / math.sqrt(len(keywords))
            if score > best_score:
                best_score = score
                best_topic = topic

        return best_topic

    def classify_index(self, text: str) -> int:
        """Return a numeric index for the topic (stable across runs)."""
        topic = self.classify(text)
        ordered = sorted(self._topics.keys())
        if topic in ordered:
            return ordered.index(topic)
        return len(ordered)  # "unknown"

    @property
    def topic_names(self) -> list[str]:
        return sorted(self._topics.keys())


# ── Sentiment scoring ──────────────────────────────────────────────────────
# AFINN-style word lists (subset).  Returns float in [-1, 1].

_POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "wonderful", "fantastic", "love",
    "happy", "best", "perfect", "beautiful", "awesome", "helpful", "thank",
    "thanks", "brilliant", "outstanding", "superb", "enjoy", "pleased",
    "impressive", "positive", "success", "correct", "agree", "nice",
}

_NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "horrible", "hate", "worst", "ugly", "wrong",
    "fail", "error", "broken", "useless", "stupid", "angry", "disappointed",
    "frustrating", "annoyed", "poor", "waste", "slow", "crash", "bug",
    "negative", "disagree", "problem", "issue", "complaint",
}


class SentimentScorer:
    def __init__(
        self,
        positive_words: set[str] | None = None,
        negative_words: set[str] | None = None,
    ):
        self._positive = positive_words or _POSITIVE_WORDS
        self._negative = negative_words or _NEGATIVE_WORDS

    def score(self, text: str) -> float:
        """Return sentiment in [-1, 1].  0 = neutral."""
        words = re.findall(r"[a-z]+", text.lower())
        if not words:
            return 0.0

        pos = sum(1 for w in words if w in self._positive)
        neg = sum(1 for w in words if w in self._negative)
        total = pos + neg

        if total == 0:
            return 0.0

        raw = (pos - neg) / total  # range [-1, 1]
        return round(raw, 4)
