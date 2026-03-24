#!/usr/bin/env python3
"""Generate A/B test significance report from experiment records or feedback DB."""

import json
import sys
import time
from pathlib import Path

from src.config import load_config
from src.feedback.store import FeedbackStore
from src.abtesting.router import ExperimentRecord
from src.abtesting.stats import analyze_experiment


def records_from_feedback(store: FeedbackStore, experiment: str) -> list[ExperimentRecord]:
    entries = store.get_by_experiment(experiment)
    return [
        ExperimentRecord(
            experiment=e.experiment,
            variant=e.variant,
            user_id=e.user_id,
            latency_ms=0.0,
            feedback=e.rating,
            token_count=0,
            timestamp=e.timestamp,
        )
        for e in entries
    ]


def records_from_jsonl(path: str) -> list[ExperimentRecord]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            records.append(
                ExperimentRecord(
                    experiment=data.get("experiment", "backtest"),
                    variant=data.get("variant", "unknown"),
                    user_id=data.get("user_id", ""),
                    latency_ms=data.get("latency_ms", 0.0),
                    feedback=data.get("rating"),
                    token_count=data.get("token_count", 0),
                    timestamp=data.get("timestamp", time.time()),
                )
            )
    return records


def print_report(records: list[ExperimentRecord], significance_level: float = 0.05) -> None:
    if not records:
        print("No records to analyze.")
        return

    experiment = records[0].experiment
    print(f"\n{'='*80}")
    print(f"A/B TEST REPORT: {experiment}")
    print(f"{'='*80}")
    print(f"Total records: {len(records)}")

    by_variant: dict[str, list[ExperimentRecord]] = {}
    for r in records:
        by_variant.setdefault(r.variant, []).append(r)

    print(f"\nVariant Distribution:")
    for v, recs in sorted(by_variant.items()):
        fb = [r.feedback for r in recs if r.feedback is not None]
        pos = sum(1 for f in fb if f > 0)
        neg = sum(1 for f in fb if f < 0)
        approval = pos / len(fb) * 100 if fb else 0
        print(f"  {v}: n={len(recs)}, feedback={len(fb)} (👍{pos} 👎{neg}, {approval:.1f}% approval)")

    for metric in ("feedback", "latency_ms", "token_count"):
        results = analyze_experiment(records, metric=metric, significance_level=significance_level)
        if not results:
            continue

        print(f"\n--- {metric.upper()} ---")
        print(f"  {'Pair':<25} {'Mean A':>8} {'Mean B':>8} {'Δ':>8} {'p-value':>9} {'Result':<20}")
        print(f"  {'-'*78}")
        for r in results:
            delta = r.mean_a - r.mean_b
            if r.significant:
                result = f"✓ Winner: {r.recommended_winner}"
            else:
                result = "✗ Not significant"
            pair = f"{r.variant_a} vs {r.variant_b}"
            print(
                f"  {pair:<25} {r.mean_a:>8.3f} {r.mean_b:>8.3f} "
                f"{delta:>+8.3f} {r.p_value:>9.6f} {result:<20}"
            )

    print(f"\n{'='*80}")
    print(f"Significance level: α = {significance_level}")
    print(f"Statistical test: Welch's t-test (two-tailed)")
    print(f"{'='*80}\n")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m evaluation.ab_test_report <experiment_name | path.jsonl>")
        print("  experiment_name  — fetch from feedback DB")
        print("  path.jsonl       — load from JSONL file")
        sys.exit(1)

    source = sys.argv[1]
    significance = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05

    if Path(source).exists() and source.endswith(".jsonl"):
        records = records_from_jsonl(source)
    else:
        config = load_config()
        store = FeedbackStore(config)
        records = records_from_feedback(store, source)
        store.close()

    print_report(records, significance)


if __name__ == "__main__":
    main()
