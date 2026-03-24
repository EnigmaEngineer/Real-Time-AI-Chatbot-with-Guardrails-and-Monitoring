#!/usr/bin/env python3
"""Backtest: replay historical conversations and compute metrics per variant."""

import asyncio
import json
import sys
import time
from pathlib import Path

from src.config import load_config, get_guardrail_profile
from src.guardrails.input_guard import InputGuard
from src.guardrails.output_guard import OutputGuard
from src.llm.client import LLMClient
from src.abtesting.stats import analyze_experiment
from src.abtesting.router import ExperimentRecord


def load_conversations(path: str) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


async def replay_conversation(conv: dict, llm: LLMClient, config: dict) -> dict:
    """Replay a single conversation, applying guardrails and measuring metrics."""
    variant = conv.get("variant", "control")
    profile_name = "strict"
    profile = get_guardrail_profile(profile_name)
    input_guard = InputGuard(profile, config)
    output_guard = OutputGuard(profile)

    user_messages = [m for m in conv["messages"] if m["role"] == "user"]
    original_responses = [m for m in conv["messages"] if m["role"] == "assistant"]

    input_violations = []
    output_violations = []
    replayed_responses = []
    total_latency_ms = 0.0

    for i, user_msg in enumerate(user_messages):
        input_result = input_guard.check(user_msg["content"], profile_name)
        if not input_result.passed:
            input_violations.extend(input_result.violations)
            replayed_responses.append(config["guardrails"]["fallback_message"])
            continue

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input_result.sanitized_input},
        ]

        start = time.monotonic()
        try:
            response = await llm.generate(messages)
        except RuntimeError:
            response = config["guardrails"]["fallback_message"]
        elapsed_ms = (time.monotonic() - start) * 1000
        total_latency_ms += elapsed_ms

        out_result = output_guard.check(response, profile_name)
        if not out_result.passed:
            output_violations.extend(out_result.violations)
            response = config["guardrails"]["fallback_message"]

        replayed_responses.append(response)

    original_rating = conv.get("rating", 0)
    original_text = " ".join(m["content"] for m in original_responses)
    replayed_text = " ".join(replayed_responses)

    return {
        "id": conv["id"],
        "variant": variant,
        "user_id": conv.get("user_id", ""),
        "original_rating": original_rating,
        "input_violations": input_violations,
        "output_violations": output_violations,
        "total_violations": len(input_violations) + len(output_violations),
        "latency_ms": round(total_latency_ms, 2),
        "original_response_tokens": len(original_text.split()),
        "replayed_response_tokens": len(replayed_text.split()),
        "bleu_approx": _simple_bleu(original_text, replayed_text),
    }


def _simple_bleu(reference: str, hypothesis: str) -> float:
    """Unigram BLEU approximation — sufficient for backtesting triage."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not hyp_tokens or not ref_tokens:
        return 0.0

    ref_set = set(ref_tokens)
    matches = sum(1 for t in hyp_tokens if t in ref_set)
    precision = matches / len(hyp_tokens)
    brevity = min(1.0, len(hyp_tokens) / len(ref_tokens)) if ref_tokens else 0.0
    return round(precision * brevity, 4)


def print_results_table(results: list[dict]) -> None:
    by_variant: dict[str, list[dict]] = {}
    for r in results:
        by_variant.setdefault(r["variant"], []).append(r)

    header = f"{'Variant':<12} {'N':>4} {'Avg Rating':>11} {'Violations':>11} {'Avg Latency':>12} {'Avg BLEU':>9}"
    print("\n" + "=" * len(header))
    print("BACKTEST RESULTS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for variant in sorted(by_variant.keys()):
        group = by_variant[variant]
        n = len(group)
        avg_rating = sum(r["original_rating"] for r in group) / n
        total_violations = sum(r["total_violations"] for r in group)
        avg_latency = sum(r["latency_ms"] for r in group) / n
        avg_bleu = sum(r["bleu_approx"] for r in group) / n
        print(f"{variant:<12} {n:>4} {avg_rating:>11.2f} {total_violations:>11} {avg_latency:>10.1f}ms {avg_bleu:>9.4f}")

    print("-" * len(header))

    # Guardrail violation breakdown
    all_violations = []
    for r in results:
        all_violations.extend(r["input_violations"])
        all_violations.extend(r["output_violations"])

    if all_violations:
        print("\nGuardrail Violation Breakdown:")
        counts: dict[str, int] = {}
        for v in all_violations:
            key = v.split(":")[0]
            counts[key] = counts.get(key, 0) + 1
        for vtype, count in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {vtype}: {count}")


def print_ab_significance(results: list[dict]) -> None:
    """Build ExperimentRecords from backtest results and run significance test."""
    records = []
    for r in results:
        records.append(
            ExperimentRecord(
                experiment="backtest",
                variant=r["variant"],
                user_id=r["user_id"],
                latency_ms=r["latency_ms"],
                feedback=r["original_rating"] if r["original_rating"] != 0 else None,
                token_count=r["replayed_response_tokens"],
                timestamp=time.time(),
            )
        )

    for metric in ("feedback", "latency_ms", "token_count"):
        test_results = analyze_experiment(records, metric=metric)
        if not test_results:
            continue

        print(f"\nA/B Significance Test — {metric}:")
        print(f"  {'A':<10} {'B':<10} {'Mean A':>8} {'Mean B':>8} {'N_A':>5} {'N_B':>5} {'t-stat':>8} {'p-value':>9} {'Sig?':>5} {'Winner':<10}")
        print(f"  {'-'*85}")
        for tr in test_results:
            sig_marker = "YES" if tr.significant else "no"
            winner = tr.recommended_winner or "—"
            print(
                f"  {tr.variant_a:<10} {tr.variant_b:<10} "
                f"{tr.mean_a:>8.2f} {tr.mean_b:>8.2f} "
                f"{tr.n_a:>5} {tr.n_b:>5} "
                f"{tr.t_statistic:>8.4f} {tr.p_value:>9.6f} "
                f"{sig_marker:>5} {winner:<10}"
            )


async def main() -> None:
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "evaluation/sample_conversations.jsonl"
    if not Path(dataset_path).exists():
        print(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    config = load_config()
    config["llm"]["mock_mode"] = True  # always use mock for backtesting
    llm = LLMClient(config)

    conversations = load_conversations(dataset_path)
    print(f"Loaded {len(conversations)} conversations from {dataset_path}")

    results = []
    for conv in conversations:
        result = await replay_conversation(conv, llm, config)
        results.append(result)

    print_results_table(results)
    print_ab_significance(results)

    await llm.close()


if __name__ == "__main__":
    asyncio.run(main())
