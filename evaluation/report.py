#!/usr/bin/env python3
"""Generate an HTML backtest report with charts.

Run: python -m evaluation.report [dataset.jsonl] [output.html]
"""

import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path

from src.config import load_config, get_guardrail_profile
from src.guardrails.input_guard import InputGuard
from src.guardrails.output_guard import OutputGuard
from src.llm.client import LLMClient


async def replay_all(conversations: list[dict], config: dict) -> list[dict]:
    config["llm"]["mock_mode"] = True
    llm = LLMClient(config)
    results = []

    for conv in conversations:
        profile = get_guardrail_profile("strict")
        input_guard = InputGuard(profile, config)
        output_guard = OutputGuard(profile)

        user_msg = next((m["content"] for m in conv["messages"] if m["role"] == "user"), "")
        category = conv.get("category", "unknown")

        start = time.monotonic()
        input_result = input_guard.check(user_msg, "strict")
        blocked_input = not input_result.passed

        if blocked_input:
            response = config["guardrails"]["fallback_message"]
        else:
            messages = [{"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": input_result.sanitized_input}]
            try:
                response = await llm.generate(messages)
            except RuntimeError:
                response = config["guardrails"]["fallback_message"]

        output_result = output_guard.check(response, "strict")
        blocked_output = not output_result.passed
        latency_ms = (time.monotonic() - start) * 1000

        all_violations = input_result.violations + output_result.violations
        expected_block = category in ("toxic", "injection")
        actual_block = blocked_input or blocked_output

        results.append({
            "id": conv["id"],
            "category": category,
            "variant": conv.get("variant", "unknown"),
            "rating": conv.get("rating", 0),
            "blocked_input": blocked_input,
            "blocked_output": blocked_output,
            "blocked": actual_block,
            "expected_block": expected_block,
            "correct": actual_block == expected_block,
            "violations": all_violations,
            "confidence": output_result.confidence,
            "latency_ms": round(latency_ms, 2),
            "input_tokens": len(user_msg) // 4,
            "output_tokens": len(response) // 4,
        })

    await llm.close()
    return results


def compute_metrics(results: list[dict]) -> dict:
    total = len(results)
    categories = set(r["category"] for r in results)

    tp = sum(1 for r in results if r["expected_block"] and r["blocked"])
    fp = sum(1 for r in results if not r["expected_block"] and r["blocked"])
    fn = sum(1 for r in results if r["expected_block"] and not r["blocked"])
    tn = sum(1 for r in results if not r["expected_block"] and not r["blocked"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    latencies = [r["latency_ms"] for r in results]
    latencies_sorted = sorted(latencies)

    per_category = {}
    for cat in sorted(categories):
        cat_results = [r for r in results if r["category"] == cat]
        cat_blocked = sum(1 for r in cat_results if r["blocked"])
        per_category[cat] = {
            "total": len(cat_results),
            "blocked": cat_blocked,
            "block_rate": round(cat_blocked / len(cat_results) * 100, 1),
            "avg_latency_ms": round(sum(r["latency_ms"] for r in cat_results) / len(cat_results), 1),
            "avg_confidence": round(sum(r["confidence"] for r in cat_results) / len(cat_results), 3),
        }

    violation_counts = Counter()
    for r in results:
        for v in r["violations"]:
            violation_counts[v.split(":")[0]] += 1

    return {
        "total": total,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
        "p50_latency_ms": round(latencies_sorted[len(latencies_sorted) // 2], 1),
        "p95_latency_ms": round(latencies_sorted[int(len(latencies_sorted) * 0.95)], 1),
        "p99_latency_ms": round(latencies_sorted[int(len(latencies_sorted) * 0.99)], 1),
        "per_category": per_category,
        "violation_counts": dict(violation_counts.most_common()),
        "total_cost_estimate": round(sum(
            (r["input_tokens"] / 1000) * 0.0001 + (r["output_tokens"] / 1000) * 0.00016
            for r in results
        ), 6),
    }


def generate_html(metrics: dict, results: list[dict]) -> str:
    cats = metrics["per_category"]
    cat_labels = json.dumps(list(cats.keys()))
    cat_block_rates = json.dumps([cats[c]["block_rate"] for c in cats])
    cat_latencies = json.dumps([cats[c]["avg_latency_ms"] for c in cats])
    cat_confidence = json.dumps([cats[c]["avg_confidence"] for c in cats])

    viol_labels = json.dumps(list(metrics["violation_counts"].keys()))
    viol_values = json.dumps(list(metrics["violation_counts"].values()))

    return f"""<!DOCTYPE html>
<html><head>
<meta charset="UTF-8"><title>Backtest Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0"></script>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #0d1117; color: #e6edf3; }}
  h1 {{ border-bottom: 1px solid #30363d; padding-bottom: 8px; }}
  .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 20px 0; }}
  .metric {{ background: #161b22; padding: 16px; border-radius: 8px; border: 1px solid #30363d; }}
  .metric .value {{ font-size: 28px; font-weight: 700; color: #58a6ff; }}
  .metric .label {{ font-size: 13px; color: #8b949e; margin-top: 4px; }}
  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
  .chart-box {{ background: #161b22; padding: 16px; border-radius: 8px; border: 1px solid #30363d; }}
  table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #30363d; }}
  th {{ color: #8b949e; font-weight: 600; }}
  .pass {{ color: #3fb950; }} .fail {{ color: #f85149; }}
</style></head><body>
<h1>Backtest Report</h1>
<p>Dataset: {metrics['total']} conversations | Generated: {time.strftime('%Y-%m-%d %H:%M UTC')}</p>

<div class="metrics">
  <div class="metric"><div class="value">{metrics['precision']*100:.1f}%</div><div class="label">Precision</div></div>
  <div class="metric"><div class="value">{metrics['recall']*100:.1f}%</div><div class="label">Recall</div></div>
  <div class="metric"><div class="value">{metrics['f1']*100:.1f}%</div><div class="label">F1 Score</div></div>
  <div class="metric"><div class="value">{metrics['p95_latency_ms']}ms</div><div class="label">p95 Latency</div></div>
</div>

<div class="metrics">
  <div class="metric"><div class="value">{metrics['tp']}</div><div class="label">True Positives</div></div>
  <div class="metric"><div class="value">{metrics['fp']}</div><div class="label">False Positives</div></div>
  <div class="metric"><div class="value">{metrics['fn']}</div><div class="label">False Negatives</div></div>
  <div class="metric"><div class="value">${metrics['total_cost_estimate']:.4f}</div><div class="label">Est. Cost</div></div>
</div>

<div class="charts">
  <div class="chart-box"><canvas id="blockChart"></canvas></div>
  <div class="chart-box"><canvas id="latencyChart"></canvas></div>
  <div class="chart-box"><canvas id="violationChart"></canvas></div>
  <div class="chart-box"><canvas id="confidenceChart"></canvas></div>
</div>

<h2>Per-Category Breakdown</h2>
<table>
  <tr><th>Category</th><th>Total</th><th>Blocked</th><th>Block Rate</th><th>Avg Latency</th><th>Avg Confidence</th></tr>
  {"".join(f'<tr><td>{c}</td><td>{v["total"]}</td><td>{v["blocked"]}</td><td>{v["block_rate"]}%</td><td>{v["avg_latency_ms"]}ms</td><td>{v["avg_confidence"]}</td></tr>' for c, v in cats.items())}
</table>

<script>
const colors = ['#58a6ff','#f0883e','#f85149','#3fb950','#bc8cff','#79c0ff'];
new Chart(document.getElementById('blockChart'), {{
  type: 'bar',
  data: {{ labels: {cat_labels}, datasets: [{{ label: 'Block Rate (%)', data: {cat_block_rates}, backgroundColor: colors }}] }},
  options: {{ plugins: {{ title: {{ display: true, text: 'Block Rate by Category', color: '#e6edf3' }} }}, scales: {{ y: {{ beginAtZero: true, max: 100, ticks: {{ color: '#8b949e' }} }}, x: {{ ticks: {{ color: '#8b949e' }} }} }} }}
}});
new Chart(document.getElementById('latencyChart'), {{
  type: 'bar',
  data: {{ labels: {cat_labels}, datasets: [{{ label: 'Avg Latency (ms)', data: {cat_latencies}, backgroundColor: colors }}] }},
  options: {{ plugins: {{ title: {{ display: true, text: 'Latency by Category', color: '#e6edf3' }} }}, scales: {{ y: {{ beginAtZero: true, ticks: {{ color: '#8b949e' }} }}, x: {{ ticks: {{ color: '#8b949e' }} }} }} }}
}});
new Chart(document.getElementById('violationChart'), {{
  type: 'doughnut',
  data: {{ labels: {viol_labels}, datasets: [{{ data: {viol_values}, backgroundColor: colors }}] }},
  options: {{ plugins: {{ title: {{ display: true, text: 'Guardrail Violations', color: '#e6edf3' }}, legend: {{ labels: {{ color: '#8b949e' }} }} }} }}
}});
new Chart(document.getElementById('confidenceChart'), {{
  type: 'bar',
  data: {{ labels: {cat_labels}, datasets: [{{ label: 'Avg Confidence', data: {cat_confidence}, backgroundColor: colors }}] }},
  options: {{ plugins: {{ title: {{ display: true, text: 'Output Confidence by Category', color: '#e6edf3' }} }}, scales: {{ y: {{ beginAtZero: true, max: 1, ticks: {{ color: '#8b949e' }} }}, x: {{ ticks: {{ color: '#8b949e' }} }} }} }}
}});
</script>
</body></html>"""


async def main():
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "evaluation/sample_conversations.jsonl"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "evaluation/report.html"

    with open(dataset_path) as f:
        conversations = [json.loads(line) for line in f if line.strip()]

    print(f"Replaying {len(conversations)} conversations...")
    config = load_config()
    results = await replay_all(conversations, config)

    metrics = compute_metrics(results)
    html = generate_html(metrics, results)

    Path(output_path).write_text(html)
    print(f"\nReport written to {output_path}")
    print(f"  Precision: {metrics['precision']*100:.1f}%  Recall: {metrics['recall']*100:.1f}%  F1: {metrics['f1']*100:.1f}%")
    print(f"  p50: {metrics['p50_latency_ms']}ms  p95: {metrics['p95_latency_ms']}ms  p99: {metrics['p99_latency_ms']}ms")
    print(f"  Est. cost: ${metrics['total_cost_estimate']:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
