"""CLI entry point for a red-team run.

Usage:
    python -m src.redteam.run --url https://localhost:8000
    python -m src.redteam.run --url https://<hf-space>.hf.space --count 200
    python -m src.redteam.run --url https://elsewhere.example.com --consent

CI gates:
    --fail-below 0.95
    --fail-below-by-category url_exfil:1.0,tool_exfil:1.0,direct_override:0.6
    --use-metric clean_rate   # default; pass defense_rate for the looser metric
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
from pathlib import Path

from src.redteam.classifier import DefenseClassifier
from src.redteam.generators.library import build_library
from src.redteam.reporter import summarise, write_report
from src.redteam.runner import AttackRunner, is_target_consented


def _parse_per_category(raw: str) -> dict[str, float]:
    """Parse 'cat1:0.95,cat2:1.0' into {'cat1': 0.95, 'cat2': 1.0}."""
    out: dict[str, float] = {}
    if not raw:
        return out
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if ":" not in piece:
            raise ValueError(f"expected cat:threshold, got {piece!r}")
        cat, thr = piece.split(":", 1)
        out[cat.strip()] = float(thr.strip())
    return out


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Red-team the chatbot platform.")
    p.add_argument("--url", required=True, help="Base URL of the deployed target")
    p.add_argument("--count", type=int, default=0,
                   help="Cap total attacks (0 = use full library)")
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--report-dir", default="reports")
    p.add_argument(
        "--use-metric",
        choices=("clean_rate", "defense_rate"),
        default="clean_rate",
        help="Which metric the fail-below gates check. clean_rate "
             "treats inconclusive as not-defended (honest). defense_rate "
             "excludes inconclusive (looser).",
    )
    p.add_argument("--fail-below", type=float, default=0.0,
                   help="Exit non-zero if the chosen metric falls below this overall (0.0-1.0)")
    p.add_argument(
        "--fail-below-by-category",
        type=str,
        default="",
        help="Per-category thresholds, e.g. 'url_exfil:1.0,tool_exfil:1.0'. "
             "Any category below its threshold fails the run.",
    )
    p.add_argument(
        "--consent",
        action="store_true",
        help="Required when targeting a host that isn't localhost or a "
             "Hugging Face Space. Confirms you own / have permission to "
             "test the target.",
    )
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _select_metric(stats, name: str) -> float:
    return stats.clean_rate if name == "clean_rate" else stats.defense_rate


async def _main(args: argparse.Namespace) -> int:
    if not is_target_consented(args.url, args.consent):
        print(
            f"ERROR: refusing to target {args.url!r} without --consent. "
            "This script will fire hundreds of jailbreak attempts. Only run "
            "it against hosts you own or have explicit permission to test.",
            file=sys.stderr,
        )
        return 3

    per_cat_gates = _parse_per_category(args.fail_below_by_category)

    attacks = build_library()
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(attacks)
    if args.count and args.count < len(attacks):
        attacks = attacks[: args.count]

    print(
        f"running {len(attacks)} attacks against {args.url} "
        f"(concurrency={args.concurrency}, metric={args.use_metric})",
        file=sys.stderr,
    )

    runner = AttackRunner(base_url=args.url, concurrency=args.concurrency)
    classifier = DefenseClassifier()

    results = await runner.run_all(attacks)
    pairs = [
        (r, classifier.classify(r.attack, r.response) if r.ok else None)
        for r in results
    ]

    summary = summarise(args.url, pairs)
    md_path, json_path = write_report(Path(args.report_dir), summary, pairs)

    overall_metric = _select_metric(summary, args.use_metric)
    print(f"clean rate:        {summary.clean_rate * 100:.2f}%", file=sys.stderr)
    print(f"defense rate:      {summary.defense_rate * 100:.2f}%", file=sys.stderr)
    print(f"breaches:          {summary.breaches} ({summary.surprise_breaches} surprise)",
          file=sys.stderr)
    print(f"inconclusive:      {summary.inconclusive}", file=sys.stderr)
    print(f"errors:            {summary.errors}", file=sys.stderr)
    print(f"wrote:             {md_path}", file=sys.stderr)
    print(f"wrote:             {json_path}", file=sys.stderr)

    failures: list[str] = []
    if args.fail_below and overall_metric < args.fail_below:
        failures.append(
            f"overall {args.use_metric}={overall_metric:.4f} "
            f"< threshold {args.fail_below}"
        )
    for cat, threshold in per_cat_gates.items():
        st = summary.by_category.get(cat)
        if st is None:
            failures.append(f"category {cat!r} had no attempts but a gate was set")
            continue
        actual = _select_metric(st, args.use_metric)
        if actual < threshold:
            failures.append(
                f"category {cat!r} {args.use_metric}={actual:.4f} "
                f"< threshold {threshold}"
            )

    if failures:
        print("", file=sys.stderr)
        print("FAIL:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 2
    return 0


def main() -> int:
    args = _parse_args(sys.argv[1:])
    return asyncio.run(_main(args))


if __name__ == "__main__":
    sys.exit(main())
