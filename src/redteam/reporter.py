"""Generate the markdown + JSON summary of a red-team run."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import quantiles
from typing import Any

from src.redteam.classifier import ClassifiedResult
from src.redteam.runner import RunResult


@dataclass
class CategoryStats:
    attempts: int = 0
    clean_blocks: int = 0
    breaches: int = 0
    inconclusive: int = 0
    errors: int = 0
    surprise_breaches: int = 0  # breaches that fired unexpected signals

    @property
    def defense_rate(self) -> float:
        """clean / (clean + breach). Excludes inconclusive."""
        scored = self.clean_blocks + self.breaches
        if scored == 0:
            return 0.0
        return self.clean_blocks / scored

    @property
    def clean_rate(self) -> float:
        """clean / total successful runs. Treats inconclusive as not-defended.
        This is the honest metric for CI gates — it never hides uncertainty."""
        scored = self.clean_blocks + self.breaches + self.inconclusive
        if scored == 0:
            return 0.0
        return self.clean_blocks / scored


@dataclass
class RunSummary:
    target_url: str
    timestamp_iso: str
    total: int
    clean_blocks: int
    breaches: int
    inconclusive: int
    errors: int
    surprise_breaches: int  # breaches with at least one unexpected signal
    by_category: dict[str, CategoryStats]
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float

    @property
    def defense_rate(self) -> float:
        scored = self.clean_blocks + self.breaches
        if scored == 0:
            return 0.0
        return self.clean_blocks / scored

    @property
    def clean_rate(self) -> float:
        scored = self.clean_blocks + self.breaches + self.inconclusive
        if scored == 0:
            return 0.0
        return self.clean_blocks / scored


def summarise(
    target_url: str,
    pairs: list[tuple[RunResult, ClassifiedResult | None]],
) -> RunSummary:
    """Roll up the per-attack results into aggregate stats."""
    by_cat: dict[str, CategoryStats] = defaultdict(CategoryStats)
    latencies: list[float] = []

    total_clean = total_breach = total_inconc = total_err = total_surprise = 0

    for run, verdict in pairs:
        cat = run.attack.category
        st = by_cat[cat]
        st.attempts += 1

        if not run.ok:
            st.errors += 1
            total_err += 1
            continue

        latencies.append(run.latency_ms)

        if verdict is None:
            st.inconclusive += 1
            total_inconc += 1
        elif verdict.verdict == "clean_block":
            st.clean_blocks += 1
            total_clean += 1
        elif verdict.verdict == "breach":
            st.breaches += 1
            total_breach += 1
            if verdict.surprise_signals_fired:
                st.surprise_breaches += 1
                total_surprise += 1
        else:
            st.inconclusive += 1
            total_inconc += 1

    p50 = p95 = p99 = 0.0
    if latencies:
        latencies.sort()
        # statistics.quantiles needs n=100 for percentiles; or compute manually.
        p50 = _percentile(latencies, 50)
        p95 = _percentile(latencies, 95)
        p99 = _percentile(latencies, 99)

    return RunSummary(
        target_url=target_url,
        timestamp_iso=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        total=sum(s.attempts for s in by_cat.values()),
        clean_blocks=total_clean,
        breaches=total_breach,
        inconclusive=total_inconc,
        errors=total_err,
        surprise_breaches=total_surprise,
        by_category=dict(by_cat),
        latency_p50_ms=round(p50, 1),
        latency_p95_ms=round(p95, 1),
        latency_p99_ms=round(p99, 1),
    )


def _render_breach(lines: list, run: "RunResult", v: "ClassifiedResult") -> None:
    chain = "+".join(run.attack.mutations) or "raw"
    lines.append(f"### {run.attack.category} / {chain}")
    lines.append("")
    lines.append(f"- **Attack id:** `{run.attack.attack_id}`")
    lines.append(f"- **Expected signals:** "
                 f"{', '.join(v.expected_signals_fired) or '(none fired)'}")
    if v.surprise_signals_fired:
        lines.append(f"- **Surprise signals:** **{', '.join(v.surprise_signals_fired)}**")
    lines.append("- **Payload:**")
    lines.append("")
    lines.append(f"  > {run.attack.payload[:300].replace(chr(10), ' ')}")
    lines.append("")
    lines.append("- **Answer (first 300 chars):**")
    lines.append("")
    lines.append(
        f"  > {str(run.response.get('answer', ''))[:300].replace(chr(10), ' ')}"
    )
    lines.append("")


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


# ── markdown rendering ──────────────────────────────────────────────────


def render_markdown(
    summary: RunSummary,
    pairs: list[tuple[RunResult, ClassifiedResult | None]],
    max_breaches_shown: int = 10,
) -> str:
    lines: list[str] = []
    lines.append(f"# Red-team report — {summary.timestamp_iso.split('T')[0]}")
    lines.append("")
    lines.append(f"**Target:** `{summary.target_url}`  ")
    lines.append(f"**Run:** {summary.timestamp_iso}  ")
    lines.append("")
    lines.append(f"- Total attempts: **{summary.total}**")
    lines.append(
        f"- Clean rate (clean / total): **{summary.clean_rate * 100:.1f}%** "
        "← honest metric, inconclusives count against"
    )
    lines.append(
        f"- Defense rate (clean / scored): **{summary.defense_rate * 100:.1f}%** "
        "← excludes inconclusive"
    )
    lines.append(f"- Clean blocks: {summary.clean_blocks}")
    lines.append(f"- Breaches: **{summary.breaches}**"
                 + (f" ({summary.surprise_breaches} surprise)"
                    if summary.surprise_breaches else ""))
    lines.append(f"- Inconclusive: {summary.inconclusive}")
    lines.append(f"- Errors: {summary.errors}")
    lines.append("")
    lines.append(
        f"Latency p50 / p95 / p99: "
        f"**{summary.latency_p50_ms} / {summary.latency_p95_ms} / "
        f"{summary.latency_p99_ms}** ms"
    )
    lines.append("")

    # By-category table
    lines.append("## Defense rate by category")
    lines.append("")
    lines.append(
        "| Category | Attempts | Clean | Breach | Surprise | Inconclusive | Clean rate | Defense rate |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for cat, st in sorted(summary.by_category.items()):
        lines.append(
            f"| {cat} | {st.attempts} | {st.clean_blocks} | "
            f"{st.breaches} | {st.surprise_breaches} | {st.inconclusive} | "
            f"{st.clean_rate * 100:.1f}% | "
            f"{st.defense_rate * 100:.1f}% |"
        )
    lines.append("")

    # Surface surprise breaches first — they're the highest-value findings
    breaches = [(run, v) for run, v in pairs if v is not None and v.verdict == "breach"]
    surprises = [(r, v) for r, v in breaches if v.surprise_signals_fired]
    expected_only = [(r, v) for r, v in breaches if not v.surprise_signals_fired]

    if surprises:
        lines.append(f"## Surprise breaches ({len(surprises)})")
        lines.append("")
        lines.append(
            "_These attacks fired breach signals the attack template did NOT predict. "
            "Investigate first — they reveal attack classes you weren't testing for._"
        )
        lines.append("")
        for run, v in surprises[:max_breaches_shown]:
            _render_breach(lines, run, v)

    if expected_only:
        lines.append(
            f"## Breaches as predicted "
            f"({len(expected_only)}, showing up to {max_breaches_shown})"
        )
        lines.append("")
        for run, v in expected_only[:max_breaches_shown]:
            _render_breach(lines, run, v)

    if not breaches:
        lines.append("## Breaches")
        lines.append("")
        lines.append("None.")
        lines.append("")

    # Inconclusive — flagged for human review
    inconc = [
        (run, v) for run, v in pairs
        if v is not None and v.verdict == "inconclusive"
    ]
    if inconc:
        lines.append(f"## Inconclusive ({len(inconc)} total — needs human review)")
        lines.append("")
        for run, _ in inconc[:5]:
            lines.append(
                f"- `{run.attack.category}` / `{'+'.join(run.attack.mutations) or 'raw'}` "
                f"— `{run.attack.attack_id}`"
            )
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "Generated by `python -m src.redteam.run`. "
        "Source: <https://github.com/EnigmaEngineer/chatbot-platform>"
    )
    return "\n".join(lines)


# ── JSON summary, for CI gates and historical comparison ────────────────


def render_summary_json(summary: RunSummary) -> str:
    obj: dict[str, Any] = {
        "target_url": summary.target_url,
        "timestamp_iso": summary.timestamp_iso,
        "total": summary.total,
        "clean_blocks": summary.clean_blocks,
        "breaches": summary.breaches,
        "surprise_breaches": summary.surprise_breaches,
        "inconclusive": summary.inconclusive,
        "errors": summary.errors,
        "defense_rate": round(summary.defense_rate, 4),
        "clean_rate": round(summary.clean_rate, 4),
        "latency_p50_ms": summary.latency_p50_ms,
        "latency_p95_ms": summary.latency_p95_ms,
        "latency_p99_ms": summary.latency_p99_ms,
        "by_category": {
            cat: {
                "attempts": st.attempts,
                "clean_blocks": st.clean_blocks,
                "breaches": st.breaches,
                "surprise_breaches": st.surprise_breaches,
                "inconclusive": st.inconclusive,
                "errors": st.errors,
                "defense_rate": round(st.defense_rate, 4),
                "clean_rate": round(st.clean_rate, 4),
            }
            for cat, st in summary.by_category.items()
        },
    }
    return json.dumps(obj, indent=2)


def write_report(
    out_dir: Path,
    summary: RunSummary,
    pairs: list[tuple[RunResult, ClassifiedResult | None]],
) -> tuple[Path, Path]:
    """Write `redteam-YYYY-MM-DD.md` and `redteam-YYYY-MM-DD.json` to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    date = summary.timestamp_iso.split("T")[0]
    md_path = out_dir / f"redteam-{date}.md"
    json_path = out_dir / f"redteam-{date}.json"
    md_path.write_text(render_markdown(summary, pairs), encoding="utf-8")
    json_path.write_text(render_summary_json(summary), encoding="utf-8")
    return md_path, json_path
