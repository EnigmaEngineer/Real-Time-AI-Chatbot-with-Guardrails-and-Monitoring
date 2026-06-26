"""Tests for the reporter — markdown render + summary aggregation."""

import json
from pathlib import Path

from src.redteam.classifier import ClassifiedResult
from src.redteam.generators.base import AttackRecord
from src.redteam.reporter import render_markdown, render_summary_json, summarise, write_report
from src.redteam.runner import RunResult


def _atk(category, attack_id="a1"):
    return AttackRecord(
        attack_id=attack_id, category=category,
        base_template="x", mutations=("raw",),
        payload="x", expected_breach_signals=(),
    )


def _run(category, ok=True, latency_ms=100.0, answer="ok"):
    return RunResult(
        attack=_atk(category),
        ok=ok,
        response={"answer": answer, "trace": [], "guardrail_violations": []},
        latency_ms=latency_ms,
        http_status=200 if ok else 500,
    )


def _verdict(category, kind, attack_id="a1"):
    return ClassifiedResult(
        attack_id=attack_id, category=category, verdict=kind,
        signals_fired=["target_token"] if kind == "breach" else [],
    )


def test_summarise_counts_correctly():
    pairs = [
        (_run("direct_override"), _verdict("direct_override", "clean_block")),
        (_run("direct_override"), _verdict("direct_override", "breach")),
        (_run("role_hijack"), _verdict("role_hijack", "clean_block")),
        (_run("role_hijack"), _verdict("role_hijack", "inconclusive")),
        (_run("role_hijack", ok=False), None),
    ]
    summary = summarise("http://x", pairs)
    assert summary.total == 5
    assert summary.clean_blocks == 2
    assert summary.breaches == 1
    assert summary.inconclusive == 1
    assert summary.errors == 1
    # Defense rate is clean / (clean + breach) — inconclusives and errors are excluded
    assert abs(summary.defense_rate - (2 / 3)) < 1e-6


def test_defense_rate_per_category():
    pairs = [
        (_run("direct_override"), _verdict("direct_override", "clean_block")),
        (_run("direct_override"), _verdict("direct_override", "clean_block")),
        (_run("direct_override"), _verdict("direct_override", "breach")),
    ]
    summary = summarise("http://x", pairs)
    st = summary.by_category["direct_override"]
    assert st.attempts == 3
    assert st.clean_blocks == 2
    assert st.breaches == 1
    assert abs(st.defense_rate - (2 / 3)) < 1e-6


def test_markdown_contains_key_sections():
    pairs = [
        (_run("direct_override"), _verdict("direct_override", "clean_block")),
        (_run("direct_override"), _verdict("direct_override", "breach")),
    ]
    summary = summarise("http://x", pairs)
    md = render_markdown(summary, pairs)
    assert "# Red-team report" in md
    assert "Defense rate" in md
    assert "Defense rate by category" in md
    assert "direct_override" in md
    assert "Breaches" in md


def test_json_summary_is_machine_readable():
    pairs = [
        (_run("role_hijack"), _verdict("role_hijack", "clean_block")),
    ]
    summary = summarise("http://x", pairs)
    js = render_summary_json(summary)
    obj = json.loads(js)
    assert obj["target_url"] == "http://x"
    assert obj["total"] == 1
    assert obj["by_category"]["role_hijack"]["clean_blocks"] == 1


def test_write_report_creates_both_files(tmp_path: Path):
    pairs = [
        (_run("direct_override"), _verdict("direct_override", "clean_block")),
    ]
    summary = summarise("http://x", pairs)
    md_path, json_path = write_report(tmp_path / "reports", summary, pairs)
    assert md_path.exists()
    assert json_path.exists()
    assert md_path.read_text(encoding="utf-8").startswith("# Red-team report")
    json.loads(json_path.read_text(encoding="utf-8"))  # valid JSON
