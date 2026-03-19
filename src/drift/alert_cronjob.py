#!/usr/bin/env python3
"""Drift alerting CronJob — runs hourly in Kubernetes.

Workflow:
  1. GET /drift/status from the chatbot API (triggers KS tests + auto-actions)
  2. If alerts were returned, send a formatted Slack message via webhook
  3. Log a summary and exit with code 1 if critical drift was detected

The API's check_all() already handles:
  - Persisting events to the drift_events table
  - Gradually shifting traffic away from drifted variants
This script adds the external notification layer.
"""

import asyncio
import os
import sys
import json

import httpx


API_URL = os.environ.get("CHATBOT_API_URL", "http://localhost:8000")
WEBHOOK_URL = os.environ.get("DRIFT_ALERT_WEBHOOK", "")
KS_CRITICAL_THRESHOLD = float(os.environ.get("DRIFT_KS_CRITICAL", "0.15"))


async def run() -> int:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{API_URL}/drift/status")
        resp.raise_for_status()
        data = resp.json()

    alerts = data.get("alerts", [])
    events = data.get("recent_events", [])

    print(f"Drift check: {len(alerts)} alert(s), {len(events)} event(s) in last 24h")

    if not alerts:
        return 0

    critical = [a for a in alerts if a.get("ks_statistic", 0) > KS_CRITICAL_THRESHOLD]

    for a in alerts:
        severity = "CRITICAL" if a in critical else "WARNING"
        print(
            f"  [{severity}] {a['direction']}/{a['metric']}: "
            f"KS={a['ks_statistic']:.4f} p={a['p_value']:.6f} "
            f"action={a.get('action_taken', 'none')}"
        )

    if WEBHOOK_URL:
        await _send_slack(alerts, critical)

    return 1 if critical else 0


async def _send_slack(alerts: list[dict], critical: list[dict]) -> None:
    severity_emoji = "🔴" if critical else "🟡"
    header = f"{severity_emoji} Drift Alert: {len(alerts)} metric(s)"

    blocks = [
        {"type": "header", "text": {"type": "plain_text", "text": header}},
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*{len(critical)} critical*, {len(alerts) - len(critical)} warning"}},
    ]

    for a in alerts[:10]:  # Cap at 10 to avoid Slack payload limits
        is_crit = a in critical
        icon = "🔴" if is_crit else "🟡"
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"{icon} *{a['direction']}/{a['metric']}*\n"
                    f"KS={a['ks_statistic']:.4f}  p={a['p_value']:.6f}\n"
                    f"μ: {a.get('reference_mean', '?')} → {a.get('current_mean', '?')}\n"
                    f"Action: `{a.get('action_taken', 'none')}`"
                ),
            },
        })

    payload = {"text": header, "blocks": blocks}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(WEBHOOK_URL, json=payload)
            resp.raise_for_status()
        print(f"Slack alert sent ({len(alerts)} alerts)")
    except Exception as exc:
        print(f"Failed to send Slack alert: {exc}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
