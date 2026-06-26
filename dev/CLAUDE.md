# CLAUDE.md — Project Context for Claude Code

Read this file first in every session. It tells you what this project is, how it's
built, the conventions you must follow, and the rules that keep the build green.

## What this project is

A production-grade real-time AI chatbot platform built as a portfolio piece. It
demonstrates AI/Data Engineering depth: guardrails, observability, drift detection,
A/B testing, and a RAG pipeline. The goal is a polished, credible, human-made-looking
project that supports job positioning in AI/Data Engineering.

Repository: https://github.com/EnigmaEngineer/chatbot-platform

## Tech stack

- Python 3.11+, FastAPI (REST + WebSocket + SSE)
- pytest for tests (currently 133 passing)
- ChromaDB + sentence-transformers (all-MiniLM-L6-v2) for RAG
- spaCy (NER), Detoxify (toxicity) for guardrails
- Prometheus + Grafana for observability
- Docker + docker-compose, Helm chart, K8s manifests
- Dependencies in pyproject.toml (`.[dev]` for dev extras)

## How to run things

```bash
# Run all tests (ALWAYS use these env vars)
LLM_MOCK_MODE=true CHATBOT_API_KEY= python -m pytest tests/ -v

# Run the API locally
LLM_MOCK_MODE=true uvicorn src.api.main:app --reload

# Full demo stack (Docker)
docker compose -f demo/docker-compose.yml up --build -d
```

## Project structure

```
src/
  api/main.py            FastAPI app — all endpoints, middleware, lifespan
  config.py              YAML loader with ${VAR:-default} env interpolation
  guardrails/
    input_guard.py       PII (regex + spaCy NER), injection scorer, toxicity
    output_guard.py      Confidence scoring, banned topics, hallucination
    rate_limiter.py      Per-user violation temp ban
    rpm_limiter.py       Per-API-key RPM sliding window
  llm/client.py          Async LLM client, circuit breaker, mock mode
  abtesting/
    router.py            SHA-256 deterministic variant assignment
    stats.py             Welch's t-test significance
  drift/
    detector.py          6-signal KS test, traffic shifting
    classifiers.py       Topic classifier, sentiment scorer
    event_store.py       Drift events SQLite persistence
    alert_cronjob.py     Hourly K8s CronJob alerting
  rag/
    chunker.py           Sentence-aware chunking with overlap
    ingest.py            PDF/TXT/MD ingestion + CLI
    vectorstore.py       ChromaDB wrapper, semantic search
  feedback/store.py      SQLite feedback collection
  monitoring/
    metrics.py           Prometheus counters/histograms/gauges
    logging.py           JSON structured logging with trace context
    slo.py               SLO definitions and burn-rate calculator
tests/
  test_integration.py    87 guardrail/A-B/API/monitoring tests
  test_rag_ingest.py     27 chunker + ingestion tests
  test_vectorstore.py    19 ChromaDB search tests
  fixtures/              Sample TXT/MD/PDF docs
evaluation/              backtest, dataset generator, locust, HTML report
deploy/                  K8s manifests, Helm chart, Prometheus, Grafana
demo/                    Self-contained docker-compose demo stack
```

## Coding conventions (FOLLOW THESE EXACTLY)

1. **One logical change per commit.** Never mix two features in one commit.
2. **Every code change ships with tests.** No new module without a matching test.
3. **Config-driven, not hardcoded.** New tunables go in `config.yaml` with an env
   override `${VAR:-default}`, read through `src/config.py`. Never hardcode paths,
   thresholds, model names, or URLs.
4. **Commit messages: lowercase imperative subject + multi-line body explaining WHY.**
   Example:
   ```
   add hybrid search combining semantic and keyword retrieval

   pure vector search misses exact-match terms (product codes, error
   strings). BM25 catches those. reciprocal rank fusion merges the two
   rankings. configurable via rag.search_mode.
   ```
5. **Match the existing style.** Type hints everywhere, dataclasses for structured
   returns, async where the surrounding code is async, lazy imports for heavy deps
   (chromadb, sentence-transformers, detoxify) so startup stays fast.
6. **Never break existing tests.** Run the full suite before every commit. If a
   change legitimately changes behavior, update the test in the SAME commit.
7. **Keep secrets out of code.** No API keys, tokens, or passwords in any file.

## Hard rules (DO NOT VIOLATE)

- Do NOT push directly. Make commits locally; the human reviews and pushes.
- Do NOT edit files under `tests/fixtures/` except to add new fixtures.
- Do NOT change the public API contract of existing endpoints without updating
  every caller and test.
- Do NOT add a dependency without adding it to `pyproject.toml`.
- Do NOT delete or rewrite git history.
- If a task is ambiguous, make the smallest reasonable assumption, state it in the
  commit body, and proceed. Don't stall.

## Definition of done (per task)

A task is done when ALL of these are true:
1. Code implemented and matches conventions above
2. Tests written and the FULL suite passes (`pytest -v`)
3. Config entries added if any new tunables
4. README updated if the feature is user-facing
5. One focused commit with a descriptive message, staged and committed locally
6. A one-line summary printed of what changed and what the human should review

## Daily workflow

The file `DEV_PLAN_30_DAYS.md` has one task per day. Each session:
1. Read this file (CLAUDE.md) and `DEV_PLAN_30_DAYS.md`.
2. Find today's task (the first unchecked `[ ]` day).
3. Implement it following the per-day spec and the conventions above.
4. Run the full test suite. Fix anything that breaks.
5. Commit locally with the message from the day's spec.
6. Mark the day `[x]` in `DEV_PLAN_30_DAYS.md` and commit that too.
7. Print a summary: what changed, test count, what to review, and the exact
   `git push` command the human should run.
