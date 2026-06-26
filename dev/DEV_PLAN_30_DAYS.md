# DEV_PLAN_30_DAYS.md — 30-Day Build Plan

This is a day-by-day plan for Claude Code to execute, one task per working day,
building toward a finished, publishable v1.0 of the chatbot platform. Each day is a
focused commit. Each week ends with a LinkedIn milestone for the human to post.

**How to use this file (Claude Code):** Read `CLAUDE.md` first. Then find the first
unchecked day below (`[ ]`), implement exactly that day's task, run the full test
suite, commit, mark the day `[x]`, and print a summary. Do ONE day per session unless
told otherwise.

**Starting point (already done — do NOT redo):**
- FastAPI + WebSocket + SSE, three guardrail layers, A/B testing, drift detection,
  Prometheus/Grafana/SLO, RAG ingestion (chunker + ingest + ChromaDB vectorstore).
- 133 tests passing. We are at Week 4 of the broader roadmap.

**Cadence:** ~5 working days per week, Monday–Friday. Weekends are buffer/LinkedIn.
Each day assumes the previous day is committed and the suite is green.

---

## DAY 0 — Reconcile starting state (run this once, first session)

Before starting Day 1, confirm what actually exists so you don't rebuild or assume
wrong. Run these checks and fix any mismatch:

```bash
# 1. Confirm the full suite is green and count the baseline
LLM_MOCK_MODE=true CHATBOT_API_KEY= python -m pytest tests/ -q

# 2. Confirm the RAG vector store layer is present (Days 3-4 of the prior plan)
ls src/rag/vectorstore.py && grep -n "rag/search\|rag/documents" src/api/main.py
```

- If `src/rag/vectorstore.py` and the `/rag/search` + `/rag/documents` endpoints
  exist → you're at the expected baseline. Proceed to Day 1.
- If they DON'T exist → the vector-store commit hasn't landed yet. Implement it first
  (ChromaDB wrapper with add/search/delete using sentence-transformers/all-MiniLM-L6-v2,
  plus `/rag/search`, `/rag/documents`, `DELETE /rag/documents/{id}`), commit it as
  `add vector store with ChromaDB backend`, THEN start Day 1.

Record the baseline test count in your summary so each later day can confirm the count
only goes up. Do not proceed until the suite is green.

---

## WEEK 1 (Days 1–5): Make RAG actually answer questions

Goal by Friday: the chatbot retrieves relevant document context and uses it in its
answers, with citations. This is the week that turns "has a vector store" into "is a
real RAG chatbot."

### [ ] Day 1 — Wire retrieval into the chat path
**Why:** The vector store exists but chat responses don't use it yet. This connects them.
**Files:** `src/api/main.py`, `src/rag/retriever.py` (new), `config.yaml`, `tests/test_rag_chat.py` (new)
**Implement:**
- Create `src/rag/retriever.py` with a `RAGRetriever` class that wraps the vector store
  and returns formatted context: top-k chunks joined with source labels.
- In `_process_chat` (main.py), when `rag.enabled` is true, retrieve context for the
  user message and inject it into the system prompt as a "Use this context:" block
  before calling the LLM.
- Add config: `rag.inject_context: true`, `rag.max_context_tokens: 2000`.
**Tests:** ingest a fixture doc, send a chat whose answer is in the doc, assert the
retrieved context appears in the prompt sent to the (mock) LLM. Assert RAG-off path
is unchanged.
**Acceptance:** full suite green; with RAG on, the mock LLM receives augmented prompt.
**Commit:** `wire RAG retrieval into chat with context injection`

### [ ] Day 2 — Add citations to responses
**Why:** Grounded answers need to show their sources. This is the trust feature.
**Files:** `src/rag/retriever.py`, `src/api/main.py`, `tests/test_rag_chat.py`
**Implement:**
- Each retrieved chunk carries `source` and `chunk_index`. Build a `citations` list
  (source, score, snippet) alongside the context.
- Add `citations` to `ChatResponse` (default empty list, so non-RAG responses are
  unaffected).
- Return citations in REST, SSE final event, and WebSocket end frame.
**Tests:** assert citations populated when RAG on, empty when off; assert each citation
has source + score.
**Acceptance:** suite green; citations appear in all three response paths.
**Commit:** `add source citations to RAG responses`

### [ ] Day 3 — Retrieval quality eval
**Why:** You can't improve retrieval you can't measure. This gives you numbers.
**Files:** `evaluation/rag_eval.py` (new), `evaluation/rag_eval_set.jsonl` (new), `tests/test_rag_eval.py` (new)
**Implement:**
- Build a 20-item eval set: question + the fixture doc + the expected source chunk.
- `rag_eval.py` computes precision@k, recall@k, and MRR (mean reciprocal rank).
- Print a table and write JSON results.
**Tests:** run eval on the fixture corpus, assert precision@5 > 0.5 (sanity threshold).
**Acceptance:** suite green; `python -m evaluation.rag_eval` prints metrics.
**Commit:** `add RAG retrieval evaluation (precision, recall, MRR)`

### [ ] Day 4 — Hybrid search (semantic + BM25)
**Why:** Vector search misses exact terms (codes, error strings). BM25 catches them.
**Files:** `src/rag/hybrid_search.py` (new), `src/rag/vectorstore.py`, `config.yaml`, `tests/test_vectorstore.py`
**Implement:**
- Add a BM25 keyword index (use `rank-bm25`, add to pyproject) over the same chunks.
- Reciprocal rank fusion to merge vector + BM25 rankings.
- Config: `rag.search_mode: hybrid|semantic|keyword` (default hybrid).
**Tests:** a query with an exact term that semantic search ranks low should rank high
under hybrid; compare recall on the eval set, assert hybrid >= semantic.
**Acceptance:** suite green; all three search modes work.
**Commit:** `add hybrid search combining semantic and keyword retrieval`

### [ ] Day 5 — Document management API + polish
**Why:** Users need to see and manage what's been ingested. Rounds out the RAG feature.
**Files:** `src/api/main.py`, `tests/test_rag_ingest.py`, `README.md`
**Implement:**
- Ensure `GET /rag/documents` returns per-doc chunk counts and ingest timestamps.
- Add `GET /rag/documents/{id}/chunks` to preview a document's chunks.
- Update README with a RAG quickstart: ingest → search → chat-with-citations.
**Tests:** full document CRUD lifecycle; chunk preview endpoint.
**Acceptance:** suite green; README RAG section complete.
**Commit:** `add document management endpoints and RAG quickstart docs`

> **LINKEDIN MILESTONE — WEEK 1 (Friday/Saturday):**
> Build-log post: "RAG that actually cites its sources." Show the citations feature.
> Angle: most RAG demos answer confidently with no provenance; yours shows exactly
> which chunk each claim came from, with retrieval-quality numbers (precision@5) to
> back it. Screenshot: a chat response with citations + the eval table.
> Hashtags: #LLMOps #RAG #AIInfrastructure

---

## WEEK 2 (Days 6–10): Real LLM + conversation memory

Goal by Friday: the platform talks to a real LLM provider (not just mock), remembers
multi-turn context, and counts tokens accurately.

### [ ] Day 6 — Provider abstraction (OpenAI / Ollama / vLLM)
**Why:** Mock mode proves the architecture; real providers prove it works.
**Files:** `src/llm/providers.py` (new), `src/llm/client.py`, `config.yaml`, `tests/test_llm_providers.py` (new)
**Implement:**
- A `Provider` protocol with `generate` and `generate_stream`.
- Implementations: OpenAI-compatible (covers OpenAI, vLLM, Together), Ollama (local),
  and the existing Mock. Select via `LLM_PROVIDER` env / `llm.provider` config.
- Keep the circuit breaker and retries in `client.py`; providers are pluggable behind it.
**Tests:** mock each provider's HTTP layer, assert correct request shape and streaming
parse. Assert mock provider still default in tests.
**Acceptance:** suite green; provider selectable by config.
**Commit:** `add pluggable LLM provider abstraction (openai, ollama, vllm)`

### [ ] Day 7 — Accurate token counting
**Why:** Cost and context-window math need real token counts, not char/4 estimates.
**Files:** `src/llm/tokens.py` (new), `src/api/main.py`, `src/monitoring/metrics.py`, `tests/test_tokens.py` (new)
**Implement:**
- `tokens.py` using `tiktoken` (add to pyproject) with a char/4 fallback when the
  model is unknown.
- Replace the `// 4` estimates in main.py with real counts for cost + metrics.
**Tests:** known string → known token count; fallback path for unknown model.
**Acceptance:** suite green; cost metric uses real tokens.
**Commit:** `add accurate token counting with tiktoken`

### [ ] Day 8 — Conversation memory (sliding window)
**Why:** A chatbot that forgets the previous turn isn't a chatbot.
**Files:** `src/memory/conversation.py` (new), `config.yaml`, `tests/test_memory.py` (new)
**Implement:**
- Per-session message history with a token-budget-aware sliding window (keep system
  prompt + most recent turns under `memory.max_tokens`).
- Config: `memory.enabled`, `memory.max_turns`, `memory.max_tokens`.
**Tests:** add turns past the budget, assert oldest user/assistant turns drop but the
system prompt stays; assert token budget respected.
**Acceptance:** suite green; window truncates correctly.
**Commit:** `add conversation memory with token-aware sliding window`

### [ ] Day 9 — Multi-turn chat in API + WebSocket + SSE
**Why:** Memory has to be reachable through every entry point.
**Files:** `src/api/main.py`, `tests/test_integration.py`
**Implement:**
- Accept optional `conversation_history` in `ChatRequest`.
- WebSocket maintains per-connection history; SSE accepts history in the body.
- Thread history through `_process_chat` so the LLM sees prior turns.
**Tests:** a two-turn conversation where turn 2 references turn 1; assert the LLM
prompt for turn 2 contains turn 1.
**Acceptance:** suite green; multi-turn works in all three paths.
**Commit:** `add multi-turn conversation support across REST, WS, and SSE`

### [ ] Day 10 — Conversation persistence (SQLite)
**Why:** History should survive a restart and be queryable.
**Files:** `src/memory/store.py` (new), `src/api/main.py`, `config.yaml`, `tests/test_memory.py`
**Implement:**
- SQLite store: conversations + messages tables. Save each turn.
- `GET /conversations/{id}` and `GET /conversations?user_id=` endpoints.
**Tests:** create, retrieve, list, and verify ordering of a conversation.
**Acceptance:** suite green; history persists across store re-open.
**Commit:** `add conversation persistence with sqlite store`

> **LINKEDIN MILESTONE — WEEK 2:**
> Build-log post: "Swappable LLM backends + real token accounting." Angle: the model
> is a plugin, not the point. Show the provider config switch (mock → ollama → openai)
> and the cost-per-conversation metric now driven by real tiktoken counts.
> Hashtags: #LLMOps #AIInfrastructure #MLOps

---

## WEEK 3 (Days 11–15): ML guardrails + hardening

Goal by Friday: guardrails upgraded from rules to ML where it matters, plus auth and
caching for production readiness.

### [ ] Day 11 — ML prompt-injection classifier
**Why:** Rule-based scoring is a good baseline; an ML model raises the ceiling.
**Files:** `src/guardrails/injection_classifier.py` (new), `src/guardrails/input_guard.py`, `config.yaml`, `tests/test_integration.py`
**Implement:**
- Load a small pre-trained text-classification model (e.g. a DistilBERT injection
  detector from HuggingFace) lazily, CPU-only. Hybrid score = max(rule, ml).
- Config: `guardrails.injection.method: rule|ml|hybrid` (default hybrid), with graceful
  fallback to rule-only if the model can't load (offline CI).
**Tests:** known injections still blocked; benign prompts still pass; ml-unavailable
falls back to rule cleanly.
**Acceptance:** suite green offline (model optional); hybrid path works when present.
**Commit:** `add ML-based prompt injection classifier with rule fallback`

### [ ] Day 12 — NLI hallucination check for RAG
**Why:** With RAG context available, you can check answers against their sources.
**Files:** `src/guardrails/hallucination_nli.py` (new), `src/guardrails/output_guard.py`, `config.yaml`, `tests/test_integration.py`
**Implement:**
- Compare each response sentence against retrieved context using an NLI model
  (entailment vs contradiction), lazy/optional. Flag low-entailment sentences.
- Config toggle `guardrails.hallucination.method: heuristic|nli`.
**Tests:** a response contradicting the context is flagged; a grounded one passes;
heuristic fallback when model absent.
**Acceptance:** suite green offline; NLI path works when model present.
**Commit:** `add NLI-based hallucination detection for grounded answers`

### [ ] Day 13 — JWT auth with roles
**Why:** Real deployments need authn/authz, not just an API key.
**Files:** `src/auth/jwt.py` (new), `src/api/main.py`, `config.yaml`, `tests/test_auth.py` (new)
**Implement:**
- JWT issue/validate. Roles: admin, user, readonly. `POST /auth/token` login.
- Admin-only: RAG document write, experiment create. Keep the existing X-API-Key path
  working in parallel (config switch `auth.mode: api_key|jwt|both`).
**Tests:** valid/expired/invalid tokens; role enforcement on a protected route.
**Acceptance:** suite green; both auth modes selectable.
**Commit:** `add JWT authentication with role-based access control`

### [ ] Day 14 — Redis caching (retrieval + semantic LLM cache)
**Why:** Repeated queries shouldn't pay full cost twice.
**Files:** `src/cache/redis_cache.py` (new), `src/api/main.py`, `config.yaml`, `tests/test_cache.py` (new)
**Implement:**
- Async Redis with TTL. Cache retrieval results by query hash; semantic LLM cache by
  hash of (system_prompt + message). In-memory fallback when Redis absent (for CI).
- Prometheus cache hit/miss counters.
**Tests:** second identical request hits cache; TTL expiry; in-memory fallback path.
**Acceptance:** suite green without a live Redis; cache counters increment.
**Commit:** `add redis caching for retrieval and llm responses`

### [ ] Day 15 — Graceful shutdown + connection draining
**Why:** Rolling deploys must not drop in-flight requests or WebSocket sessions.
**Files:** `src/api/main.py`, `deploy/k8s-manifests.yaml`, `tests/test_integration.py`
**Implement:**
- On SIGTERM: stop accepting new work, drain active WebSocket connections (timeout),
  finish in-flight requests, health endpoint returns `draining`.
- K8s: `terminationGracePeriodSeconds` + preStop hook.
**Tests:** simulate shutdown signal, assert health flips to draining and no new work
is accepted.
**Acceptance:** suite green; draining state observable.
**Commit:** `add graceful shutdown with connection draining`

> **LINKEDIN MILESTONE — WEEK 3:**
> Thesis-ish post: "Guardrails: when rules beat ML, and when they don't." Show the
> hybrid injection classifier and NLI hallucination check. Angle: you don't reach for
> a model first — you start with rules, measure, then add ML only where it moves the
> precision/recall numbers. Include a before/after on the eval set.
> Hashtags: #LLMOps #AISafety #MLOps

---

## WEEK 4 (Days 16–20): Observability depth + cost control

Goal by Friday: distributed tracing, anomaly detection, model routing by complexity,
and cost budgets — the SRE-credible layer.

### [ ] Day 16 — OpenTelemetry tracing
**Why:** One trace ID per request across guardrails → retrieval → LLM → output.
**Files:** `src/monitoring/tracing.py` (new), `src/api/main.py`, `docker-compose.yml`, `tests/test_integration.py`
**Implement:**
- OTel spans for each stage; export to a Jaeger service in compose. Reuse the existing
  trace_id as the OTel trace id.
**Tests:** a request creates the expected span tree (use an in-memory span exporter).
**Acceptance:** suite green; spans created with correct parent/child links.
**Commit:** `add opentelemetry distributed tracing`

### [ ] Day 17 — Token-level latency (TTFT + ITL)
**Why:** Streaming UX is judged by time-to-first-token, not total latency.
**Files:** `src/api/main.py`, `src/monitoring/metrics.py`, `deploy/grafana/dashboards/chatbot.json`, `tests/test_integration.py`
**Implement:**
- Measure time-to-first-token and inter-token latency in the streaming paths.
- New histograms; add Grafana panels.
**Tests:** TTFT recorded and < total latency; ITL > 0 for multi-token responses.
**Acceptance:** suite green; new metrics exposed.
**Commit:** `add token-level latency tracking (ttft, itl)`

### [ ] Day 18 — Model routing by query complexity
**Why:** Cheap model for simple queries, big model for hard ones = cost savings.
**Files:** `src/llm/model_router.py` (new), `src/api/main.py`, `config.yaml`, `tests/test_llm_providers.py`
**Implement:**
- Classify complexity (length, question type, retrieved-context size) → route to
  small vs large model. Fallback chain preserved.
- Track estimated savings vs always-large.
**Tests:** simple query → small model; complex query → large model.
**Acceptance:** suite green; routing decision logged.
**Commit:** `add complexity-based model routing for cost efficiency`

### [ ] Day 19 — Cost budgets + alerts
**Why:** Unbounded agent spend is a real production risk.
**Files:** `src/monitoring/cost_alert.py` (new), `src/api/main.py`, `config.yaml`, `tests/test_integration.py`
**Implement:**
- Daily/monthly cost tracking; alert at threshold; optional hard limit that rejects
  with 402/503 when the daily budget is exhausted (config `cost.hard_limit`).
- `GET /admin/cost?period=month` breakdown by model.
**Tests:** budget exceeded → requests rejected when hard limit on; soft mode only alerts.
**Acceptance:** suite green; cost endpoint returns breakdown.
**Commit:** `add cost budgets with alerting and optional hard limit`

### [ ] Day 20 — Anomaly detection on metrics
**Why:** Drift catches distribution shift; anomalies catch sudden point spikes.
**Files:** `src/monitoring/anomaly.py` (new), `deploy/prometheus_alerts.yml`, `tests/test_integration.py`
**Implement:**
- EWMA + z-score over latency / error-rate / cost. Emit an anomaly-score gauge and a
  Prometheus alert rule.
**Tests:** inject a synthetic spike, assert anomaly score crosses threshold.
**Acceptance:** suite green; anomaly gauge exposed.
**Commit:** `add ewma z-score anomaly detection on key metrics`

> **LINKEDIN MILESTONE — WEEK 4:**
> Build-log post: "You can't run AI in prod blind." Show the Jaeger trace of a single
> request flowing through every layer, plus the TTFT panel and cost breakdown. Angle:
> observability isn't a dashboard you add at the end — it's a trace ID threaded through
> every decision the system makes.
> Hashtags: #Observability #LLMOps #SRE

---

## WEEK 5+ (Days 21–30): Polish, tests, and the v1.0 launch

Goal: e2e tests, CI/CD, docs, demo polish, and a tagged v1.0.0 release worth a launch post.

### [ ] Day 21 — End-to-end tests (Playwright)
**Files:** `tests/e2e/test_chat_ui.py` (new), `.github/workflows/`
**Implement:** browser tests over the chat UI: send message → stream → citations;
injection → blocked message shown; reconnect on disconnect. Wire into CI.
**Commit:** `add end-to-end browser tests for chat ui`

### [ ] Day 22 — Chaos / resilience tests
**Files:** `tests/chaos/test_resilience.py` (new)
**Implement:** fault injection — LLM timeout, vector store down, Redis down. Assert
graceful degradation and correct fallback messages.
**Commit:** `add chaos tests for dependency failures`

### [ ] Day 23 — CI/CD pipeline
**Files:** `.github/workflows/ci.yml` (new), `.github/workflows/release.yml` (new)
**Implement:** lint + test + Docker build on every PR; tag → build → push image to
GHCR. Badges in README.
**Commit:** `add ci/cd pipeline with lint, test, and release workflows`

### [ ] Day 24 — API docs
**Files:** `docs/API.md` (new)
**Implement:** every endpoint with request/response examples, auth, error codes.
**Commit:** `add complete api documentation`

### [ ] Day 25 — Deployment + runbooks
**Files:** `docs/DEPLOYMENT.md` (new), `docs/RUNBOOKS.md` (new)
**Implement:** step-by-step deploy (EKS/GKE/bare metal); incident runbook per alert.
**Commit:** `add deployment guide and incident runbooks`

### [ ] Day 26 — Sample datasets + guided demo
**Files:** `demo/sample_data/` (new), `demo/scenarios/` (new), `Makefile`
**Implement:** 2–3 prebuilt RAG knowledge bases + scripted scenario walkthrough;
`make demo-guided`.
**Commit:** `add sample datasets and guided demo scenarios`

### [ ] Day 27 — Architecture diagrams + screenshots
**Files:** `docs/architecture.md`, `docs/screenshots/`
**Implement:** Mermaid architecture diagram; capture Grafana, chat UI, trace, eval.
**Commit:** `add architecture diagrams and screenshots`

### [ ] Day 28 — README rewrite + feature matrix
**Files:** `README.md`, `CHANGELOG.md` (new)
**Implement:** badges, feature matrix, comparison table, quickstart; semver changelog.
**Commit:** `rewrite readme with feature matrix and changelog`

### [ ] Day 29 — Performance pass + cleanup
**Files:** various
**Implement:** move torch to an optional `[gpu]` extra (slim default image), remove
dead code (vulture), tighten startup time, ensure `make demo` cold-starts fast.
**Commit:** `slim default image and clean up dead code`

### [ ] Day 30 — v1.0.0 release
**Files:** `CHANGELOG.md`, git tag
**Implement:** final full-suite run; tag `v1.0.0`; release notes summarizing every
subsystem. Print the GitHub Release steps for the human.
**Commit:** `release v1.0.0`

> **LINKEDIN MILESTONE — WEEK 5/6 (LAUNCH):**
> The big one. "I built a production AI platform in public over 6 weeks. Here's
> everything in it, and it runs in 5 minutes." Carousel: architecture, guardrails,
> RAG with citations, observability/tracing, cost control, the eval numbers, and the
> 5-minute demo. Link the v1.0.0 release. This is the post the whole series was
> building toward — reference the earlier build-logs so it reads as a journey with
> receipts. Hashtags: #LLMOps #AIInfrastructure #RAG #Observability

---

## Progress tracker

Week 1: [ ][ ][ ][ ][ ]   RAG answers with citations
Week 2: [ ][ ][ ][ ][ ]   Real LLM + memory
Week 3: [ ][ ][ ][ ][ ]   ML guardrails + auth + cache
Week 4: [ ][ ][ ][ ][ ]   Tracing + cost + anomalies
Week 5+:[ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]   Polish + v1.0.0

When all boxes are checked, the project is v1.0 and every week has shipped a LinkedIn post.
