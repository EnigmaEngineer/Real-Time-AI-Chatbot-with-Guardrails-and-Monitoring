# Design Decisions

This document captures key architectural tradeoffs and the reasoning behind them.

---

## 1. Rule-Based vs ML Guardrails

**Decision:** Hybrid — rule-based PII/injection detection with optional ML toxicity (Detoxify).

**Tradeoffs considered:**

- Pure ML (e.g., fine-tuned classifier for every guardrail): Higher accuracy on nuanced cases, but adds 50–200ms per request and requires GPU or a separate inference service. For a small team, the operational burden of maintaining a second model pipeline isn't justified at launch.
- Pure rules (regex, keyword lists): Fast and deterministic but brittle — misses creative rephrasing of toxic content and produces false positives on benign strings that happen to match patterns.
- Hybrid (chosen): Regex for PII (where patterns are well-defined), keyword matching for prompt injection (where the attack surface is enumerable), and ML for toxicity (where semantic understanding matters). The ML path has a fallback to keywords if the model isn't installed, keeping the system functional in constrained environments.

**Revisit trigger:** When guardrail bypass rate exceeds 2% in production logs, invest in a fine-tuned classifier.

---

## 2. A/B Testing Framework: In-Process vs External

**Decision:** In-process deterministic hashing with SQLite recording.

**Alternatives rejected:**

- External services (LaunchDarkly, Statsig): Excellent but add a network dependency on the critical path. A feature flag service going down shouldn't block chat responses. Cost also scales with traffic.
- Database-backed assignment tables: Durable but adds latency per request for a lookup that can be computed deterministically.

**Why this works for a small team:** SHA-256 hashing of `user_id:experiment` gives consistent assignment without any external state. The experiment configuration lives in `config.yaml`, which is version-controlled and deployed atomically. Records are written asynchronously to SQLite (swappable to Postgres via the store interface).

**Known limitation:** No real-time experiment dashboard — you run `make report` or hit `/ab/results/{experiment}`. Acceptable at our scale; at >100K daily users, migrate to a streaming analytics pipeline.

---

## 3. Drift Detection: KS Test vs Population Stability Index

**Decision:** Kolmogorov-Smirnov test with standard deviation alerting.

**Why KS over PSI:**
- KS is distribution-free — no binning decisions required, which removes a hyperparameter that's easy to get wrong.
- PSI requires choosing bins and is sensitive to bin boundaries, especially for long-tailed distributions like query length.
- KS gives a p-value directly, making the alert threshold interpretable.

**Why not embedding-based drift:** Embedding drift (comparing cosine similarity distributions of query embeddings) would catch semantic shift, but requires running an embedding model on every request. Deferred until we have a dedicated embedding service.

**Alert threshold:** 2 standard deviations from the reference mean AND KS p-value < 0.05. Both conditions must hold to avoid noisy alerts from natural variance.

---

## 4. Circuit Breaker Pattern for LLM Calls

**Decision:** Per-model circuit breaker with automatic fallback.

The LLM provider is the single biggest external dependency. We've seen:
- Provider rate-limiting during traffic spikes (503s for 2+ minutes)
- Cold-start latency on serverless inference endpoints (first request takes 30s)

The circuit breaker opens after 5 consecutive failures, waits 60 seconds, then allows a single probe request. If the probe succeeds, the circuit closes. If not, it stays open for another cycle.

**Fallback chain:** Primary model → fallback model → static safe message. This means the chatbot always responds, even if degraded. Users see a generic fallback message rather than an error page.

---

## 5. Streaming Architecture: WebSocket vs SSE

**Decision:** WebSocket for streaming, with REST endpoints as fallback.

**Why WebSocket over SSE:**
- Bidirectional: the client can send feedback, typing indicators, or cancel signals on the same connection.
- SSE is simpler but uni-directional — feedback requires a separate HTTP call, which complicates client logic.
- WebSocket connections are well-supported by our Nginx ingress with the proxy-read-timeout annotation.

**Why keep REST endpoints:** Not every consumer needs streaming. Batch evaluation, internal tools, and mobile clients with poor WebSocket support use `/chat/strict` and `/chat/creative` as synchronous endpoints.

---

## 6. Feedback Storage: SQLite vs Postgres

**Decision:** SQLite for v1, with a clean interface for swapping.

SQLite handles our current scale (< 10K feedback events/day) without any infrastructure. The `FeedbackStore` class abstracts the storage — replacing `sqlite3` with `asyncpg` requires changing one file and zero API changes.

**Migration trigger:** When write contention causes > 50ms p99 on feedback submission, or when we need multi-pod writes (SQLite doesn't support concurrent writers across processes).

---

## 7. Hallucination Detection: Heuristic vs NLI Model

**Decision:** Heuristic scoring for v1.

The current approach counts certainty markers ("definitely," "guaranteed") relative to sentence count. This catches the most common failure mode: the model stating fabricated facts with high confidence.

**Why not NLI:** A proper Natural Language Inference model (e.g., deberta-v3-base-mnli) would compare claims against retrieved documents. This is the right long-term solution but requires:
1. A retrieval step (RAG) to have source documents to compare against
2. Running a second model per response (~100ms added latency)

**Planned upgrade path:** When RAG is integrated, add NLI-based hallucination detection as an output guardrail that compares each response sentence against retrieved passages.

---

## 8. Prometheus vs OpenTelemetry

**Decision:** Prometheus client library with Grafana dashboards.

**Why not OTel:** OpenTelemetry is the future standard, but for a small team deploying to a single Kubernetes cluster, Prometheus is simpler to operate. We don't need distributed tracing across microservices — it's a single service with an LLM backend.

**Structured logging with correlation IDs** gives us request-level tracing within the service. If we split into multiple services later, adding OTel instrumentation is a localized change in the metrics module.

---

## 9. Configuration: YAML + Environment Variable Overrides

**Decision:** YAML as the base configuration with `${VAR:-default}` interpolation.

**Why not just env vars:** Nested configuration (guardrail profiles, experiment variants) is painful to express as flat environment variables. YAML preserves structure while env vars handle secrets and per-environment overrides.

**Why not a config service:** Adds latency and a failure mode. Config changes go through the normal deployment pipeline (edit YAML → commit → deploy), which gives us an audit trail and rollback capability.
