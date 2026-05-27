"""FastAPI application with WebSocket + SSE streaming, REST endpoints, and Prometheus metrics."""

import asyncio
import os
import time
import json
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse
from pydantic import BaseModel, Field
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, start_http_server

from src.config import load_config, get_guardrail_profile
from src.llm.client import LLMClient
from src.guardrails.input_guard import InputGuard
from src.guardrails.output_guard import OutputGuard
from src.guardrails.rate_limiter import ViolationRateLimiter
from src.guardrails.rpm_limiter import RPMRateLimiter
from src.abtesting.router import ABRouter, ExperimentRecord
from src.drift.detector import DriftDetector
from src.feedback.store import FeedbackStore, FeedbackEntry
from src.agents.coordinator import AgentCoordinator
from src.agents.policy import PolicyEngine
from src.agents.tools.registry import get_default_registry
from src.agents.audit import AuditStore
from src.monitoring.metrics import (
    REQUEST_LATENCY,
    REQUEST_TOTAL,
    SLO_LATENCY_VIOLATIONS,
    SLO_ERROR_TOTAL,
    ACTIVE_CONNECTIONS,
    ERROR_COUNTER,
    BUILD_INFO,
    AB_VARIANT_LATENCY,
    AB_VARIANT_FEEDBACK,
    DRIFT_P_VALUE,
    record_cost,
)
from src.monitoring.logging import (
    logger,
    set_trace_id,
    get_trace_id,
    set_user_context,
    set_variant,
    set_guardrail_action,
    reset_request_context,
)
from src.monitoring.slo import LATENCY_THRESHOLD_SECONDS


class ChatRequest(BaseModel):
    message: str
    user_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    conversation_id: str = Field(default_factory=lambda: uuid4().hex)
    experiment: str = ""
    system_prompt: str = "You are a helpful assistant."


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    message_id: str
    model: str
    experiment: str = ""
    variant: str = ""
    guardrail_violations: list[str] = []
    latency_ms: float = 0.0
    output_confidence: float = 1.0


class FeedbackRequest(BaseModel):
    conversation_id: str
    message_id: str
    user_id: str
    rating: int = Field(ge=-1, le=1)
    experiment: str = ""
    variant: str = ""
    comment: str = ""


# Global state
_llm: LLMClient | None = None
_ab_router: ABRouter | None = None
_drift: DriftDetector | None = None
_feedback: FeedbackStore | None = None
_rate_limiter: ViolationRateLimiter | None = None
_rpm_limiter: RPMRateLimiter | None = None
_vectorstore = None  # VectorStore — lazy import to avoid heavy deps at module level
_agent: AgentCoordinator | None = None
_agent_registry = None
_agent_audit: AuditStore | None = None
_config: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _llm, _ab_router, _drift, _feedback, _rate_limiter, _rpm_limiter, _vectorstore
    global _agent, _agent_registry, _agent_audit, _config
    _config = load_config()

    BUILD_INFO.info({"version": "1.0.0", "mock_mode": str(_config["llm"].get("mock_mode", False))})
    logger.info("Starting chatbot API")

    _llm = LLMClient(_config)
    _ab_router = ABRouter(_config)
    _drift = DriftDetector(_config)
    _feedback = FeedbackStore(_config)
    _rate_limiter = ViolationRateLimiter(_config)
    _rpm_limiter = RPMRateLimiter(_config)

    # Vector store — only init if RAG is enabled (avoids loading embedding model when unused)
    if _config.get("rag", {}).get("enabled", False):
        from src.rag.vectorstore import VectorStore
        _vectorstore = VectorStore(_config)
        logger.info("VectorStore enabled")

    # Multi-agent subsystem (Week 3)
    agents_cfg = _config.get("agents", {})
    if agents_cfg.get("enabled", True):
        injection_patterns = _config.get("guardrails", {}).get("injection_patterns", [])
        _agent_registry = get_default_registry(vectorstore=_vectorstore)
        _agent_audit = AuditStore(db_path=agents_cfg.get("audit_db_path", "data/agent_audit.db"))
        _agent = AgentCoordinator(
            llm=_llm,
            registry=_agent_registry,
            policy=PolicyEngine(injection_patterns=injection_patterns or None),
            audit=_agent_audit,
            max_iterations=int(agents_cfg.get("max_iterations", 5)),
            max_wall_seconds=float(agents_cfg.get("max_wall_seconds", 15.0)),
            max_obs_chars=int(agents_cfg.get("max_obs_chars", 8000)),
            tool_timeout_s=float(agents_cfg.get("tool_timeout_seconds", 10.0)),
        )
        logger.info("Agent coordinator enabled",
                    extra={"tools": [t.name for t in _agent_registry.all_tools()]})

    prom_port = _config.get("monitoring", {}).get("prometheus_port", 9090)
    try:
        start_http_server(prom_port)
        logger.info(f"Prometheus metrics on :{prom_port}")
    except OSError:
        logger.warning(f"Prometheus port {prom_port} already bound")

    yield

    if _llm:
        await _llm.close()
    if _feedback:
        _feedback.close()
    if _drift:
        _drift.close()
    if _agent_audit:
        _agent_audit.close()
    logger.info("Chatbot API shutdown complete")


app = FastAPI(title="AI Chatbot Platform", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── API key authentication middleware ──────────────────────────────────────

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


_PUBLIC_PATHS = {"/health", "/metrics", "/docs", "/openapi.json", "/redoc"}


class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        api_key = _config.get("server", {}).get("api_key", "")
        if not api_key:
            return await call_next(request)
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)
        if request.scope.get("type") == "websocket":
            return await call_next(request)
        provided = request.headers.get("X-API-Key", "")
        if provided != api_key:
            return JSONResponse(status_code=401, content={"error": "Invalid or missing API key"})

        # Per-API-key RPM rate limiting
        if _rpm_limiter and _rpm_limiter.enabled and not _rpm_limiter.allow(provided):
            remaining = _rpm_limiter.get_remaining(provided)
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded", "retry_after_seconds": 60},
                headers={"Retry-After": "60", "X-RateLimit-Remaining": str(remaining)},
            )

        return await call_next(request)


app.add_middleware(APIKeyMiddleware)


async def _process_chat(
    message: str,
    user_id: str,
    conversation_id: str,
    profile_name: str,
    experiment: str = "",
    system_prompt: str = "You are a helpful assistant.",
) -> ChatResponse:
    message_id = uuid4().hex[:16]
    start = time.monotonic()

    # ── Set structured log context for this request ────────────────────
    set_user_context(user_id)

    # ── Rate limit check — before any compute ─────────────────────────
    if _rate_limiter and _rate_limiter.is_banned(user_id):
        latency_s = time.monotonic() - start
        set_guardrail_action("rate_limit")
        REQUEST_TOTAL.labels(endpoint=f"/chat/{profile_name}", status="rate_limited").inc()
        REQUEST_LATENCY.labels(endpoint=f"/chat/{profile_name}", model="none", status="rate_limited").observe(latency_s)
        ban_msg = _config["guardrails"].get("ban_message", "You have been temporarily restricted.")
        logger.info("Request rate-limited", extra={"status": "rate_limited", "latency_ms": round(latency_s * 1000, 2)})
        return ChatResponse(
            response=ban_msg, conversation_id=conversation_id, message_id=message_id,
            model="none", experiment=experiment, guardrail_violations=["rate_limited"],
            latency_ms=round(latency_s * 1000, 2), output_confidence=0.0,
        )

    profile = get_guardrail_profile(profile_name)
    input_guard = InputGuard(profile, _config)
    output_guard = OutputGuard(profile)

    # ── A/B routing ────────────────────────────────────────────────────
    variant_name = ""
    model = _config["llm"]["default_model"]
    if _ab_router and experiment:
        assignment = _ab_router.assign(user_id, experiment)
        if assignment:
            model = assignment.variant.model
            system_prompt = assignment.variant.system_prompt
            profile_name = assignment.variant.guardrail_profile
            variant_name = assignment.variant_name
            profile = get_guardrail_profile(profile_name)
            input_guard = InputGuard(profile, _config)
            output_guard = OutputGuard(profile)
    set_variant(variant_name)

    # ── Input guardrails ───────────────────────────────────────────────
    input_result = input_guard.check(message, profile_name)
    if not input_result.passed:
        if _rate_limiter:
            _rate_limiter.record_violation(user_id)
        latency_s = time.monotonic() - start
        set_guardrail_action("block")
        REQUEST_TOTAL.labels(endpoint=f"/chat/{profile_name}", status="blocked").inc()
        REQUEST_LATENCY.labels(endpoint=f"/chat/{profile_name}", model=model, status="blocked").observe(latency_s)
        logger.info(
            "Input guardrail blocked request",
            extra={"status": "blocked", "latency_ms": round(latency_s * 1000, 2), "model": model},
        )
        return ChatResponse(
            response=_config["guardrails"]["fallback_message"],
            conversation_id=conversation_id, message_id=message_id, model=model,
            experiment=experiment, variant=variant_name,
            guardrail_violations=input_result.violations,
            latency_ms=round(latency_s * 1000, 2), output_confidence=0.0,
        )

    set_guardrail_action("pass")

    # ── Drift tracking (enriched: length + topic + sentiment) ──────────
    if _drift:
        _drift.record_input(input_result.sanitized_input)

    # ── LLM call ───────────────────────────────────────────────────────
    llm_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_result.sanitized_input},
    ]
    llm_failed = False
    try:
        response_text = await _llm.generate(llm_messages, model=model)
    except RuntimeError as exc:
        ERROR_COUNTER.labels(error_type="llm_failure").inc()
        SLO_ERROR_TOTAL.inc()
        logger.error(f"LLM call failed: {exc}", extra={"model": model})
        response_text = _config["guardrails"]["fallback_message"]
        llm_failed = True

    # ── Output guardrails ──────────────────────────────────────────────
    output_result = output_guard.check(response_text, profile_name)
    violations = input_result.violations + output_result.violations
    was_refused = not output_result.passed
    if was_refused:
        set_guardrail_action("block")
        response_text = _config["guardrails"]["fallback_message"]
        if _rate_limiter:
            _rate_limiter.record_violation(user_id)

    # ── Drift tracking for output (length + sentiment + refusal) ───────
    if _drift:
        _drift.record_output(response_text, was_refused)

    # ── Finalize metrics ───────────────────────────────────────────────
    latency_s = time.monotonic() - start
    status = "blocked" if violations else ("error" if llm_failed else "ok")
    REQUEST_LATENCY.labels(endpoint=f"/chat/{profile_name}", model=model, status=status).observe(latency_s)
    REQUEST_TOTAL.labels(endpoint=f"/chat/{profile_name}", status=status).inc()

    # SLO counters
    if latency_s > LATENCY_THRESHOLD_SECONDS:
        SLO_LATENCY_VIOLATIONS.inc()

    # Cost tracking (rough token estimate: 1 token ≈ 4 chars)
    input_tokens = len(input_result.sanitized_input) // 4
    output_tokens = len(response_text) // 4
    cost = record_cost(model, input_tokens, output_tokens)

    # A/B variant-level metrics
    if variant_name:
        AB_VARIANT_LATENCY.labels(experiment=experiment, variant=variant_name).observe(latency_s)

    # A/B experiment record
    if _ab_router and experiment and variant_name:
        _ab_router.record(
            ExperimentRecord(
                experiment=experiment, variant=variant_name, user_id=user_id,
                latency_ms=latency_s * 1000, feedback=None,
                token_count=output_tokens, timestamp=time.time(),
            )
        )

    # ── Structured log for the completed request ───────────────────────
    logger.info(
        "Chat request completed",
        extra={
            "status": status,
            "model": model,
            "endpoint": f"/chat/{profile_name}",
            "latency_ms": round(latency_s * 1000, 2),
            "confidence": output_result.confidence,
            "tokens_in": input_tokens,
            "tokens_out": output_tokens,
            "cost_usd": round(cost, 6),
        },
    )

    return ChatResponse(
        response=response_text, conversation_id=conversation_id,
        message_id=message_id, model=model, experiment=experiment,
        variant=variant_name, guardrail_violations=violations,
        latency_ms=round(latency_s * 1000, 2), output_confidence=output_result.confidence,
    )


@app.post("/chat/strict", response_model=ChatResponse)
async def chat_strict(req: ChatRequest):
    reset_request_context()
    set_trace_id(uuid4().hex[:16])
    return await _process_chat(
        req.message, req.user_id, req.conversation_id, "strict", req.experiment, req.system_prompt
    )


@app.post("/chat/creative", response_model=ChatResponse)
async def chat_creative(req: ChatRequest):
    reset_request_context()
    set_trace_id(uuid4().hex[:16])
    return await _process_chat(
        req.message, req.user_id, req.conversation_id, "creative", req.experiment, req.system_prompt
    )


@app.post("/chat/stream/{profile}")
async def chat_stream_sse(profile: str, req: ChatRequest):
    """Stream tokens via Server-Sent Events.

    Each token arrives as: data: {"token": "word "}\n\n
    Final event:           data: {"done": true, "confidence": 0.95, ...}\n\n

    Client usage (JavaScript):
        const es = new EventSource('/chat/stream/strict', {method: 'POST', ...});
        // or with fetch:
        const resp = await fetch('/chat/stream/strict', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({message: 'Hello'})
        });
        const reader = resp.body.getReader();
    """
    reset_request_context()
    set_trace_id(uuid4().hex[:16])
    set_user_context(req.user_id)

    # Validate profile
    try:
        guardrail_profile = get_guardrail_profile(profile)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown profile: {profile}")

    input_guard = InputGuard(guardrail_profile, _config)
    input_result = input_guard.check(req.message, profile)

    # Input guardrails — reject immediately without streaming
    if not input_result.passed:
        if _rate_limiter:
            _rate_limiter.record_violation(req.user_id)
        set_guardrail_action("block")
        REQUEST_TOTAL.labels(endpoint=f"/chat/stream/{profile}", status="blocked").inc()

        async def blocked_stream():
            payload = json.dumps({
                "error": True,
                "message": _config["guardrails"]["fallback_message"],
                "violations": input_result.violations,
            })
            yield f"data: {payload}\n\n"

        return StreamingResponse(
            blocked_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # A/B routing
    model = _config["llm"]["default_model"]
    system_prompt = req.system_prompt
    variant_name = ""
    if _ab_router and req.experiment:
        assignment = _ab_router.assign(req.user_id, req.experiment)
        if assignment:
            model = assignment.variant.model
            system_prompt = assignment.variant.system_prompt
            variant_name = assignment.variant_name
    set_variant(variant_name)
    set_guardrail_action("pass")

    # Drift tracking
    if _drift:
        _drift.record_input(input_result.sanitized_input)

    async def token_stream():
        start = time.monotonic()
        llm_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_result.sanitized_input},
        ]
        full_response = ""
        llm_failed = False

        try:
            async for chunk in _llm.generate_stream(llm_messages, model=model):
                full_response += chunk
                yield f"data: {json.dumps({'token': chunk})}\n\n"
        except RuntimeError:
            full_response = _config["guardrails"]["fallback_message"]
            llm_failed = True
            SLO_ERROR_TOTAL.inc()
            yield f"data: {json.dumps({'token': full_response})}\n\n"

        # Output guardrails on the complete response
        output_guard = OutputGuard(guardrail_profile)
        output_result = output_guard.check(full_response, profile)
        was_refused = not output_result.passed

        if was_refused:
            set_guardrail_action("block")
            if _rate_limiter:
                _rate_limiter.record_violation(req.user_id)

        # Drift tracking for output
        if _drift:
            _drift.record_output(full_response, was_refused)

        # Metrics
        latency_s = time.monotonic() - start
        status = "blocked" if was_refused else ("error" if llm_failed else "ok")
        REQUEST_TOTAL.labels(endpoint=f"/chat/stream/{profile}", status=status).inc()
        REQUEST_LATENCY.labels(endpoint=f"/chat/stream/{profile}", model=model, status=status).observe(latency_s)
        if latency_s > LATENCY_THRESHOLD_SECONDS:
            SLO_LATENCY_VIOLATIONS.inc()

        input_tokens = len(input_result.sanitized_input) // 4
        output_tokens = len(full_response) // 4
        record_cost(model, input_tokens, output_tokens)

        if variant_name:
            AB_VARIANT_LATENCY.labels(experiment=req.experiment, variant=variant_name).observe(latency_s)

        # Final SSE event with metadata
        done_payload = {
            "done": True,
            "confidence": output_result.confidence,
            "model": model,
            "variant": variant_name,
            "latency_ms": round(latency_s * 1000, 2),
        }
        if was_refused:
            done_payload["blocked"] = True
            done_payload["message"] = _config["guardrails"]["fallback_message"]
            done_payload["violations"] = output_result.violations
        yield f"data: {json.dumps(done_payload)}\n\n"

        logger.info(
            "SSE stream completed",
            extra={"status": status, "model": model, "latency_ms": round(latency_s * 1000, 2)},
        )

    return StreamingResponse(
        token_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.websocket("/ws/chat/{profile}")
async def websocket_chat(websocket: WebSocket, profile: str = "strict"):
    await websocket.accept()
    ACTIVE_CONNECTIONS.inc()
    reset_request_context()
    set_trace_id(uuid4().hex[:16])

    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
                continue

            # New trace per message in the WS session
            reset_request_context()
            set_trace_id(uuid4().hex[:16])

            message = payload.get("message", "")
            user_id = payload.get("user_id", uuid4().hex[:12])
            conversation_id = payload.get("conversation_id", uuid4().hex)
            experiment = payload.get("experiment", "")
            set_user_context(user_id)

            # Rate limit check
            if _rate_limiter and _rate_limiter.is_banned(user_id):
                set_guardrail_action("rate_limit")
                REQUEST_TOTAL.labels(endpoint=f"/ws/{profile}", status="rate_limited").inc()
                ban_msg = _config["guardrails"].get("ban_message", "You have been temporarily restricted.")
                rl_status = _rate_limiter.get_status(user_id)
                await websocket.send_json({
                    "type": "error", "message": ban_msg,
                    "violations": ["rate_limited"],
                    "ban_remaining_seconds": rl_status["ban_remaining_seconds"],
                })
                continue

            start = time.monotonic()
            guardrail_profile = get_guardrail_profile(profile)
            input_guard = InputGuard(guardrail_profile, _config)

            # Input guardrails
            input_result = input_guard.check(message, profile)
            if not input_result.passed:
                set_guardrail_action("block")
                if _rate_limiter:
                    _rate_limiter.record_violation(user_id)
                REQUEST_TOTAL.labels(endpoint=f"/ws/{profile}", status="blocked").inc()
                logger.info("WS input blocked", extra={"status": "blocked"})
                await websocket.send_json({
                    "type": "error",
                    "message": _config["guardrails"]["fallback_message"],
                    "violations": input_result.violations,
                })
                continue

            set_guardrail_action("pass")

            # A/B routing
            model = _config["llm"]["default_model"]
            system_prompt = "You are a helpful assistant."
            variant_name = ""
            if _ab_router and experiment:
                assignment = _ab_router.assign(user_id, experiment)
                if assignment:
                    model = assignment.variant.model
                    system_prompt = assignment.variant.system_prompt
                    variant_name = assignment.variant_name
            set_variant(variant_name)

            llm_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_result.sanitized_input},
            ]

            # Stream response
            await websocket.send_json({"type": "stream_start", "conversation_id": conversation_id})
            full_response = ""
            try:
                async for chunk in _llm.generate_stream(llm_messages, model=model):
                    full_response += chunk
                    await websocket.send_json({"type": "stream_chunk", "content": chunk})
            except RuntimeError:
                full_response = _config["guardrails"]["fallback_message"]
                SLO_ERROR_TOTAL.inc()
                await websocket.send_json({"type": "stream_chunk", "content": full_response})

            # Output guardrails
            output_guard = OutputGuard(guardrail_profile)
            output_result = output_guard.check(full_response, profile)

            latency_s = time.monotonic() - start
            if not output_result.passed:
                set_guardrail_action("block")
                if _rate_limiter:
                    _rate_limiter.record_violation(user_id)

            status = "blocked" if not output_result.passed else "ok"
            REQUEST_TOTAL.labels(endpoint=f"/ws/{profile}", status=status).inc()
            REQUEST_LATENCY.labels(endpoint=f"/ws/{profile}", model=model, status=status).observe(latency_s)
            if latency_s > LATENCY_THRESHOLD_SECONDS:
                SLO_LATENCY_VIOLATIONS.inc()

            input_tokens = len(input_result.sanitized_input) // 4
            output_tokens = len(full_response) // 4
            record_cost(model, input_tokens, output_tokens)

            if variant_name:
                AB_VARIANT_LATENCY.labels(experiment=experiment, variant=variant_name).observe(latency_s)

            if not output_result.passed:
                await websocket.send_json({
                    "type": "stream_end", "blocked": True,
                    "message": _config["guardrails"]["fallback_message"],
                    "violations": output_result.violations,
                    "confidence": output_result.confidence,
                    "latency_ms": round(latency_s * 1000, 2),
                })
            else:
                await websocket.send_json({
                    "type": "stream_end", "blocked": False,
                    "latency_ms": round(latency_s * 1000, 2),
                    "confidence": output_result.confidence,
                    "model": model, "variant": variant_name,
                })

            logger.info(
                "WS request completed",
                extra={"status": status, "model": model, "latency_ms": round(latency_s * 1000, 2)},
            )

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    finally:
        ACTIVE_CONNECTIONS.dec()


@app.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    entry = FeedbackEntry(
        conversation_id=req.conversation_id,
        message_id=req.message_id,
        user_id=req.user_id,
        rating=req.rating,
        experiment=req.experiment,
        variant=req.variant,
        comment=req.comment,
    )
    row_id = _feedback.submit(entry)
    if req.experiment and req.variant:
        rating_label = "up" if req.rating > 0 else "down"
        AB_VARIANT_FEEDBACK.labels(experiment=req.experiment, variant=req.variant, rating=rating_label).inc()
    return {"status": "ok", "id": row_id}


@app.get("/feedback/summary")
async def feedback_summary(experiment: str = Query(default="")):
    return _feedback.get_summary(experiment)


@app.get("/drift/status")
async def drift_status():
    if not _drift:
        return {"alerts": [], "recent_events": []}
    alerts = _drift.check_all(ab_router=_ab_router)
    recent = _drift.get_recent_events(hours=24.0)
    return {
        "alerts": [
            {
                "metric": a.metric,
                "direction": a.direction,
                "ks_statistic": round(a.ks_statistic, 4),
                "p_value": round(a.p_value, 6),
                "reference_mean": round(a.reference_mean, 2),
                "current_mean": round(a.current_mean, 2),
                "action_taken": a.action_taken,
            }
            for a in alerts
        ],
        "recent_events": [
            {
                "metric": e.metric,
                "direction": e.direction,
                "ks_statistic": round(e.ks_statistic, 4),
                "p_value": round(e.p_value, 6),
                "action_taken": e.action_taken,
                "variant_affected": e.variant_affected,
                "timestamp": e.timestamp,
            }
            for e in recent
        ],
    }


@app.get("/ab/experiments")
async def list_experiments():
    if not _ab_router:
        return {"experiments": {}}
    return {"experiments": _ab_router.experiments}


@app.get("/ab/results/{experiment}")
async def experiment_results(experiment: str):
    from src.abtesting.stats import analyze_experiment

    records = _ab_router.get_records(experiment)
    if not records:
        raise HTTPException(status_code=404, detail="No records for experiment")
    results = analyze_experiment(records, metric="latency_ms")
    return {
        "results": [
            {
                "variant_a": r.variant_a,
                "variant_b": r.variant_b,
                "metric": r.metric,
                "mean_a": round(r.mean_a, 2),
                "mean_b": round(r.mean_b, 2),
                "n_a": r.n_a,
                "n_b": r.n_b,
                "t_statistic": round(r.t_statistic, 4),
                "p_value": round(r.p_value, 6),
                "significant": r.significant,
                "recommended_winner": r.recommended_winner,
            }
            for r in results
        ]
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "mock_mode": _config["llm"].get("mock_mode", False)}


# ── Document ingestion endpoints ──────────────────────────────────────────

from fastapi import UploadFile, File
import tempfile
import shutil


@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Upload and ingest a document. Chunks are stored in ChromaDB if RAG is enabled."""
    from src.rag.ingest import DocumentIngestor, SUPPORTED_EXTENSIONS

    ext = Path(file.filename).suffix.lower() if file.filename else ""
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {ext}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        ingestor = DocumentIngestor(_config)
        result = ingestor.ingest(tmp_path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        os.unlink(tmp_path)

    # Persist chunks to vector store if available
    indexed = 0
    document_id = f"doc_{file.filename}_{int(time.time())}"
    if _vectorstore and result.chunks:
        indexed = _vectorstore.add_chunks(result.chunks, document_id=document_id)

    return {
        "filename": file.filename,
        "document_id": document_id,
        "format": result.format,
        "char_count": result.char_count,
        "chunk_count": result.chunk_count,
        "indexed": indexed,
        "duration_ms": result.duration_ms,
        "errors": result.errors,
        "chunks": [
            {
                "index": c.index,
                "token_count": c.token_count,
                "text_preview": c.text[:200],
            }
            for c in result.chunks
        ],
    }


@app.get("/rag/search")
async def rag_search(q: str = Query(..., min_length=1), top_k: int = Query(default=5, ge=1, le=20)):
    """Semantic search over ingested documents."""
    if not _vectorstore:
        raise HTTPException(status_code=503, detail="RAG not enabled. Set rag.enabled: true in config.")
    results = _vectorstore.search(q, top_k=top_k)
    return {
        "query": q,
        "results": [
            {
                "text": r.text[:500],
                "score": r.score,
                "document_id": r.document_id,
                "chunk_index": r.chunk_index,
            }
            for r in results
        ],
    }


@app.get("/rag/documents")
async def rag_list_documents():
    """List all ingested documents with chunk counts."""
    if not _vectorstore:
        raise HTTPException(status_code=503, detail="RAG not enabled.")
    return {"documents": _vectorstore.list_documents(), "stats": _vectorstore.get_stats()}


@app.delete("/rag/documents/{document_id}")
async def rag_delete_document(document_id: str):
    """Remove a document and all its chunks from the vector store."""
    if not _vectorstore:
        raise HTTPException(status_code=503, detail="RAG not enabled.")
    deleted = _vectorstore.delete_document(document_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"No chunks found for document '{document_id}'")
    return {"deleted_chunks": deleted, "document_id": document_id}


@app.get("/ratelimit/{user_id}")
async def ratelimit_status(user_id: str):
    if not _rate_limiter:
        return {"banned": False, "violations_in_window": 0, "ban_remaining_seconds": 0}
    return _rate_limiter.get_status(user_id)


@app.get("/metrics")
async def metrics():
    return JSONResponse(content=generate_latest().decode(), media_type=CONTENT_TYPE_LATEST)


# ── Agent endpoints (Week 3: multi-agent tool use with policy boundaries) ──


class AgentRequest(BaseModel):
    message: str
    user_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    conversation_id: str = Field(default_factory=lambda: uuid4().hex)
    profile: str = "default"  # default | readonly | math_only


class AgentToolCallView(BaseModel):
    iteration: int
    tool: str
    args: dict
    status: str
    violations: list[str]
    latency_ms: float
    output_preview: str
    error: str = ""


class AgentResponse(BaseModel):
    answer: str
    trace: list[AgentToolCallView]
    iterations_used: int
    terminated_reason: str
    latency_ms: float
    guardrail_violations: list[str] = []


@app.post("/chat/agent", response_model=AgentResponse)
async def chat_agent(req: AgentRequest):
    """ReAct-style agent with tool use. Every tool call is policy-checked
    and audit-logged. Try to make it call a tool it shouldn't —
    every blocked attempt shows up at GET /agents/audit."""
    if not _agent or not _agent_registry:
        raise HTTPException(status_code=503, detail="Agent subsystem not enabled")

    reset_request_context()
    trace_id = uuid4().hex[:16]
    set_trace_id(trace_id)
    set_user_context(req.user_id)

    # Rate-limit check — agents are expensive, treat them like chat requests.
    if _rate_limiter and _rate_limiter.is_banned(req.user_id):
        ban_msg = _config["guardrails"].get("ban_message", "You have been temporarily restricted.")
        REQUEST_TOTAL.labels(endpoint="/chat/agent", status="rate_limited").inc()
        return AgentResponse(
            answer=ban_msg, trace=[], iterations_used=0,
            terminated_reason="rate_limited", latency_ms=0.0,
            guardrail_violations=["rate_limited"],
        )

    # Input guard on the original user message — agents see the same input
    # filtering as /chat/strict so jailbreak attempts never reach the loop.
    profile = get_guardrail_profile("strict")
    input_guard = InputGuard(profile, _config)
    input_result = input_guard.check(req.message, "strict")
    if not input_result.passed:
        if _rate_limiter:
            _rate_limiter.record_violation(req.user_id)
        set_guardrail_action("block")
        REQUEST_TOTAL.labels(endpoint="/chat/agent", status="blocked").inc()
        logger.info("Agent request blocked at input guard",
                    extra={"violations": input_result.violations})
        return AgentResponse(
            answer=_config["guardrails"]["fallback_message"],
            trace=[], iterations_used=0,
            terminated_reason="input_guard",
            latency_ms=0.0,
            guardrail_violations=input_result.violations,
        )

    # Drift tracking sees the input like any other endpoint.
    if _drift:
        _drift.record_input(input_result.sanitized_input)

    start = time.monotonic()
    result = await _agent.run(
        user_message=input_result.sanitized_input,
        profile=req.profile,
        trace_id=trace_id,
        user_id=req.user_id,
    )

    # Output guard on the final answer — protects against the agent emitting
    # banned content even when no individual tool output was flagged.
    output_guard = OutputGuard(profile)
    output_result = output_guard.check(result.answer, "strict")
    output_violations: list[str] = []
    final_answer = result.answer
    if not output_result.passed:
        set_guardrail_action("block")
        final_answer = _config["guardrails"]["fallback_message"]
        output_violations = output_result.violations
        if _rate_limiter:
            _rate_limiter.record_violation(req.user_id)

    if _drift:
        _drift.record_output(final_answer, not output_result.passed)

    latency_s = time.monotonic() - start
    status = "blocked" if output_violations else "ok"
    REQUEST_TOTAL.labels(endpoint="/chat/agent", status=status).inc()
    REQUEST_LATENCY.labels(endpoint="/chat/agent", model="agent", status=status).observe(latency_s)
    if latency_s > LATENCY_THRESHOLD_SECONDS:
        SLO_LATENCY_VIOLATIONS.inc()

    logger.info(
        "Agent request completed",
        extra={
            "status": status, "endpoint": "/chat/agent",
            "iterations": result.iterations_used,
            "terminated_reason": result.terminated_reason,
            "latency_ms": round(latency_s * 1000, 2),
        },
    )

    return AgentResponse(
        answer=final_answer,
        trace=[
            AgentToolCallView(
                iteration=t.iteration, tool=t.tool, args=t.args, status=t.status,
                violations=t.violations, latency_ms=round(t.latency_ms, 2),
                output_preview=t.output_preview, error=t.error,
            )
            for t in result.trace
        ],
        iterations_used=result.iterations_used,
        terminated_reason=result.terminated_reason,
        latency_ms=round(latency_s * 1000, 2),
        guardrail_violations=output_violations,
    )


@app.get("/agents/tools")
async def list_agent_tools(profile: str = Query(default="default")):
    """List tools available to a given profile, with their declared policies."""
    if not _agent_registry:
        raise HTTPException(status_code=503, detail="Agent subsystem not enabled")
    try:
        tools = _agent_registry.allowed_for(profile)
    except Exception:
        tools = []
    return {
        "profile": profile,
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
                "policies": list(t.policies),
            }
            for t in tools
        ],
    }


@app.get("/agents/audit")
async def agent_audit(
    limit: int = Query(default=50, ge=1, le=500),
    status: str = Query(default=""),
):
    """Recent agent tool calls. Powers the public 'break it' transparency view:
    every blocked attempt is visible here, indexed by status."""
    if not _agent_audit:
        raise HTTPException(status_code=503, detail="Agent subsystem not enabled")
    rows = _agent_audit.recent(limit=limit, status=status or None)
    return {"rows": rows, "stats": _agent_audit.stats()}
