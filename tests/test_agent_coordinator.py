"""Coordinator tests driven by a scripted fake LLM."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from src.agents.audit import AuditStore
from src.agents.coordinator import AgentCoordinator
from src.agents.policy import PolicyEngine
from src.agents.tools.base import Tool, ToolResult
from src.agents.tools.registry import ToolRegistry


class _ScriptedLLM:
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = 0

    async def generate(self, messages, model=None, **kwargs):
        self.calls += 1
        if not self._replies:
            return "FINAL: out of replies"
        return self._replies.pop(0)


class _EchoTool(Tool):
    name = "echo"
    description = "Echoes its input"
    input_schema = {"type": "object", "properties": {"text": {"type": "string"}}}
    policies = ()

    async def run(self, args):
        return ToolResult(ok=True, output=f"echo:{args.get('text', '')}")


class _BoomTool(Tool):
    name = "boom"
    description = "Always raises"
    input_schema = {"type": "object", "properties": {}}
    policies = ()

    async def run(self, args):
        raise RuntimeError("kaboom")


class _SlowTool(Tool):
    name = "slow"
    description = "Sleeps forever"
    input_schema = {"type": "object", "properties": {}}
    policies = ()

    async def run(self, args):
        await asyncio.sleep(10)
        return ToolResult(ok=True, output="slept")


def _registry_with(*tools, profile_tools=None):
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    reg.allow("default", profile_tools or [t.name for t in tools])
    return reg


def _audit_store():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return AuditStore(db_path=tmp.name), tmp.name


def test_happy_path_one_tool_then_final():
    llm = _ScriptedLLM([
        'ACTION: {"tool": "echo", "args": {"text": "hello"}}',
        "FINAL: I said hello",
    ])
    audit, _ = _audit_store()
    agent = AgentCoordinator(
        llm=llm,
        registry=_registry_with(_EchoTool()),
        policy=PolicyEngine(),
        audit=audit,
        max_iterations=3,
    )
    result = asyncio.run(agent.run(user_message="please echo hello"))
    assert result.terminated_reason == "final"
    assert result.iterations_used == 2
    assert "I said hello" in result.answer
    assert len(result.trace) == 1
    assert result.trace[0].tool == "echo"
    assert result.trace[0].status == "ok"


def test_unknown_tool_blocked_then_loop_continues():
    llm = _ScriptedLLM([
        'ACTION: {"tool": "nonexistent", "args": {}}',
        "FINAL: ok I'll stop trying",
    ])
    audit, _ = _audit_store()
    agent = AgentCoordinator(
        llm=llm,
        registry=_registry_with(_EchoTool()),
        policy=PolicyEngine(),
        audit=audit,
    )
    result = asyncio.run(agent.run(user_message="x"))
    assert result.terminated_reason == "final"
    assert len(result.trace) == 1
    assert result.trace[0].status == "blocked_pre"
    assert "unknown_or_disallowed_tool" in result.trace[0].violations


def test_iteration_cap_terminates_loop():
    # LLM keeps asking for tools, never returns FINAL.
    llm = _ScriptedLLM(['ACTION: {"tool": "echo", "args": {"text": "x"}}'] * 20)
    audit, _ = _audit_store()
    agent = AgentCoordinator(
        llm=llm,
        registry=_registry_with(_EchoTool()),
        policy=PolicyEngine(),
        audit=audit,
        max_iterations=3,
    )
    result = asyncio.run(agent.run(user_message="loop forever"))
    assert result.terminated_reason == "max_iterations"
    assert result.iterations_used == 3
    assert len(result.trace) == 3


def test_tool_exception_recorded_then_continues():
    llm = _ScriptedLLM([
        'ACTION: {"tool": "boom", "args": {}}',
        "FINAL: gave up on boom",
    ])
    audit, _ = _audit_store()
    agent = AgentCoordinator(
        llm=llm,
        registry=_registry_with(_BoomTool()),
        policy=PolicyEngine(),
        audit=audit,
    )
    result = asyncio.run(agent.run(user_message="break it"))
    assert result.terminated_reason == "final"
    assert result.trace[0].status == "error"
    assert "tool_exception" in result.trace[0].violations


def test_tool_timeout_recorded():
    llm = _ScriptedLLM([
        'ACTION: {"tool": "slow", "args": {}}',
        "FINAL: gave up",
    ])
    audit, _ = _audit_store()
    agent = AgentCoordinator(
        llm=llm,
        registry=_registry_with(_SlowTool()),
        policy=PolicyEngine(),
        audit=audit,
        tool_timeout_s=0.1,
        max_wall_seconds=5.0,
    )
    result = asyncio.run(agent.run(user_message="run slow"))
    assert result.trace[0].status == "error"
    assert "tool_timeout" in result.trace[0].violations


def test_parse_error_treated_as_final():
    # If the LLM drifts off-format, return what it said and mark parse_error.
    llm = _ScriptedLLM(["I'm just going to refuse to follow your format."])
    audit, _ = _audit_store()
    agent = AgentCoordinator(
        llm=llm,
        registry=_registry_with(_EchoTool()),
        policy=PolicyEngine(),
        audit=audit,
    )
    result = asyncio.run(agent.run(user_message="x"))
    assert result.terminated_reason == "parse_error"
    assert "refuse to follow" in result.answer


def test_audit_log_records_each_call():
    llm = _ScriptedLLM([
        'ACTION: {"tool": "echo", "args": {"text": "a"}}',
        'ACTION: {"tool": "echo", "args": {"text": "b"}}',
        "FINAL: done",
    ])
    audit, _ = _audit_store()
    agent = AgentCoordinator(
        llm=llm,
        registry=_registry_with(_EchoTool()),
        policy=PolicyEngine(),
        audit=audit,
        max_iterations=5,
    )
    asyncio.run(agent.run(user_message="hi", trace_id="t1", user_id="u1"))
    rows = audit.recent(limit=10)
    assert len(rows) == 2
    assert all(r["tool"] == "echo" for r in rows)
    assert all(r["trace_id"] == "t1" for r in rows)
