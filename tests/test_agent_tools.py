"""Unit tests for the individual tools."""

import asyncio

import httpx
import pytest

from src.agents.tools.calculator import CalculatorTool
from src.agents.tools.rag_search import RagSearchTool
from src.agents.tools.web_fetch import WebFetchTool, _host_allowed


# ── calculator ──────────────────────────────────────────────────────────


def _run(coro):
    return asyncio.run(coro)


def test_calculator_basic_arithmetic():
    r = _run(CalculatorTool().run({"expression": "(2 + 3) * 4"}))
    assert r.ok
    assert r.output == "20"


def test_calculator_rejects_function_call():
    r = _run(CalculatorTool().run({"expression": "__import__('os').system('echo')"}))
    assert not r.ok
    assert "not allowed" in r.error.lower() or "syntax" in r.error.lower()


def test_calculator_rejects_name_reference():
    r = _run(CalculatorTool().run({"expression": "x + 1"}))
    assert not r.ok


def test_calculator_caps_exponent():
    r = _run(CalculatorTool().run({"expression": "2 ** 10000"}))
    assert not r.ok
    assert "exponent" in r.error.lower()


def test_calculator_divide_by_zero():
    r = _run(CalculatorTool().run({"expression": "1 / 0"}))
    assert not r.ok


def test_calculator_non_string_arg():
    r = _run(CalculatorTool().run({"expression": 12}))
    assert not r.ok


# ── web_fetch ───────────────────────────────────────────────────────────


def test_host_allowed_accepts_wikipedia():
    assert _host_allowed("https://en.wikipedia.org/wiki/Foo")


def test_host_allowed_rejects_unknown():
    assert not _host_allowed("https://evil.example.com/exfil")


def test_host_allowed_rejects_non_http_scheme():
    # file:// and ftp:// must never escape the allowlist
    assert not _host_allowed("file:///etc/passwd")
    assert not _host_allowed("ftp://en.wikipedia.org/foo")


def test_web_fetch_blocks_unlisted_host_even_without_policy():
    # Defense-in-depth: tool itself rejects, not just the policy engine.
    r = _run(WebFetchTool().run({"url": "https://evil.example.com/"}))
    assert not r.ok
    assert "allowlist" in r.error.lower()


def test_web_fetch_empty_url():
    r = _run(WebFetchTool().run({"url": ""}))
    assert not r.ok


def test_web_fetch_truncates_body(monkeypatch):
    """Force a >64 KiB response and verify the tool truncates it."""

    big = "x" * (100 * 1024)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=big)

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    tool = WebFetchTool(http_client=client)
    r = _run(tool.run({"url": "https://en.wikipedia.org/wiki/Test"}))
    asyncio.run(client.aclose())
    assert r.ok
    assert len(r.output) <= 64 * 1024
    assert r.metadata["truncated"] is True


# ── rag_search ──────────────────────────────────────────────────────────


class _StubHit:
    def __init__(self, text, score, doc_id, chunk_index):
        self.text = text
        self.score = score
        self.document_id = doc_id
        self.chunk_index = chunk_index


class _StubVectorStore:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query, top_k=3):
        return self._hits[:top_k]


def test_rag_search_returns_formatted_hits():
    vs = _StubVectorStore([_StubHit("hello world", 0.9, "doc1", 0)])
    r = _run(RagSearchTool(vs).run({"query": "hello"}))
    assert r.ok
    assert "hello world" in r.output
    assert "score=0.900" in r.output


def test_rag_search_handles_no_vectorstore():
    r = _run(RagSearchTool(None).run({"query": "anything"}))
    assert not r.ok
    assert "not enabled" in r.error.lower()


def test_rag_search_empty_query_rejected():
    r = _run(RagSearchTool(_StubVectorStore([])).run({"query": "  "}))
    assert not r.ok


def test_rag_search_invalid_top_k():
    r = _run(RagSearchTool(_StubVectorStore([])).run({"query": "x", "top_k": 99}))
    assert not r.ok


def test_rag_search_no_hits_is_ok():
    r = _run(RagSearchTool(_StubVectorStore([])).run({"query": "x"}))
    assert r.ok
    assert "no relevant" in r.output.lower()
