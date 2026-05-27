"""HTTP GET tool with a strict URL allowlist."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import httpx

from src.agents.tools.base import Tool, ToolResult


# Hosts the agent can fetch from. Adding a host is a deliberate code change,
# not a config tweak — allowlist edits are security-relevant.
ALLOWED_HOSTS: frozenset[str] = frozenset(
    {
        "en.wikipedia.org",
        "arxiv.org",
        "developer.mozilla.org",
        "docs.python.org",
    }
)

MAX_BYTES = 64 * 1024
TIMEOUT_SECONDS = 5.0


def _host_allowed(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    return parsed.netloc.lower() in ALLOWED_HOSTS


class WebFetchTool(Tool):
    name = "web_fetch"
    description = (
        "GET a URL from the curated allowlist (Wikipedia, arXiv, MDN, Python docs) "
        "and return the first 64 KiB of the response body. "
        "Use only when the user explicitly asks for a citation or external lookup."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Full https URL"},
        },
        "required": ["url"],
    }
    policies = ("max_arg_length", "url_allowlist", "injection_scan_output")

    def __init__(self, http_client: httpx.AsyncClient | None = None) -> None:
        self._client = http_client

    async def run(self, args: dict[str, Any]) -> ToolResult:
        url = args.get("url", "")
        if not isinstance(url, str) or not url:
            return ToolResult(ok=False, output="", error="url must be a non-empty string")

        # Defense in depth: PolicyEngine also enforces url_allowlist, but
        # the tool refuses too in case it's invoked directly.
        if not _host_allowed(url):
            return ToolResult(
                ok=False, output="", error=f"host not in allowlist: {urlparse(url).netloc}"
            )

        client = self._client or httpx.AsyncClient(timeout=TIMEOUT_SECONDS, follow_redirects=False)
        owns_client = self._client is None
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            body = resp.text[:MAX_BYTES]
            truncated = len(resp.text) > MAX_BYTES
            return ToolResult(
                ok=True,
                output=body,
                metadata={
                    "url": url,
                    "status_code": resp.status_code,
                    "bytes": len(body),
                    "truncated": truncated,
                },
            )
        except httpx.HTTPStatusError as exc:
            return ToolResult(
                ok=False, output="", error=f"HTTP {exc.response.status_code} from {url}"
            )
        except (httpx.RequestError, httpx.TimeoutException) as exc:
            return ToolResult(ok=False, output="", error=f"network error: {exc}")
        finally:
            if owns_client:
                await client.aclose()
