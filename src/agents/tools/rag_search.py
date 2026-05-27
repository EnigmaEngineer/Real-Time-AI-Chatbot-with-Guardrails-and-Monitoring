"""Semantic search over the ChromaDB vector store, exposed as an agent tool."""

from __future__ import annotations

from typing import Any

from src.agents.tools.base import Tool, ToolResult


class RagSearchTool(Tool):
    name = "rag_search"
    description = (
        "Search the ingested document collection by semantic similarity. "
        "Returns up to top_k passages with a relevance score. "
        "Use when the user asks about specific facts that may be in uploaded docs."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language search query"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 3},
        },
        "required": ["query"],
    }
    # injection_scan_output runs after the tool returns. Retrieved content
    # can contain prompt-injection payloads, so we scan before passing to the LLM.
    policies = ("max_arg_length", "no_pii_in_args", "injection_scan_output")

    def __init__(self, vectorstore: Any) -> None:
        self._vs = vectorstore

    async def run(self, args: dict[str, Any]) -> ToolResult:
        if self._vs is None:
            return ToolResult(
                ok=False,
                output="",
                error="RAG is not enabled on this deployment (vectorstore unavailable)",
            )

        query = args.get("query", "")
        top_k = int(args.get("top_k", 3))
        if not isinstance(query, str) or not query.strip():
            return ToolResult(ok=False, output="", error="query must be a non-empty string")
        if top_k < 1 or top_k > 10:
            return ToolResult(ok=False, output="", error="top_k must be between 1 and 10")

        try:
            hits = self._vs.search(query, top_k=top_k)
        except Exception as exc:
            return ToolResult(ok=False, output="", error=f"vectorstore error: {exc}")

        if not hits:
            return ToolResult(
                ok=True,
                output="No relevant passages found in the ingested documents.",
                metadata={"hits": 0},
            )

        lines = []
        for i, h in enumerate(hits, start=1):
            text = (h.text or "")[:500].replace("\n", " ").strip()
            lines.append(f"[{i}] (score={h.score:.3f}, doc={h.document_id}) {text}")
        return ToolResult(
            ok=True,
            output="\n".join(lines),
            metadata={"hits": len(hits), "query": query},
        )
