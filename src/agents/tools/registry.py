"""Tool registry: holds tool instances and per-profile allowlists."""

from __future__ import annotations

from src.agents.tools.base import Tool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._allowlists: dict[str, set[str]] = {}

    # ── registration ────────────────────────────────────────────────────

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} already registered")
        self._tools[tool.name] = tool

    def allow(self, profile: str, tool_names: list[str]) -> None:
        unknown = set(tool_names) - set(self._tools)
        if unknown:
            raise ValueError(f"cannot allow unregistered tools: {sorted(unknown)}")
        self._allowlists.setdefault(profile, set()).update(tool_names)

    # ── lookup ──────────────────────────────────────────────────────────

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def allowed_for(self, profile: str) -> list[Tool]:
        names = self._allowlists.get(profile, set())
        return [self._tools[n] for n in sorted(names)]

    def is_allowed(self, profile: str, tool_name: str) -> bool:
        return tool_name in self._allowlists.get(profile, set())

    def describe_for_prompt(self, profile: str) -> str:
        tools = self.allowed_for(profile)
        if not tools:
            return "(no tools available)"
        return "\n".join(t.describe_for_prompt() for t in tools)


def get_default_registry(vectorstore=None, http_client=None) -> ToolRegistry:
    """Build the registry used by the production app.

    Profiles:
      - default:   calculator, rag_search, web_fetch
      - readonly:  calculator, rag_search
      - math_only: calculator
    """
    from src.agents.tools.calculator import CalculatorTool
    from src.agents.tools.rag_search import RagSearchTool
    from src.agents.tools.web_fetch import WebFetchTool

    reg = ToolRegistry()
    reg.register(CalculatorTool())
    reg.register(RagSearchTool(vectorstore))
    reg.register(WebFetchTool(http_client))

    reg.allow("default", ["calculator", "rag_search", "web_fetch"])
    reg.allow("readonly", ["calculator", "rag_search"])
    reg.allow("math_only", ["calculator"])
    return reg
