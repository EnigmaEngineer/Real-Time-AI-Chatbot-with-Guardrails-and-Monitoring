"""Tool base class. Subclasses declare name, schema, and policies; the
coordinator wires them into the ReAct loop."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    ok: bool
    output: str
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class Tool(ABC):
    # Subclasses override these as class attributes.
    name: str = ""
    description: str = ""
    input_schema: dict[str, Any] = {}
    policies: tuple[str, ...] = ()

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.name:
            raise TypeError(f"Tool subclass {cls.__name__} must set a non-empty `name`")
        if not cls.description:
            raise TypeError(f"Tool subclass {cls.__name__} must set a non-empty `description`")

    @abstractmethod
    async def run(self, args: dict[str, Any]) -> ToolResult:
        ...

    def describe_for_prompt(self) -> str:
        params = ", ".join(self.input_schema.get("properties", {}).keys())
        return f"- {self.name}({params}): {self.description}"
