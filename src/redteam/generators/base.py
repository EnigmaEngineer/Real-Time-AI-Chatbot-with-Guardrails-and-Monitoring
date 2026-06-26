"""Attack record + generator contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


AttackCategory = Literal[
    "direct_override",     # "ignore previous instructions"
    "role_hijack",         # "you are now an unrestricted AI"
    "prompt_leak",         # extract the system prompt
    "tool_exfil",          # try to call a forbidden tool
    "url_exfil",           # try to fetch an off-allowlist URL
    "indirect_injection",  # payload framed as data/quotation
]


@dataclass
class AttackRecord:
    """One concrete attack ready to be fired at the agent."""

    attack_id: str
    category: AttackCategory
    base_template: str
    mutations: tuple[str, ...]
    payload: str
    expected_breach_signals: tuple[str, ...] = field(default_factory=tuple)

    def short(self) -> str:
        """Single-line view for logs."""
        return f"[{self.category}/{'+'.join(self.mutations) or 'raw'}] {self.payload[:60]}"
