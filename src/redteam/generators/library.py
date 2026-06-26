"""Compose the full attack library from base templates × mutation chains."""

from __future__ import annotations

import hashlib

from src.redteam.generators.base import AttackRecord
from src.redteam.generators.mutations import MUTATIONS, apply
from src.redteam.generators.templates import ALL_TEMPLATES


# Which mutation chains to apply to every base attack. Adding new chains here
# is the cheap way to grow the library — each chain multiplies the count by
# the number of base templates.
DEFAULT_CHAINS: tuple[tuple[str, ...], ...] = (
    ("raw",),
    ("caps",),
    ("leet",),
    ("paraphrase",),
    ("embed",),
    ("b64",),
    ("multilingual",),
    ("caps", "paraphrase"),
    ("leet", "embed"),
)


def _attack_id(category: str, base: str, chain: tuple[str, ...]) -> str:
    """Stable id derived from inputs so the same attack reproduces day to day."""
    raw = f"{category}|{base}|{','.join(chain)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def build_library(
    chains: tuple[tuple[str, ...], ...] = DEFAULT_CHAINS,
    seed: int = 0,
) -> list[AttackRecord]:
    """Build the full attack set: every (category × base × chain) combo."""
    records: list[AttackRecord] = []
    for category, base_attacks in ALL_TEMPLATES.items():
        for base_text, expected_signals in base_attacks:
            for chain in chains:
                # Validate the chain before generating, so unknown mutations
                # blow up here rather than at runtime.
                for name in chain:
                    if name not in MUTATIONS:
                        raise ValueError(f"unknown mutation: {name!r}")

                payload = apply(base_text, chain, seed=seed)
                records.append(
                    AttackRecord(
                        attack_id=_attack_id(category, base_text, chain),
                        category=category,
                        base_template=base_text,
                        mutations=chain,
                        payload=payload,
                        expected_breach_signals=expected_signals,
                    )
                )
    return records


def category_counts(records: list[AttackRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in records:
        counts[r.category] = counts.get(r.category, 0) + 1
    return counts
