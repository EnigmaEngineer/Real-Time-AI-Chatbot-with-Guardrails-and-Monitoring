"""A/B testing router: deterministic variant assignment and experiment tracking."""

import hashlib
import time
from dataclasses import dataclass, field

from src.monitoring.metrics import AB_ASSIGNMENT_COUNTER
from src.monitoring.logging import logger


@dataclass
class VariantConfig:
    model: str
    weight: float
    system_prompt: str
    guardrail_profile: str


@dataclass
class Assignment:
    experiment: str
    variant_name: str
    variant: VariantConfig
    timestamp: float = field(default_factory=time.time)


@dataclass
class ExperimentRecord:
    experiment: str
    variant: str
    user_id: str
    latency_ms: float
    feedback: int | None  # 1 = up, -1 = down, None = no feedback
    token_count: int
    timestamp: float


class ABRouter:
    def __init__(self, config: dict):
        ab_cfg = config.get("abtesting", {})
        self.enabled = ab_cfg.get("enabled", False)
        self.experiments: dict[str, dict] = {}
        self._records: list[ExperimentRecord] = []

        if self.enabled:
            for name, exp in ab_cfg.get("experiments", {}).items():
                if exp.get("active", False):
                    self.experiments[name] = exp

    def assign(self, user_id: str, experiment_name: str | None = None) -> Assignment | None:
        if not self.enabled or not self.experiments:
            return None

        exp_name = experiment_name or next(iter(self.experiments))
        exp = self.experiments.get(exp_name)
        if not exp:
            return None

        variants = exp["variants"]
        bucket = self._hash_to_bucket(user_id, exp_name)
        cumulative = 0.0
        chosen_name = None
        chosen_cfg = None

        for vname, vcfg in variants.items():
            cumulative += vcfg["weight"]
            if bucket < cumulative:
                chosen_name = vname
                chosen_cfg = vcfg
                break

        if chosen_name is None:
            chosen_name = list(variants.keys())[-1]
            chosen_cfg = variants[chosen_name]

        AB_ASSIGNMENT_COUNTER.labels(experiment=exp_name, variant=chosen_name).inc()
        logger.info(
            f"A/B assignment: {exp_name}/{chosen_name}",
            extra={"experiment": exp_name, "variant": chosen_name, "user_id": user_id},
        )

        return Assignment(
            experiment=exp_name,
            variant_name=chosen_name,
            variant=VariantConfig(
                model=chosen_cfg["model"],
                weight=chosen_cfg["weight"],
                system_prompt=chosen_cfg["system_prompt"],
                guardrail_profile=chosen_cfg["guardrail_profile"],
            ),
        )

    def record(self, record: ExperimentRecord) -> None:
        self._records.append(record)

    def get_records(self, experiment: str | None = None) -> list[ExperimentRecord]:
        if experiment:
            return [r for r in self._records if r.experiment == experiment]
        return list(self._records)

    @staticmethod
    def _hash_to_bucket(user_id: str, experiment: str) -> float:
        """Deterministic hash → [0, 1) for consistent assignment."""
        h = hashlib.sha256(f"{user_id}:{experiment}".encode()).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF
