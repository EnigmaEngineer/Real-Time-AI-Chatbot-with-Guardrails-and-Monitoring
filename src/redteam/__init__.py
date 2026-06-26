"""Adversarial red-team automation. Generates synthetic jailbreak attempts,
fires them at a running agent, classifies which got through."""

from src.redteam.classifier import DefenseClassifier, Verdict
from src.redteam.generators.base import AttackRecord
from src.redteam.generators.library import build_library
from src.redteam.runner import AttackRunner, RunResult

__all__ = [
    "AttackRecord",
    "AttackRunner",
    "DefenseClassifier",
    "RunResult",
    "Verdict",
    "build_library",
]
