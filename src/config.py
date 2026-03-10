import os
import re
from pathlib import Path
from functools import lru_cache
from typing import Any

import yaml


_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)(?::-(.*?))?\}")


def _resolve_env_vars(value: str) -> str:
    """Replace ${VAR:-default} patterns with environment values."""

    def replacer(match: re.Match) -> str:
        var_name = match.group(1)
        default = match.group(2) if match.group(2) is not None else ""
        return os.environ.get(var_name, default)

    return _ENV_VAR_PATTERN.sub(replacer, value)


def _walk_and_resolve(obj: Any) -> Any:
    if isinstance(obj, str):
        resolved = _resolve_env_vars(obj)
        if resolved.lower() in ("true", "false"):
            return resolved.lower() == "true"
        try:
            return int(resolved)
        except ValueError:
            pass
        try:
            return float(resolved)
        except ValueError:
            pass
        return resolved
    if isinstance(obj, dict):
        return {k: _walk_and_resolve(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_and_resolve(item) for item in obj]
    return obj


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict:
    config_path = Path(path or os.environ.get("CONFIG_PATH", "config.yaml"))
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return _walk_and_resolve(raw)


def get_guardrail_profile(profile_name: str) -> dict:
    cfg = load_config()
    profiles = cfg["guardrails"]["profiles"]
    if profile_name not in profiles:
        raise ValueError(f"Unknown guardrail profile: {profile_name}. Available: {list(profiles.keys())}")
    return profiles[profile_name]
