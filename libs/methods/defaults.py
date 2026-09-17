"""Default method configuration loaded from config/default.yaml."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "config" / "default.yaml"


@lru_cache(maxsize=1)
def load_default_config() -> dict[str, Any]:
    """Load project defaults from config/default.yaml."""
    if not DEFAULT_CONFIG_PATH.exists():
        return {}
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def get_method_config(family: str, method: str) -> dict[str, Any]:
    """Return the config block for one registered method."""
    config = load_default_config()
    methods = config.get("methods", {})
    family_config = methods.get(str(family), {}) if isinstance(methods, dict) else {}
    method_config = family_config.get(str(method), {}) if isinstance(family_config, dict) else {}
    return deepcopy(method_config if isinstance(method_config, dict) else {})


def get_method_defaults(family: str, method: str) -> dict[str, Any]:
    """Return default_options for one registered method."""
    return deepcopy(get_method_config(family, method).get("default_options", {}) or {})


def get_method_option_schema(family: str, method: str) -> dict[str, Any]:
    """Return option_schema for one registered method."""
    return deepcopy(get_method_config(family, method).get("option_schema", {}) or {})
