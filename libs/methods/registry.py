"""Central framework method registry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import MethodSpec
from .dispatch import DUMMY_MERIT_ORDER_METHOD
from .economic import BFSA_DLMP_METHOD
from .opf import SOCP_METHOD
from .physical import BFSA_PHYSICAL_METHOD


METHOD_REGISTRY: dict[str, dict[str, MethodSpec]] = {
    "opf": {
        SOCP_METHOD.key: SOCP_METHOD,
    },
    "dispatch": {
        DUMMY_MERIT_ORDER_METHOD.key: DUMMY_MERIT_ORDER_METHOD,
    },
    "physical": {
        BFSA_PHYSICAL_METHOD.key: BFSA_PHYSICAL_METHOD,
    },
    "economic": {
        BFSA_DLMP_METHOD.key: BFSA_DLMP_METHOD,
    },
}


def list_method_specs(family: str | None = None) -> dict[str, dict[str, MethodSpec]] | dict[str, MethodSpec]:
    """Return registered methods, optionally limited to one family."""
    if family is None:
        return {family_key: dict(methods) for family_key, methods in METHOD_REGISTRY.items()}
    family_key = _normalize_key(family)
    return dict(METHOD_REGISTRY.get(family_key, {}))


def get_method_spec(family: str, method: str) -> MethodSpec:
    """Return a registered method spec or raise a useful error."""
    family_key = _normalize_key(family)
    method_key = _normalize_key(method)
    try:
        return METHOD_REGISTRY[family_key][method_key]
    except KeyError as exc:
        known = ", ".join(sorted(METHOD_REGISTRY.get(family_key, {}))) or "none"
        raise ValueError(f"Unknown {family_key} method '{method_key}'. Available: {known}") from exc


def build_method(family: str, method: str | None, options: Mapping[str, Any] | None = None):
    """Build a registered method instance. Returns None for disabled methods."""
    method_key = _normalize_key(method)
    if method_key in {"", "none"}:
        return None
    return get_method_spec(family, method_key).build(**dict(options or {}))





def _normalize_key(value: str | None) -> str:
    return str(value or "").strip().lower()
