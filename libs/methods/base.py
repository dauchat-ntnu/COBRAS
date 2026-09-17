"""Method registry primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from libs.shared import MethodCapabilities


MethodBuilder = Callable[..., Any]


@dataclass(frozen=True)
class MethodSpec:
    """Metadata and builder for a registered framework method."""

    family: str
    key: str
    label: str
    builder: MethodBuilder
    capabilities: MethodCapabilities
    description: str = ""
    default_options: dict[str, Any] = field(default_factory=dict)
    option_schema: dict[str, Any] = field(default_factory=dict)

    def build(self, **options: Any) -> Any:
        merged_options = dict(self.default_options)
        merged_options.update(options)
        return self.builder(**merged_options)
