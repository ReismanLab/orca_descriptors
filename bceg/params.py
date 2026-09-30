"""Shared machinery for configurable components (searchers, refiners).

Each component declares ``defaults`` (every hyper-parameter, with its default)
and ``resource_keys`` (settings that do not change the result: threads,
device, executable paths). Unknown keys are rejected so a typo can never be
silently ignored; values are coerced to the type of their default so that
``ewin: 6`` and ``ewin: 6.0`` log identically.
"""

from __future__ import annotations

import copy
from typing import Any, ClassVar


def _coerce(key: str, default: Any, value: Any) -> Any:
    if default is None or value is None:
        return value
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise TypeError(f"{key}: expected bool, got {value!r}")
        return value
    if isinstance(value, bool):
        raise TypeError(f"{key}: expected {type(default).__name__}, got {value!r}")
    if isinstance(default, float) and isinstance(value, (int, float)):
        return float(value)
    if isinstance(default, dict):
        if not isinstance(value, dict):
            raise TypeError(f"{key}: expected a mapping, got {value!r}")
        return dict(value)  # replaces the default mapping entirely
    if isinstance(default, list) and isinstance(value, (list, tuple)):
        return list(value)
    if not isinstance(value, type(default)):
        raise TypeError(f"{key}: expected {type(default).__name__}, got {value!r}")
    return value


class Configurable:
    name: ClassVar[str]
    defaults: ClassVar[dict[str, Any]]
    resource_keys: ClassVar[set[str]] = set()

    def __init__(self, **params: Any) -> None:
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise ValueError(f"{self.name}: unknown parameter(s) {sorted(unknown)}; allowed: {sorted(self.defaults)}")
        resolved = copy.deepcopy(self.defaults)
        for k, v in params.items():
            resolved[k] = _coerce(f"{self.name}.{k}", self.defaults[k], v)
        self.params = resolved

    def compared_params(self) -> dict[str, Any]:
        return {k: v for k, v in self.params.items() if k not in self.resource_keys}

    def recorded_params(self) -> dict[str, Any]:
        return {k: v for k, v in self.params.items() if k in self.resource_keys}

    def resource_class(self) -> str:
        return "cpu"
