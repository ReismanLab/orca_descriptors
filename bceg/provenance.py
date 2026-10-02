"""Hyper-parameter logs (params.json) and the skip / rename rule.

A params file has two parts:
  - ``compared``: everything that determines the ensemble. Must match exactly
    for existing results to be reused.
  - ``recorded``: resource settings (threads, device, ...) kept for reference only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


class ParamsMismatch(RuntimeError):
    pass


def normalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): normalize(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [normalize(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def digest(compared: dict) -> str:
    blob = json.dumps(normalize(compared), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def make_params(compared: dict, recorded: dict | None = None) -> dict:
    compared = normalize(compared)
    return {"sha256": digest(compared), "compared": compared, "recorded": normalize(recorded or {})}


def flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(d, dict):
        out = {}
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}{k}."))
        return out or {prefix.rstrip("."): {}}
    return {prefix.rstrip("."): d}


def diff(old: dict, new: dict) -> list[str]:
    a, b = flatten(old), flatten(new)
    lines = []
    for key in sorted(set(a) | set(b)):
        if key not in a:
            lines.append(f"  + {key}: {b[key]!r}")
        elif key not in b:
            lines.append(f"  - {key}: {a[key]!r}")
        elif a[key] != b[key]:
            lines.append(f"  ~ {key}: {a[key]!r} -> {b[key]!r}")
    return lines


def check_or_write(path: Path, params: dict, rename_hint: str = "", write: bool = True) -> str:
    """Create ``path`` if absent (atomically), else require the compared parameters to match.

    Returns "new" or "match"; raises ``ParamsMismatch`` with a diff otherwise.
    With ``write=False`` nothing is created (read-only check).
    """
    if not path.exists():
        if not write:
            return "new"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "x") as fh:  # exclusive create: concurrent array tasks cannot both write
                json.dump(params, fh, indent=2, sort_keys=True)
            return "new"
        except FileExistsError:
            pass
    old = json.loads(path.read_text())
    if old.get("sha256") == params["sha256"] and old.get("compared") == params["compared"]:
        return "match"
    lines = diff(old.get("compared", {}), params["compared"])
    raise ParamsMismatch(
        f"Hyper-parameters differ from the existing log {path}:\n" + "\n".join(lines)
        + f"\nExisting results were generated with different parameters. {rename_hint}".rstrip()
    )


def read_record(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def write_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(normalize(record), indent=2))
    tmp.replace(path)
