"""SMILES transformations (depth-1 folders), e.g. N,N-ligand -> LNiF2.

A transformation is a Python file defining::

    def transform(smiles: str, **params) -> str | dict | Structure | None

returning ``None`` to skip a molecule, or the product SMILES, a dict of ``Structure`` fields
(``smiles``, ``charge``, ``multiplicity``, ``coords``, ``metadata``), or a
``Structure``. The input is the canonical SMILES of the input molecule; ``params``
come from the config, so one script can serve several transformations::

    transformations:
      NiF2: {script: ../transforms/ni_x2.py, params: {x: F}, multiplicity: 1}

``charge`` / ``multiplicity`` given in the config override whatever the script
returns (and also apply to ``none``). Multiplicity defaults to 1.
"""

from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from bceg.structure import Structure

IDENTITY = "none"


@dataclass
class Transformation:
    name: str
    script: Path | None = None
    params: dict[str, Any] = field(default_factory=dict)
    charge: int | None = None  # overrides the charge returned by the script / formal charge
    multiplicity: int | None = None  # overrides the multiplicity returned by the script (default 1)

    def __post_init__(self) -> None:
        self._fn: Callable | None = None
        if self.script is not None:
            self.script = Path(self.script)
            if not self.script.exists():
                raise FileNotFoundError(f"Transformation script not found: {self.script}")

    @property
    def sha256(self) -> str | None:
        return None if self.script is None else hashlib.sha256(self.script.read_bytes()).hexdigest()

    def identity(self) -> dict:
        return {"name": self.name, "script": None if self.script is None else self.script.name,
                "script_sha256": self.sha256, "params": self.params,
                "charge": self.charge, "multiplicity": self.multiplicity}

    def _load(self) -> Callable:
        if self._fn is None:
            spec = importlib.util.spec_from_file_location(f"bceg_transform_{self.name}", self.script)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if not hasattr(module, "transform"):
                raise AttributeError(f"{self.script} does not define transform(smiles)")
            self._fn = module.transform
        return self._fn

    def apply(self, canonical_smiles: str) -> Structure | None:
        """``None`` means the transformation does not apply to this molecule (it is skipped)."""
        out = canonical_smiles if self.script is None else self._load()(canonical_smiles, **self.params)
        if out is None:
            return None
        if isinstance(out, str):
            out = {"smiles": out}
        if isinstance(out, dict):
            out = dict(out)
            if self.charge is not None:
                out["charge"] = self.charge
            if self.multiplicity is not None:
                out["multiplicity"] = self.multiplicity
            return Structure(**out)
        if isinstance(out, Structure):
            if self.charge is not None:
                out.charge = self.charge
            if self.multiplicity is not None:
                out.multiplicity = self.multiplicity
            out.check_spin()
            return out
        raise TypeError(f"{self.script}: transform() returned {type(out).__name__}")
