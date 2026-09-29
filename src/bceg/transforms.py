"""SMILES transformations (depth-1 folders), e.g. N,N-ligand -> LNiF2.

A transformation is a Python file defining::

    def transform(smiles: str) -> str | dict | Structure | None

returning ``None`` to skip a molecule, or the product SMILES, a dict of ``Structure`` fields
(``smiles``, ``charge``, ``multiplicity``, ``coords``, ``metadata``), or a
``Structure``. The input is the canonical SMILES of the input molecule.

NOTE: the interface will be finalised against the sample scripts.
"""

from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from bceg.structure import Structure

IDENTITY = "none"


@dataclass
class Transformation:
    name: str
    script: Path | None = None

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
        return {"name": self.name, "script": None if self.script is None else self.script.name, "script_sha256": self.sha256}

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
        if self.script is None:
            return Structure(canonical_smiles)
        out = self._load()(canonical_smiles)
        if out is None or isinstance(out, Structure):
            return out
        if isinstance(out, str):
            return Structure(out)
        if isinstance(out, dict):
            return Structure(**out)
        raise TypeError(f"{self.script}: transform() returned {type(out).__name__}")
