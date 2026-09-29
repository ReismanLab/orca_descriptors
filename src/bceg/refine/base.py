from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bceg.params import Configurable
from bceg.structure import Structure
from bceg.xyz import Conformer


@dataclass
class RefineResult:
    conformers: list[Conformer]
    versions: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


class Refiner(Configurable):
    @property
    def lot(self) -> str:
        """Level-of-theory label used in ``raw_<i>_<lot>.xyz``."""
        raise NotImplementedError

    def run(self, structure: Structure, conformers: list[Conformer], workdir: Path) -> RefineResult:
        raise NotImplementedError
