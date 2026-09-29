from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bceg.chem import embed
from bceg.params import Configurable
from bceg.structure import Structure
from bceg.xyz import Conformer

SEED_RULE = "seed = seed_offset + replicate"

# Keys shared by searchers that start from a single 3D geometry (CREST, GOAT, molassembler)
START_DEFAULTS = {
    "start_geometry": "etkdg",  # "etkdg": new ETKDG geometry per replicate seed; "input": coords from the transformation
    "start_ff": "auto",  # force field to relax the ETKDG start: auto | MMFF94 | MMFF94s | UFF | none
}


@dataclass
class SearchResult:
    conformers: list[Conformer]
    versions: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    log_files: list[Path] = field(default_factory=list)  # copied to <method>/logs/


class Searcher(Configurable):
    def seed(self, replicate: int) -> int:
        return self.params["seed_offset"] + replicate

    def applicable(self, structure: Structure) -> str | None:
        """Return a reason string if this searcher cannot handle ``structure``."""
        return None

    def run(self, structure: Structure, replicate: int, workdir: Path) -> SearchResult:
        raise NotImplementedError

    def start_geometry(self, structure: Structure, seed: int) -> tuple[np.ndarray, str]:
        mode = self.params["start_geometry"]
        if mode == "input":
            if structure.coords is None:
                raise ValueError("start_geometry='input' but the transformation provided no coordinates")
            return structure.coords, "input"
        if mode != "etkdg":
            raise ValueError(f"Unknown start_geometry {mode!r}")
        try:
            return embed(structure.mol, seed, ff=self.params["start_ff"]), f"etkdg(seed={seed}, ff={self.params['start_ff']})"
        except RuntimeError:
            if structure.coords is None:
                raise
            return structure.coords, "input (ETKDG embedding failed)"
