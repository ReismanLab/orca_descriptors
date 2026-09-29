from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bceg.geometry import make_start_geometry
from bceg.params import Configurable
from bceg.structure import Structure
from bceg.xyz import Conformer, write_xyz

SEED_RULE = "seed = seed_offset + replicate"

# Keys shared by searchers that start from a single 3D geometry (CREST, GOAT, molassembler).
# See bceg.geometry for the procedure.
START_DEFAULTS = {
    "start_geometry": "etkdg",  # "etkdg": new ETKDG embedding per replicate seed; "input": coords from the transformation
    "start_ff": "auto",  # relax the ETKDG start: auto (MMFF94s if parametrised, else none) | MMFF94 | MMFF94s | UFF | none
    "start_preopt": "gfn2",  # xtb pre-optimisation of the start: gfn2 | gfn1 | gfn0 | gfnff | none
    "max_metal_ligand_distance": 2.5,  # Angstrom; longer bonds to a metal reject the start geometry
    "start_max_attempts": 10,  # re-embeddings (new seeds) before giving up
    "xtb_executable": "xtb",
}
START_RESOURCE_KEYS = {"xtb_executable"}


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

    def start_geometry(self, structure: Structure, seed: int, workdir: Path) -> tuple[np.ndarray, dict]:
        """Checked (and optionally xtb pre-optimised) start geometry; also written to start_geometry.xyz."""
        coords, info = make_start_geometry(structure, seed, workdir, self.params)
        write_xyz(workdir / "start_geometry.xyz", structure.elements, [Conformer(coords, info["preopt_energy"])])
        return coords, info
