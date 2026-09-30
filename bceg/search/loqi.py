"""LoQI diffusion conformer generator (organics only): https://github.com/isayevlab/LoQI

Called through its CLI so it can live in its own environment (set ``executable``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rdkit import Chem

from bceg.chem import reorder_to_template
from bceg.external import find_executable, program_version, run
from bceg.search.base import SearchResult, Searcher
from bceg.structure import Structure
from bceg.xyz import Conformer


class LoQI(Searcher):
    name = "loqi"
    defaults = {
        "n_confs": 100,
        "extra_args": [],  # additional CLI arguments (e.g. model/batch options)
        "seed_offset": 0,
        "executable": "loqi",
    }
    resource_keys = {"executable"}

    def resource_class(self) -> str:
        return "gpu"

    def applicable(self, structure: Structure) -> str | None:
        if not structure.is_organic:
            return "LoQI is trained on organic (ChEMBL) molecules only"
        return None

    def run(self, structure: Structure, replicate: int, workdir: Path) -> SearchResult:
        exe = find_executable(self.params["executable"])
        cmd = [exe, "sample", "--smiles", structure.smiles, "--n-confs", str(self.params["n_confs"]),
               "--output", "confs.sdf", *self.params["extra_args"]]
        proc = run(cmd, workdir, log_name="loqi.out")
        sdf = workdir / "confs.sdf"
        if proc.returncode != 0 or not sdf.exists():
            raise RuntimeError(f"LoQI failed (exit {proc.returncode}):\n{proc.stdout[-3000:]}")

        confs = []
        for mol in Chem.SDMolSupplier(str(sdf), removeHs=False):
            if mol is None:
                continue
            order = reorder_to_template(mol, structure.mol)
            confs.append(Conformer(np.array(mol.GetConformer().GetPositions())[order]))
        if not confs:
            raise RuntimeError("LoQI returned no readable conformers")
        return SearchResult(
            confs,
            versions={"loqi": program_version([exe, "--version"], match="") or "unknown"},
            details={"seed": None, "seed_note": "LoQI CLI seeding not exposed; replicates differ by sampling"},
            log_files=[workdir / "loqi.out"],
        )
