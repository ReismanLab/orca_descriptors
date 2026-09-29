from __future__ import annotations

from pathlib import Path

import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

from bceg.chem import ff_for
from bceg.search.base import SearchResult, Searcher
from bceg.structure import Structure
from bceg.xyz import HARTREE_TO_KCAL, Conformer


class RDKitETKDG(Searcher):
    """ETKDGv3 embedding, optionally followed by force-field optimisation (energies then in Hartree)."""

    name = "rdkit_etkdg"
    defaults = {
        "n_confs": 300,
        "prune_rms": -1.0,
        "ff_optimize": "none",  # none | auto | MMFF94 | MMFF94s | UFF
        "max_iters": 2000,
        "use_small_ring_torsions": True,
        "use_macrocycle_torsions": True,
        "use_random_coords": False,
        "seed_offset": 0,
        "threads": 0,
    }
    resource_keys = {"threads"}

    def run(self, structure: Structure, replicate: int, workdir: Path) -> SearchResult:
        p = self.params
        mol = Chem.Mol(structure.mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = self.seed(replicate)
        params.pruneRmsThresh = p["prune_rms"]
        params.numThreads = p["threads"]
        params.useSmallRingTorsions = p["use_small_ring_torsions"]
        params.useMacrocycleTorsions = p["use_macrocycle_torsions"]
        params.useRandomCoords = p["use_random_coords"]
        params.enforceChirality = True
        cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=p["n_confs"], params=params))
        if not cids:
            raise RuntimeError("ETKDG embedded no conformers")

        ff = ff_for(mol) if p["ff_optimize"] == "auto" else p["ff_optimize"]
        energies = [None] * len(cids)
        converged = [None] * len(cids)
        if ff.startswith("MMFF"):
            res = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=p["threads"], maxIters=p["max_iters"], mmffVariant=ff)
        elif ff == "UFF":
            res = AllChem.UFFOptimizeMoleculeConfs(mol, numThreads=p["threads"], maxIters=p["max_iters"])
        elif ff == "none":
            res = None
        else:
            raise ValueError(f"Unknown ff_optimize {ff!r}")
        if res is not None:
            converged = [nc == 0 for nc, _ in res]
            energies = [e / HARTREE_TO_KCAL for _, e in res]

        confs = []
        for cid, e, conv in zip(cids, energies, converged):
            info = {} if conv is None else {"ff_converged": conv}
            confs.append(Conformer(np.array(mol.GetConformer(cid).GetPositions()), e, info))
        return SearchResult(
            confs,
            versions={"rdkit": rdkit.__version__},
            details={"seed": self.seed(replicate), "ff_used": ff, "n_embedded": len(cids)},
        )
