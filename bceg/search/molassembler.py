"""SCINE Molassembler distance-geometry ensembles (suited to transition-metal complexes).

The molecular graph comes from the RDKit template; local shapes and stereo
(e.g. square-planar vs tetrahedral Ni) are perceived from the xtb pre-optimised
start geometry, which differs per replicate (set ``fixed_start_seed: true`` to
use one start geometry, seeded by ``seed_offset``, for all replicates).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bceg.search.base import START_DEFAULTS, START_RESOURCE_KEYS, SearchResult, Searcher
from bceg.structure import Structure
from bceg.xyz import BOHR_TO_ANGSTROM, Conformer


class Molassembler(Searcher):
    name = "molassembler"
    defaults = {
        "n_confs": 100,  # or a formula in the rotatable-bond count, e.g. "(n_rot + 2) * 30"
        "dg_config": {},  # attributes of scine_molassembler.dg.Configuration
        "fixed_start_seed": False,
        **START_DEFAULTS,
        "seed_offset": 0,
    }
    resource_keys = START_RESOURCE_KEYS

    def run(self, structure: Structure, replicate: int, workdir: Path) -> SearchResult:
        import scine_molassembler as masm
        import scine_utilities as utils

        seed = self.seed(replicate)
        start_seed = self.params["seed_offset"] if self.params["fixed_start_seed"] else seed
        coords, start = self.start_geometry(structure, start_seed, workdir)

        elements = [getattr(utils.ElementType, el) for el in structure.elements]
        atoms = utils.AtomCollection(elements, np.asarray(coords) / BOHR_TO_ANGSTROM)
        bonds = utils.BondOrderCollection(len(elements))
        for b in structure.mol.GetBonds():
            bonds.set_order(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), max(b.GetBondTypeAsDouble(), 1.0))
        interpreted = masm.interpret.molecules(atoms, bonds, masm.interpret.BondDiscretization.Binary)
        if len(interpreted.molecules) != 1:
            raise RuntimeError(f"Graph interpreted as {len(interpreted.molecules)} molecules, expected 1")
        molecule = interpreted.molecules[0]
        if [str(e) for e in molecule.graph.elements()] != [str(e) for e in elements]:
            raise RuntimeError("Molassembler reordered atoms")

        config = masm.dg.Configuration()
        for k, v in self.params["dg_config"].items():
            setattr(config, k, v)
        n_confs, n_rot = self.n_confs(structure)
        results = masm.dg.generate_ensemble(molecule, n_confs, seed, config)
        confs = [Conformer(np.asarray(r) * BOHR_TO_ANGSTROM) for r in results if isinstance(r, np.ndarray)]
        if not confs:
            raise RuntimeError("Molassembler generated no conformers")
        return SearchResult(
            confs,
            versions={"scine_molassembler": getattr(masm, "__version__", "unknown")},
            details={"seed": seed, "start_geometry": start, "n_rot": n_rot, "n_confs": n_confs,
                     "n_failed": len(results) - len(confs)},
            log_files=[workdir / "start_geometry.xyz"],
        )
