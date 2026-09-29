"""Start geometries for single-start searchers (CREST, GOAT, molassembler).

Each replicate gets its own ETKDG embedding (seed = replicate seed), optionally
pre-optimised with xtb, then checked:

- every bond to a metal (e.g. Ni-N, Ni-F, Ni-H) must be <= ``max_metal_ligand_distance``;
- every other bond of the graph must stay < 1.35 x the covalent-radii sum;
- no two non-bonded atoms may be closer than the covalent-radii sum (e.g. collapsed H-H).

A rejected geometry is re-embedded with the next seed (``seed + attempt * 10007``),
so the result is deterministic per replicate while replicates still differ
(the N-Ni-N angle is free to vary).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rdkit import Chem

from bceg.chem import ORGANIC_ELEMENTS, embed
from bceg.structure import Structure
from bceg.xtb import xtb_optimize

SEED_STRIDE = 10007
BOND_STRETCH_FACTOR = 1.35
CLASH_FACTOR = 1.0


def coordination_problems(mol: Chem.Mol, coords: np.ndarray, max_metal_ligand_distance: float) -> list[str]:
    pt = Chem.GetPeriodicTable()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    radii = np.array([pt.GetRcovalent(a.GetAtomicNum()) for a in mol.GetAtoms()])
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    bonded = Chem.GetAdjacencyMatrix(mol).astype(bool)

    problems = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        d, label = dist[i, j], f"{symbols[i]}{i}-{symbols[j]}{j}"
        if symbols[i] not in ORGANIC_ELEMENTS or symbols[j] not in ORGANIC_ELEMENTS:
            if d > max_metal_ligand_distance:
                problems.append(f"{label} {d:.2f} A > {max_metal_ligand_distance} A (decoordinated)")
        elif d > BOND_STRETCH_FACTOR * (radii[i] + radii[j]):
            problems.append(f"{label} {d:.2f} A (bond broken)")
    iu = np.triu_indices(len(symbols), k=1)
    for i, j in zip(*iu):
        if not bonded[i, j] and dist[i, j] < CLASH_FACTOR * (radii[i] + radii[j]):
            problems.append(f"{symbols[i]}{i}...{symbols[j]}{j} {dist[i, j]:.2f} A (non-bonded clash)")
    return problems


def make_start_geometry(structure: Structure, seed: int, workdir: Path, params: dict) -> tuple[np.ndarray, dict]:
    """Return (coords, info) following the start_* parameters of a searcher."""
    rejected = []
    for attempt in range(params["start_max_attempts"]):
        s = seed + attempt * SEED_STRIDE
        if params["start_geometry"] == "input":
            if structure.coords is None:
                raise ValueError("start_geometry='input' but the transformation provided no coordinates")
            coords, source = structure.coords, "input"
        elif params["start_geometry"] == "etkdg":
            try:
                coords, source = embed(structure.mol, s, ff=params["start_ff"]), f"etkdg(seed={s}, ff={params['start_ff']})"
            except RuntimeError as exc:
                rejected.append({"seed": s, "problems": [str(exc)]})
                continue
        else:
            raise ValueError(f"Unknown start_geometry {params['start_geometry']!r}")

        energy = None
        if params["start_preopt"] != "none":
            coords, energy = xtb_optimize(
                structure.elements, coords, structure.charge, structure.uhf, workdir / f"preopt_{attempt}",
                method=params["start_preopt"], executable=params["xtb_executable"])
        problems = coordination_problems(structure.mol, coords, params["max_metal_ligand_distance"])
        if not problems:
            return coords, {"source": source, "seed": s, "attempts": attempt + 1, "preopt": params["start_preopt"],
                            "preopt_energy": energy, "rejected": rejected}
        rejected.append({"seed": s, "problems": problems})
        if params["start_geometry"] == "input":
            break
    raise RuntimeError(f"No acceptable start geometry after {len(rejected)} attempt(s): {rejected}")
