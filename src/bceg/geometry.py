"""Start geometries for single-start searchers (CREST, GOAT, molassembler).

Each replicate gets its own ETKDG embedding (seed = replicate seed), optionally
pre-optimised with xtb, then checked:

- every bond to a metal (e.g. Ni-N, Ni-F, Ni-H) must be <= ``max_metal_ligand_distance``;
- every other bond of the graph must stay < 1.35 x the covalent-radii sum;
- no two non-bonded atoms may be closer than the covalent-radii sum (e.g. collapsed H-H).

A rejected geometry is re-embedded with the next seed (``seed + attempt * 10007``),
so the result is deterministic per replicate.

The full pre-optimisation (``start_preopt_level: normal``) takes rigid chelates
(e.g. bpy-NiF2) to the same minimum from every seed. This is accepted: the N-Ni-N bite
angle varies between ligands but should be consistent for one ligand, and replicate
variability comes from the searches. The info returned records the RMSD to the raw
embedding and the geometry at each metal (angles, ``tau4``) for monitoring.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from rdkit.Geometry import Point3D

from bceg.chem import ORGANIC_ELEMENTS, embed
from bceg.external import resolve_threads
from bceg.structure import Structure
from bceg.xtb import xtb_identity, xtb_optimize

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


def _with_coords(mol: Chem.Mol, coords: np.ndarray) -> Chem.Mol:
    m = Chem.Mol(mol)
    m.RemoveAllConformers()
    conf = Chem.Conformer(m.GetNumAtoms())
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    m.AddConformer(conf)
    return m


def heavy_atom_rmsd(mol: Chem.Mol, a: np.ndarray, b: np.ndarray) -> float:
    """Symmetry-aware heavy-atom RMSD (Angstrom) between two geometries of ``mol``."""
    ma = Chem.RemoveHs(_with_coords(mol, a), sanitize=False)
    mb = Chem.RemoveHs(_with_coords(mol, b), sanitize=False)
    return float(rdMolAlign.GetBestRMS(ma, mb, maxMatches=10000))


def metal_geometry(mol: Chem.Mol, coords: np.ndarray) -> dict[str, dict]:
    """Ligand-metal-ligand angles and, for 4-coordinate metals, tau4 (0 square planar, 1 tetrahedral)."""
    out = {}
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in ORGANIC_ELEMENTS:
            continue
        m, nbrs = atom.GetIdx(), [n.GetIdx() for n in atom.GetNeighbors()]
        angles = {}
        for k, i in enumerate(nbrs):
            for j in nbrs[k + 1:]:
                u, v = coords[i] - coords[m], coords[j] - coords[m]
                cos = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
                label = f"{mol.GetAtomWithIdx(i).GetSymbol()}{i}-{atom.GetSymbol()}{m}-{mol.GetAtomWithIdx(j).GetSymbol()}{j}"
                angles[label] = round(float(np.degrees(np.arccos(np.clip(cos, -1, 1)))), 1)
        info = {"angles_deg": angles}
        if len(nbrs) == 4:
            a, b = sorted(angles.values())[-2:]
            info["tau4"] = round((360.0 - a - b) / 141.0, 3)
        out[f"{atom.GetSymbol()}{m}"] = info
    return out


def make_start_geometry(structure: Structure, seed: int, workdir: Path, params: dict) -> tuple[np.ndarray, dict]:
    """Return (coords, info) following the start_* parameters of a searcher."""
    rejected = []
    preopt = params["start_preopt"]
    # which xtb ran is not in the params hash (it is a resource key), so record it here
    xtb_path, xtb_ver = xtb_identity(params["xtb_executable"]) if preopt != "none" else (None, None)
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

        energy, embedded = None, coords
        if params["start_preopt"] != "none":
            coords, energy = xtb_optimize(
                structure.elements, coords, structure.charge, structure.uhf, workdir / f"preopt_{attempt}",
                method=params["start_preopt"], level=params["start_preopt_level"],
                executable=params["xtb_executable"], threads=resolve_threads(params.get("threads", params.get("nprocs"))))
        problems = coordination_problems(structure.mol, coords, params["max_metal_ligand_distance"])
        if not problems:
            return coords, {
                "source": source, "seed": s, "attempts": attempt + 1, "preopt": preopt,
                "preopt_level": params["start_preopt_level"], "preopt_energy": energy, "rejected": rejected,
                "preopt_program": xtb_path, "preopt_version": xtb_ver,
                "rmsd_preopt_vs_embedded": round(heavy_atom_rmsd(structure.mol, embedded, coords), 3),
                "metal_geometry": metal_geometry(structure.mol, coords),
            }
        rejected.append({"seed": s, "problems": problems})
        if params["start_geometry"] == "input":
            break
    raise RuntimeError(f"No acceptable start geometry after {len(rejected)} attempt(s): {rejected}")
