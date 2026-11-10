"""Did a geometry keep the molecular graph it was generated from?

The conformer search (CREST with ``reftopo: false`` in particular), the post-optimisation and the DFT
optimisation can all move a structure to another isomer: a ligand decoordinates from the metal, two
hydrides reductively eliminate, a stereocentre inverts. :func:`topology_changes` compares one geometry with
the graph (``Structure.mol``, or ``<LIB>/<InChIKey>/structure.mol``; the atoms in xyz order) and lists every
difference by kind:

``decoordinated``      a graph bond to a metal longer than ``max_metal_ligand_distance`` (Ni-N, Ni-H, Ni-F)
``bond_broken``        any other graph bond longer than BOND_STRETCH_FACTOR x the covalent-radii sum
``new_bond``           two non-bonded non-metal atoms closer than the radii sum + ``new_bond_tolerance``
                       (H-H after reductive elimination, even when H2 stays sigma-bound to the metal; C-H, C-C)
``agostic``            a non-bonded metal...H contact shorter than ``agostic_max`` where the H is bonded to
                       C, Si or B in the graph (a C-H agostic interaction)
``new_metal_contact``  any other non-bonded metal...H contact shorter than ``agostic_max`` (N-H, O-H), or
                       metal...heavy atom contact shorter than ``max_metal_ligand_distance``
``stereo``             a stereocentre assigned in the graph has the other configuration in 3D

:func:`disallowed` drops the kinds a caller accepts (by default ``agostic``): what is left means the
geometry is another species.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem

from bceg.chem import ORGANIC_ELEMENTS, stereo_centers
from bceg.geometry import BOND_STRETCH_FACTOR, _with_coords, pair_geometry

KINDS = ("decoordinated", "bond_broken", "new_bond", "agostic", "new_metal_contact", "stereo")
#: the defaults of the ``topology`` blocks (pruning and numbering)
DEFAULTS = {"allow": ["agostic"], "max_metal_ligand_distance": 2.5, "agostic_max": 2.3, "new_bond_tolerance": 0.4}
AGOSTIC_PARTNERS = {"C", "Si", "B"}


@dataclass(frozen=True)
class Change:
    kind: str
    atoms: tuple[int, ...]
    distance: float | None
    message: str

    def __str__(self) -> str:
        return self.message


def check_params(block, where: str) -> dict:
    """A ``topology`` block with defaults filled in, validated. Raises ValueError."""
    if block is True:
        block = {}
    if not isinstance(block, dict):
        raise ValueError(f"{where}: expected a mapping (or true for the defaults), got {block!r}")
    unknown = set(block) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(DEFAULTS)}")
    out = {**DEFAULTS, **block}
    allow = out["allow"]
    if isinstance(allow, str):
        allow = [allow]
    if not isinstance(allow, list) or any(k not in KINDS for k in allow):
        raise ValueError(f"{where}.allow must be a list of {list(KINDS)}, got {out['allow']!r}")
    out["allow"] = sorted(set(allow), key=KINDS.index)
    for key in ("max_metal_ligand_distance", "agostic_max", "new_bond_tolerance"):
        v = out[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
            raise ValueError(f"{where}.{key} must be a positive number, got {v!r}")
        out[key] = float(v)
    return out


def _stereo_changes(mol: Chem.Mol, coords: np.ndarray) -> list[Change]:
    """Stereocentres assigned in the graph ('?' ones are ignored) whose 3D configuration differs."""
    tagged = {i: label for i, label in stereo_centers(mol) if label != "?"}
    if not tagged:
        return []
    from_3d = _with_coords(mol, coords)
    Chem.AssignStereochemistryFrom3D(from_3d, confId=from_3d.GetConformer().GetId(), replaceExistingTags=True)
    found = dict(stereo_centers(from_3d))
    return [Change("stereo", (i,), None, f"{mol.GetAtomWithIdx(i).GetSymbol()}{i} {label} -> {found.get(i, '?')} (stereo)")
            for i, label in sorted(tagged.items()) if found.get(i) != label]


def topology_changes(mol: Chem.Mol, coords: np.ndarray, max_metal_ligand_distance: float = 2.5,
                     agostic_max: float = 2.3, new_bond_tolerance: float = 0.4, stereo: bool = True) -> list[Change]:
    """Every difference between the geometry ``coords`` (Angstrom, atoms in ``mol``'s order) and ``mol``'s
    graph. Raises ValueError if the atom counts differ."""
    coords = np.asarray(coords, dtype=float)
    if coords.shape != (mol.GetNumAtoms(), 3):
        raise ValueError(f"geometry has {len(coords)} atoms, the graph {mol.GetNumAtoms()}")
    symbols, radii, dist, bonded = pair_geometry(mol, coords)
    metal = np.array([s not in ORGANIC_ELEMENTS for s in symbols])

    def label(i, j, sep="-"):
        return f"{symbols[i]}{i}{sep}{symbols[j]}{j}"

    out: list[Change] = []
    for b in mol.GetBonds():
        i, j = sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
        d = float(dist[i, j])
        if metal[i] or metal[j]:
            if d > max_metal_ligand_distance:
                out.append(Change("decoordinated", (i, j), d,
                                  f"{label(i, j)} {d:.2f} A > {max_metal_ligand_distance} A (decoordinated)"))
        elif d > BOND_STRETCH_FACTOR * (radii[i] + radii[j]):
            out.append(Change("bond_broken", (i, j), d, f"{label(i, j)} {d:.2f} A (bond broken)"))

    for i, j in zip(*np.triu_indices(len(symbols), k=1)):
        if bonded[i, j]:
            continue
        i, j, d = int(i), int(j), float(dist[i, j])
        if metal[i] or metal[j]:
            m, x = (i, j) if metal[i] else (j, i)
            if metal[x]:
                continue  # metal-metal pairs: not judged
            if symbols[x] == "H":  # an X-H pointing at the metal: judged at agostic_max, whatever X is
                if d >= agostic_max:
                    continue
                partners = {mol.GetAtomWithIdx(int(n)).GetSymbol() for n in np.flatnonzero(bonded[x])}
                if partners & AGOSTIC_PARTNERS:
                    out.append(Change("agostic", (m, x), d, f"{label(m, x, '...')} {d:.2f} A (agostic)"))
                else:
                    out.append(Change("new_metal_contact", (m, x), d,
                                      f"{label(m, x, '...')} {d:.2f} A < {agostic_max} A (new metal contact)"))
            elif d < max_metal_ligand_distance:
                out.append(Change("new_metal_contact", (m, x), d,
                                  f"{label(m, x, '...')} {d:.2f} A < {max_metal_ligand_distance} A (new metal contact)"))
        elif d < radii[i] + radii[j] + new_bond_tolerance:
            out.append(Change("new_bond", (i, j), d, f"{label(i, j)} {d:.2f} A (new bond)"))

    if stereo:
        out.extend(_stereo_changes(mol, coords))
    return out


def disallowed(changes: list[Change], allow=("agostic",)) -> list[Change]:
    """The changes whose kind is not in ``allow``: non-empty means the geometry is another species."""
    return [c for c in changes if c.kind not in allow]


def check(mol: Chem.Mol, coords: np.ndarray, params: dict) -> tuple[list[Change], list[Change]]:
    """(all changes, the disallowed ones) under a validated ``topology`` block (:func:`check_params`)."""
    changes = topology_changes(mol, coords, params["max_metal_ligand_distance"], params["agostic_max"],
                               params["new_bond_tolerance"])
    return changes, disallowed(changes, params["allow"])
