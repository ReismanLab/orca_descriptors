"""Annotators: optional per-conformer columns derived from the molecule and its numbering.

An annotator is ``fn(mol, indices0) -> dict[str, value]`` where ``indices0`` is the
``label -> 0-based index`` map produced by the numbering backend. Annotators never change
indices; orientation belongs to SMARTS stages or ``orient`` rules.

``lnih2_ligand_class`` reproduces the taxonomy of the former ``atom_numbering.py``:

* ``ligand_type``  ``core_5`` / ``core_6`` from the ring size the metal sits in
* ``ligand_class`` ``pyox``, ``biox``, ``biim``, ``bpy`` (5-membered metallacycle) or
  ``box``, ``pyr6`` (6-membered metallacycle), ``unknown`` otherwise
"""
from __future__ import annotations

from typing import Callable, Mapping

from rdkit import Chem

ANNOTATORS: dict[str, Callable[[object, Mapping[str, int]], dict]] = {}


def register_annotator(name: str):
    def deco(fn):
        ANNOTATORS[name] = fn
        return fn
    return deco


def _without_metal_donor_bonds(mol, metal_idx: int):
    """Copy of ``mol`` with metal-N bonds removed so RDKit does not merge the chelate ring."""
    rw = Chem.RWMol(mol)
    metal = rw.GetAtomWithIdx(metal_idx)
    for nb in list(metal.GetNeighbors()):
        if nb.GetAtomicNum() == 7:
            rw.RemoveBond(metal_idx, nb.GetIdx())
    m = rw.GetMol()
    m.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(m)
    return m


@register_annotator("lnih2_ligand_class")
def lnih2_ligand_class(mol, indices0: Mapping[str, int], metal_label: str = "Ni") -> dict:
    need = {metal_label, "N1", "N2", "C1", "C2"}
    if not need.issubset(indices0):
        return {"ligand_type": None, "ligand_class": None}
    ni = int(indices0[metal_label])
    ni_atom = mol.GetAtomWithIdx(ni)
    in5, in6 = ni_atom.IsInRingSize(5), ni_atom.IsInRingSize(6)
    if in5 and not in6:
        ligand_type = "core_5"
    elif in6 and not in5:
        ligand_type = "core_6"
    else:
        ligand_type = "unknown"

    m = _without_metal_donor_bonds(mol, ni)
    n1, n2, c1, c2 = (int(indices0[k]) for k in ("N1", "N2", "C1", "C2"))
    n1_6, n1_5 = m.GetAtomWithIdx(n1).IsInRingSize(6), m.GetAtomWithIdx(n1).IsInRingSize(5)
    n2_6, n2_5 = m.GetAtomWithIdx(n2).IsInRingSize(6), m.GetAtomWithIdx(n2).IsInRingSize(5)

    def neighbours(idx):
        return [a.GetAtomicNum() for a in m.GetAtomWithIdx(idx).GetNeighbors()]

    ligand_class = "unknown"
    if ligand_type == "core_5":
        if n1_6 and n2_6:
            ligand_class = "bpy"
        elif n1_5 and n2_5:
            if 8 in neighbours(c1) or 8 in neighbours(c2):
                ligand_class = "biox"
            elif neighbours(c1).count(7) == 2 or neighbours(c2).count(7) == 2:
                ligand_class = "biim"
        elif (n1_6 and n2_5) or (n2_6 and n1_5):
            ligand_class = "pyox"
    elif ligand_type == "core_6":
        ligand_class = "box" if (n1_6 and n1_5 and n2_6 and n2_5) else "pyr6"
    return {"ligand_type": ligand_type, "ligand_class": ligand_class}
