"""File-level helpers for the numbering stage (moved from the former ``atom_numbering.py``).

* xyz -> sdf conversion with Open Babel (bond perception for RDKit)
* loading the sdf as an RDKit molecule with explicit hydrogens
* SMILES of the whole molecule or of the ligand after stripping labelled atoms
* SCF energy from the ORCA property JSON via ``OrcaReader``
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from rdkit import Chem
from rdkit.Chem import GetMolFrags

import desc_utils.orca_descriptors as od


def get_sdf_from_xyz(conf_dir: str | Path, geometry_file: str = "optimization.xyz", overwrite: bool = False) -> Path:
    """Write ``<stem>.sdf`` next to the xyz file (Open Babel bond perception). Returns the sdf path."""
    conf_dir = Path(conf_dir)
    xyz_path = conf_dir / geometry_file
    sdf_path = xyz_path.with_suffix(".sdf")
    if sdf_path.exists() and not overwrite:
        return sdf_path
    from openbabel import pybel  # heavy import, only needed when converting

    pybel.ob.obErrorLog.StopLogging()
    for mol in pybel.readfile("xyz", str(xyz_path)):
        mol.write("sdf", str(sdf_path), overwrite=True)  # multi-frame xyz: last frame wins (legacy)
    return sdf_path


def get_xyz_ids(conf_dir: str | Path, geometry_file: str = "optimization.xyz") -> Path:
    """Legacy side file listing the geometry file path (kept for compatibility with old scripts)."""
    conf_dir = Path(conf_dir)
    out = conf_dir / "xyz_ids.txt"
    with open(out, "w") as fh:
        fh.write(str(conf_dir / geometry_file) + os.linesep)
    return out


def load_mol(sdf_path: str | Path):
    """First valid molecule of an sdf file, hydrogens kept."""
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mol = next((m for m in suppl if m is not None), None)
    if mol is None:
        raise ValueError(f"RDKit could not read a molecule from {sdf_path}")
    return mol


def smiles_for_mol(mol, mode: str = "whole_molecule", strip_indices: Iterable[int] = ()) -> str:
    """SMILES for one conformer.

    ``whole_molecule``: SMILES of the molecule as read.
    ``strip_atoms``: remove ``strip_indices`` (0-based) and return the SMILES of the first
    fragment with more than one atom, i.e. the ligand once the metal (and any monoatomic
    ligands) are gone. This is the legacy ``get_smiles(type_mol="ni_complex")`` behaviour.
    """
    if mode == "whole_molecule":
        return Chem.MolToSmiles(mol)
    if mode != "strip_atoms":
        raise ValueError(f"unknown smiles mode {mode!r}")
    rw = Chem.RWMol(mol)
    for idx in sorted({int(i) for i in strip_indices}, reverse=True):
        rw.RemoveAtom(idx)
    for frag in GetMolFrags(rw, asMols=True, sanitizeFrags=False):
        if frag.GetNumAtoms() > 1:
            return Chem.MolToSmiles(frag)
    return Chem.MolToSmiles(rw)


def scf_energy(conf_dir: str | Path, basename: str = "singlepoint") -> float:
    """SCF energy (Hartree) from ``<conf_dir>/<basename>.property.json`` via ``OrcaReader``."""
    conf_dir = Path(conf_dir)
    reader = od.OrcaReader(directory=str(conf_dir), basename=basename, out_file_path=str(conf_dir / "calcParam.out"))
    reader.parse_property_json()
    reader.get_scfenergy()
    return reader.results["SCF_energy"]
