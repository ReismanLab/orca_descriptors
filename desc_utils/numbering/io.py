"""File-level helpers for the numbering stage (moved from the former ``atom_numbering.py``).

* xyz -> sdf conversion with Open Babel (bond perception for RDKit)
* loading the sdf as an RDKit molecule with explicit hydrogens
* SMILES of the whole molecule or of the ligand after stripping labelled atoms
* SCF energy from the ORCA property JSON via ``OrcaReader``
"""
from __future__ import annotations

import json
import os
import re
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


def load_mol(sdf_path: str | Path, notes: list[str] | None = None):
    """First valid molecule of an sdf file, hydrogens kept.

    Open Babel sometimes perceives an impossible bond order from a DFT geometry (a five-valent
    carbon, a four-valent neutral nitrogen), and RDKit then refuses to sanitize the whole molecule.
    Rather than losing the conformer, the molecule is loaded without sanitization and only the
    connectivity-level properties are computed (degrees, rings), which is what the connectivity
    SMARTS patterns of a spec need; bond orders and aromaticity are unreliable in that case.
    A note is appended to ``notes`` (when given) so the numbering table records it.
    """
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mol = next((m for m in suppl if m is not None), None)
    if mol is not None:
        return mol
    mol = Chem.MolFromMolFile(str(sdf_path), removeHs=False, sanitize=False)
    if mol is None:
        raise ValueError(f"RDKit could not read a molecule from {sdf_path}")
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    if notes is not None:
        notes.append("sdf: RDKit sanitization failed (Open Babel bond perception); connectivity only")
    return mol


def _to_smiles(mol, remove_hs: bool) -> str:
    if not remove_hs:
        return Chem.MolToSmiles(mol)
    try:
        return Chem.MolToSmiles(Chem.RemoveHs(mol))
    except Exception:  # a molecule loaded by connectivity only may not sanitise
        return Chem.MolToSmiles(Chem.RemoveHs(mol, sanitize=False))


def smiles_for_mol(mol, mode: str = "whole_molecule", strip_indices: Iterable[int] = (), remove_hs: bool = False) -> str:
    """SMILES for one conformer.

    ``whole_molecule``: SMILES of the molecule as read.
    ``strip_atoms``: remove ``strip_indices`` (0-based) and return the SMILES of the first
    fragment with more than one atom, i.e. the ligand once the metal (and any monoatomic
    ligands) are gone. This is the legacy ``get_smiles(type_mol="ni_complex")`` behaviour.
    ``remove_hs``: drop the explicit hydrogens first (the xyz-derived molecule carries them all),
    giving the usual implicit-H SMILES; the ``smiles_noH`` column of the tables.
    """
    if mode == "whole_molecule":
        return _to_smiles(mol, remove_hs)
    if mode != "strip_atoms":
        raise ValueError(f"unknown smiles mode {mode!r}")
    rw = Chem.RWMol(mol)
    for idx in sorted({int(i) for i in strip_indices}, reverse=True):
        rw.RemoveAtom(idx)
    for frag in GetMolFrags(rw, asMols=True, sanitizeFrags=False):
        if frag.GetNumAtoms() > 1:
            return _to_smiles(frag, remove_hs)
    return _to_smiles(rw, remove_hs)


REGISTRY_FILE = "molecule_registry.json"


def submitted_smiles(source: str, data_root: str | Path, smiles_file: str | Path | None = None) -> dict[str, str]:
    """Molecule id (InChIKey) -> the SMILES the structures were generated from.

    ``registry``: ``<data_root>/molecule_registry.json``, written by ``orcadesc submit`` (and by the adoption of
    older libraries): the submitted SMILES of every molecule. ``file``: a SMILES list (``.csv`` with a ``smiles``
    column, or text, one SMILES per line with an optional name after it), each SMILES keyed by the InChIKey of
    its canonical form, the rule ``orcadesc`` uses to name the molecule folders.
    """
    if source == "registry":
        path = Path(data_root) / REGISTRY_FILE
        if not path.is_file():
            raise FileNotFoundError(f"numbering.smiles.source registry: no {path}")
        return {ik: entry["smiles"] for ik, entry in json.loads(path.read_text()).items() if entry.get("smiles")}
    if source == "file":
        from bceg.inputs import load_inputs
        return {m.inchikey: m.input_smiles for m in load_inputs(smiles_file)}
    raise ValueError(f"unknown smiles source {source!r}")


def canonical_noH(smiles: str) -> str | None:
    """RDKit canonical SMILES (implicit hydrogens) of a submitted SMILES; None when it does not parse."""
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else Chem.MolToSmiles(mol)


#: 1 Hartree in each unit accepted by ``numbering.energy.units`` (CODATA 2018).
HARTREE_IN = {"hartree": 1.0, "eV": 27.211386245988, "kcal/mol": 627.5094740631, "kJ/mol": 2625.4996394799}


def to_hartree(value: float, units: str) -> float:
    """Convert an energy from ``units`` (``hartree``, ``eV``, ``kcal/mol``, ``kJ/mol``) to Hartree."""
    try:
        return float(value) / HARTREE_IN[units]
    except KeyError:
        raise ValueError(f"unknown energy units {units!r}; expected one of {list(HARTREE_IN)}") from None


def xyz_comment_energy(conf_dir: str | Path, geometry_file: str = "optimization.xyz", key: str = "energy",
                       units: str = "eV") -> float:
    """``<key>=<value>`` of the comment line of the geometry file, converted from ``units`` to Hartree.

    This is the extended-xyz header ASE writes (``energy=-2104.109 free_energy=... pbc="F F F"``), so a
    geometry optimised with UMA or any other ASE calculator carries its own energy. Only the first frame is
    read. Raises ``ValueError`` when the key is absent or not a number.
    """
    path = Path(conf_dir) / geometry_file
    with open(path, encoding="utf-8") as fh:
        fh.readline()
        comment = fh.readline()
    # `energy=` must not match inside `free_energy=`: no word character may precede the key
    m = re.search(rf"(?<![A-Za-z0-9_]){re.escape(key)}=\"?([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\"?(?=\s|$)", comment)
    if m is None:
        raise ValueError(f"{path.name}: no '{key}=<number>' in the comment line: {comment.strip()[:120]!r}")
    return to_hartree(float(m.group(1)), units)


def scf_energy(conf_dir: str | Path, basename: str = "singlepoint",
               condensed_file: str | None = od.CONDENSED_FILENAME) -> float:
    """SCF energy (Hartree) of ``<basename>`` via ``OrcaReader``.

    Read from ``<conf_dir>/<condensed_file>`` (``desc_utils.condense``) when present, otherwise
    from ``<conf_dir>/<basename>.property.json``.
    """
    conf_dir = Path(conf_dir)
    reader = od.OrcaReader(directory=str(conf_dir), basename=basename, out_file_path=str(conf_dir / "calcParam.out"),
                           condensed_file=condensed_file)
    reader.parse_property_json()
    reader.get_scfenergy()
    return reader.results["SCF_energy"]
