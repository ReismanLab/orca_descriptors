"""RDKit helpers: parsing (incl. dative metal bonds), identifiers, starting geometries."""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# Elements treated as "organic" for applicability checks (LoQI, MMFF, ...)
ORGANIC_ELEMENTS = {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I"}


def mol_from_smiles(smiles: str, add_hs: bool = True) -> Chem.Mol:
    """Parse SMILES, tolerating valences RDKit rejects for metal complexes."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            raise ValueError(f"Could not parse SMILES: {smiles}")
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    return Chem.AddHs(mol) if add_hs else mol


def canonical_smiles(smiles: str) -> str:
    return Chem.MolToSmiles(mol_from_smiles(smiles, add_hs=False))


def inchikey(smiles: str) -> str:
    key = Chem.MolToInchiKey(mol_from_smiles(smiles, add_hs=False))
    if not key:
        raise ValueError(f"Could not compute InChIKey for {smiles}")
    return key


def elements_of(mol: Chem.Mol) -> list[str]:
    return [a.GetSymbol() for a in mol.GetAtoms()]


def is_organic(elements: list[str]) -> bool:
    return set(elements) <= ORGANIC_ELEMENTS


def embed(mol: Chem.Mol, seed: int, ff: str = "auto", max_iters: int = 2000) -> np.ndarray:
    """One ETKDGv3 conformer, optionally force-field relaxed. Returns (n, 3) coords in Angstrom.

    ``ff``: "auto" (MMFF94s if parametrised, else none), "MMFF94", "MMFF94s", "UFF", or "none".
    UFF is not used automatically: with dative Ni bonds it can collapse ligands (e.g. H-H in NiH2).
    """
    mol = Chem.Mol(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.enforceChirality = True
    if not is_organic(elements_of(mol)):
        params.useRandomCoords = True
    cid = AllChem.EmbedMolecule(mol, params)
    if cid < 0:
        params.useRandomCoords = True
        cid = AllChem.EmbedMolecule(mol, params)
    if cid < 0:
        raise RuntimeError("ETKDG embedding failed")
    ff = ff_for(mol) if ff == "auto" else ff
    if ff.startswith("MMFF"):
        AllChem.MMFFOptimizeMolecule(mol, mmffVariant=ff, maxIters=max_iters)
    elif ff == "UFF":
        AllChem.UFFOptimizeMolecule(mol, maxIters=max_iters)
    return np.array(mol.GetConformer().GetPositions())


def ff_for(mol: Chem.Mol) -> str:
    """Force field chosen by ``auto``: MMFF94s if parametrised, else none (UFF breaks dative Ni complexes)."""
    return "MMFF94s" if AllChem.MMFFHasAllMoleculeParams(mol) else "none"


def reorder_to_template(mol: Chem.Mol, template: Chem.Mol) -> list[int]:
    """Return ``order`` with ``template atom i`` == ``mol atom order[i]``."""
    match = mol.GetSubstructMatch(template, useChirality=False)
    if len(match) != template.GetNumAtoms() or mol.GetNumAtoms() != template.GetNumAtoms():
        raise ValueError("Could not map generated structure onto the template atom order")
    return list(match)
