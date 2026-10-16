"""RDKit helpers: parsing (incl. dative metal bonds), identifiers, starting geometries."""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

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

    RDKit's chirality enforcement rejects every embedding of some chelates with two stereocentres
    next to the donor atoms (C2-symmetric bis(oxazoline)/bis(imidazoline) on Ni), although the
    unenforced embedding has the right configuration. So when the enforced embeddings fail, the
    embedding is repeated without enforcement and its stereocentres are checked against the
    template from the 3D coordinates; a mismatch is an embedding failure (the caller moves to the
    next seed).
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
        params.enforceChirality = False
        cid = AllChem.EmbedMolecule(mol, params)
        if cid >= 0 and not stereo_matches(mol):
            raise RuntimeError("ETKDG embedding failed (unenforced embedding has the wrong configuration)")
    if cid < 0:
        raise RuntimeError("ETKDG embedding failed")
    ff = ff_for(mol) if ff == "auto" else ff
    if ff.startswith("MMFF"):
        AllChem.MMFFOptimizeMolecule(mol, mmffVariant=ff, maxIters=max_iters)
    elif ff == "UFF":
        AllChem.UFFOptimizeMolecule(mol, maxIters=max_iters)
    return np.array(mol.GetConformer().GetPositions())


def stereo_centers(mol: Chem.Mol) -> list[tuple[int, str]]:
    """(atom index, CIP label or '?') of every tetrahedral stereocentre, from the graph's tags."""
    return Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)


def stereo_matches(mol: Chem.Mol) -> bool:
    """True if the stereocentres read from ``mol``'s conformer agree with its tags (SMILES)."""
    from_3d = Chem.Mol(mol)
    Chem.AssignStereochemistryFrom3D(from_3d, confId=mol.GetConformer().GetId(), replaceExistingTags=True)
    return stereo_centers(from_3d) == stereo_centers(mol)


def ff_for(mol: Chem.Mol) -> str:
    """Force field chosen by ``auto``: MMFF94s if parametrised, else none (UFF breaks dative Ni complexes)."""
    return "MMFF94s" if AllChem.MMFFHasAllMoleculeParams(mol) else "none"


def reorder_to_template(mol: Chem.Mol, template: Chem.Mol) -> list[int]:
    """Return ``order`` with ``template atom i`` == ``mol atom order[i]``."""
    match = mol.GetSubstructMatch(template, useChirality=False)
    if len(match) != template.GetNumAtoms() or mol.GetNumAtoms() != template.GetNumAtoms():
        raise ValueError("Could not map generated structure onto the template atom order")
    return list(match)


def rotatable_bonds(mol: Chem.Mol) -> int:
    """RDKit's (strict) rotatable-bond count on the heavy-atom graph.

    The hydrogens are removed first: with explicit H every methyl would count as a rotor. Bonds in
    rings, including chelate rings through a metal, and dative bonds are not rotatable.
    """
    heavy = Chem.RemoveHs(mol, sanitize=False)
    Chem.FastFindRings(heavy)
    return rdMolDescriptors.CalcNumRotatableBonds(heavy)
