"""RDKit helpers: parsing (incl. dative metal bonds), identifiers, starting geometries."""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom, rdMolDescriptors
from rdkit.DistanceGeometry import DoTriangleSmoothing

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


#: floors of RDKit's distance bounds around a metal, as fractions of the covalent-radii sum: a metal-ligand
#: bond at least METAL_BOND_FLOOR x, a 1-3 pair through or onto the metal at least METAL_13_FLOOR x. Both stay
#: below real distances (Sm-I 3.1 A vs 2.86 A floor; Sm...C through Sm-N-C 3.3-3.5 A vs 3.29 A floor)
METAL_BOND_FLOOR = 0.85
METAL_13_FLOOR = 1.2


def metal_bounds_matrix(mol: Chem.Mol) -> np.ndarray | None:
    """ETKDG's bounds matrix with floors around every metal atom, or None if no floor raises a bound.

    RDKit has no parameters for most metals and sets loose generic bounds for their bonds: on SmI2·L the
    I...I lower bound (1-3 through Sm) is 1.0 A and Sm-I's 1.1 A, so most embeddings put the two iodides
    on top of each other (I...I 0.0 or 0.8 A), which no pre-optimisation undoes. Raised here: each
    metal-ligand bond's lower bound to METAL_BOND_FLOOR x the covalent-radii sum, and that of every 1-3
    pair through or onto the metal (two ligands of the metal not bonded to each other; the metal and a
    donor's substituent: Sm...C, Sm...H of a bound OH) to METAL_13_FLOOR x. Upper bounds are lifted to
    stay above the new lower bounds, then the matrix is triangle-smoothed again; if smoothing fails, the
    embedding uses RDKit's own bounds (None).

    Organic molecules are untouched; on a metal complex the embedding for a given seed differs from
    before this floor existed (still deterministic per seed).
    """
    metals = [a for a in mol.GetAtoms() if a.GetSymbol() not in ORGANIC_ELEMENTS]
    if not metals:
        return None
    pt = Chem.GetPeriodicTable()
    rcov = [pt.GetRcovalent(a.GetAtomicNum()) for a in mol.GetAtoms()]
    bm = rdDistGeom.GetMoleculeBoundsMatrix(mol, useMacrocycle14config=True)  # as ETKDGv3 builds it
    changed = False

    def floor(i: int, j: int, lower: float) -> None:
        nonlocal changed
        i, j = min(i, j), max(i, j)  # bm[i][j] (i < j) upper, bm[j][i] lower
        if bm[j][i] < lower:
            bm[j][i] = lower
            bm[i][j] = max(bm[i][j], lower + 0.1)
            changed = True

    for m in metals:
        ligands = [n.GetIdx() for n in m.GetNeighbors()]
        for i in ligands:
            floor(m.GetIdx(), i, METAL_BOND_FLOOR * (rcov[m.GetIdx()] + rcov[i]))
        for a, i in enumerate(ligands):
            for j in ligands[a + 1:]:
                if mol.GetBondBetweenAtoms(i, j) is None:
                    floor(i, j, METAL_13_FLOOR * (rcov[i] + rcov[j]))
            for k in mol.GetAtomWithIdx(i).GetNeighbors():  # M-D-k: the donor's own substituents
                k = k.GetIdx()
                if k != m.GetIdx() and mol.GetBondBetweenAtoms(m.GetIdx(), k) is None:
                    floor(m.GetIdx(), k, METAL_13_FLOOR * (rcov[m.GetIdx()] + rcov[k]))
    if not changed or not DoTriangleSmoothing(bm):
        return None
    return bm


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

    Around a metal the distance bounds get floors (:func:`metal_bounds_matrix`), so ligands of one metal
    (I-Sm-I of SmI2·L) do not embed on top of each other.
    """
    mol = Chem.Mol(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.enforceChirality = True
    if not is_organic(elements_of(mol)):
        params.useRandomCoords = True
        bounds = metal_bounds_matrix(mol)
        if bounds is not None:
            params.SetBoundsMat(bounds)
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
