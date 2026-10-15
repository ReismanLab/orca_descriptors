"""Ligand L -> L-M(X)n: every donor atom of a (multidentate) ligand bound datively to one metal M that also
carries n anionic ligands X, e.g. a bipyridine, a terpyridine or a glyme -> L-SmI2.

Config usage::

    transformation:
      name: SmI2
      script: metal_x.py
      params: {metal: Sm, x: I, n_x: 2, donors: "[n;X2]"}
      multiplicity: 7              # Sm(II), f6: not set here

The donor atoms are the matches of ``donors`` (a SMARTS; with atom maps only the mapped atoms count, so
``"[O;X1:1]=P"`` picks the O of HMPA). Every match binds, which is how the denticity is chosen: an ether
SMARTS on 18-crown-6 gives six bonds to the metal. A ligand with fewer than ``min_donors`` donors (or more
than ``max_donors``, when set) is skipped. The metal gets the formal charge ``n_x`` and each X the charge
-1, so the complex is neutral; X is bound datively to the metal, as ``ni_x2.py`` does.

The 3D geometry around the metal (e.g. I-Sm-I at 180 degrees) is not set here: impose it with
``pre_refinement: constrained`` and ``conformer_search.constraints`` (docs/stages/2_pre_refinement.md).
"""

from __future__ import annotations

from rdkit import Chem

DEFAULT_DONORS = "[n;X2,$([N;X2]=[#6])]"  # pyridine-type and imine N, the donors of ni_x2.py's scaffolds
X_ATOMIC_NUMBER = {"H": 1, "F": 9, "Cl": 17, "Br": 35, "I": 53}


def donor_atoms(ligand: Chem.Mol, donors: str) -> list[int]:
    pattern = Chem.MolFromSmarts(donors)
    if pattern is None:
        raise ValueError(f"donors: not a valid SMARTS: {donors!r}")
    mapped = [a.GetIdx() for a in pattern.GetAtoms() if a.GetAtomMapNum()]
    positions = mapped or [0]  # without atom maps the first atom of the pattern is the donor
    return sorted({m[p] for m in ligand.GetSubstructMatches(pattern, maxMatches=1000) for p in positions})


def transform(smiles: str, metal: str = "Sm", x: str = "I", n_x: int = 2, donors: str = DEFAULT_DONORS,
              min_donors: int = 1, max_donors: int | None = None) -> dict | None:
    """Return the L-M(X)n complex, or None if the ligand has too few (or too many) donor atoms."""
    if x not in X_ATOMIC_NUMBER:
        raise ValueError(f"x must be one of {sorted(X_ATOMIC_NUMBER)}, got {x!r}")
    if Chem.GetPeriodicTable().GetAtomicNumber(metal) <= 0:
        raise ValueError(f"metal: unknown element {metal!r}")
    ligand = Chem.MolFromSmiles(smiles)
    if ligand is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    sites = donor_atoms(ligand, donors)
    if len(sites) < min_donors or (max_donors is not None and len(sites) > max_donors):
        return None

    complex_ = Chem.RWMol(ligand)
    m = complex_.AddAtom(Chem.Atom(metal))
    complex_.GetAtomWithIdx(m).SetFormalCharge(n_x)
    complex_.GetAtomWithIdx(m).SetNoImplicit(True)
    for d in sites:
        complex_.AddBond(d, m, Chem.BondType.DATIVE)  # donor -> metal
    for _ in range(n_x):
        a = complex_.AddAtom(Chem.Atom(X_ATOMIC_NUMBER[x]))
        complex_.GetAtomWithIdx(a).SetFormalCharge(-1)
        complex_.GetAtomWithIdx(a).SetNoImplicit(True)
        complex_.AddBond(a, m, Chem.BondType.DATIVE)
    product = complex_.GetMol()
    Chem.SanitizeMol(product)
    return {
        "smiles": Chem.MolToSmiles(product),
        "charge": 0,
        "metadata": {"metal": metal, "x": x, "n_x": n_x, "donors": donors, "n_donors": len(sites)},
    }
