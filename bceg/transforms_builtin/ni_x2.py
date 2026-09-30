"""N,N-ligand L -> L-Ni(X)2 complex (X = F or H), adapted from scripts_4_structure/make_Ni{F2,H2}.py.

Config usage::

    transformations:
      NiF2: {script: ../transforms/ni_x2.py, params: {x: F}, multiplicity: 1}
      NiH2: {script: ../transforms/ni_x2.py, params: {x: H}, multiplicity: 1}

The two N donors of the ligand are bound datively to Ni(+2); X is added as X(-1)
(Be is used as a placeholder during the reaction). Net charge of the complex is 0.
The spin multiplicity is not set here: it defaults to 1 and is set with
``multiplicity:`` in the config (or ``bceg submit --multiplicity``).
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import rdChemReactions

# Ligand scaffolds recognised (checked in this order)
TEMPLATES = {
    "pyox": "C1(c2ncccc2)=NCCO1",
    "box": "C1(CC2=NCCO2)=NCCO1",
    "biox": "C1(C2=NCCO2)=NCCO1",
    "bilm": "C1(C2=NCCN2)=NCCN1",
    "bipy": "c1(c2ncccc2)ncccc1",
}

# Reaction binding both N donors to Ni. (Atom map :0 of the original scripts means
# "unmapped" in RDKit and was renumbered to :20.)
REACTIONS = {
    "pyox": "[C:20]1([c:1]2[n:2][c:3][c:4][c:5][c:6]2)=[N:7][C:8][C:9][O:10]1.[Ni:11]>>"
            "[C:20]1([c:1]2[n:2](->[Ni:11]3)[c:3][c:4][c:5][c:6]2)=[N:7]->3[C:8][C:9][O:10]1",
    "box": "[C:20]1([C:1][C:2]2=[N:3][C:4][C:5][O:6]2)=[N:7][C:8][C:9][O:10]1.[Ni:11]>>"
           "[C:20]1([C:1][C:2]2=[N:3](->[Ni:11]3)[C:4][C:5][O:6]2)=[N:7]->3[C:8][C:9][O:10]1",
    "biox": "[C:20]1([C:2]2=[N:3][C:4][C:5][O:6]2)=[N:7][C:8][C:9][O:10]1.[Ni:11]>>"
            "[C:20]1([C:2]2=[N:3](->[Ni:11]3)[C:4][C:5][O:6]2)=[N:7]->3[C:8][C:9][O:10]1",
    "bilm": "[C:20]1([C:2]2=[N:3][C:4][C:5][N:6]2)=[N:7][C:8][C:9][N:10]1.[Ni:11]>>"
            "[C:20]1([C:2]2=[N:3](->[Ni:11]3)[C:4][C:5][N:6]2)=[N:7]->3[C:8][C:9][N:10]1",
    "bipy": "[c:1]1([c:2]2[n:3][c:4][c:5][c:6][c:13]2)[n:7][c:8][c:9][c:10][c:11]1.[Ni:12]>>"
            "[c:1]1([c:2]2[n:3](->[Ni:12]3)[c:4][c:5][c:6][c:13]2)[n:7]->3[c:8][c:9][c:10][c:11]1",
}

X_ATOMIC_NUMBER = {"F": 9, "H": 1}


def identify_scaffold(mol: Chem.Mol) -> str | None:
    for name, smarts in TEMPLATES.items():
        if mol.HasSubstructMatch(Chem.MolFromSmiles(smarts)):
            return name
    return None


def transform(smiles: str, x: str) -> dict | None:
    """Return the L-Ni(X)2 complex, or None if the ligand matches no known scaffold."""
    if x not in X_ATOMIC_NUMBER:
        raise ValueError(f"x must be one of {sorted(X_ATOMIC_NUMBER)}, got {x!r}")
    ligand = Chem.MolFromSmiles(smiles)
    if ligand is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    scaffold = identify_scaffold(ligand)
    if scaffold is None:
        return None

    ni = Chem.MolFromSmiles("[Ni](<-[Be])<-[Be]")  # Be = placeholder for X
    products = rdChemReactions.ReactionFromSmarts(REACTIONS[scaffold]).RunReactants((ligand, ni))
    if not products:
        raise ValueError(f"{smiles}: no product from the {scaffold} reaction template")

    complex_ = Chem.AddHs(Chem.MolFromSmiles(Chem.MolToSmiles(products[0][0])))
    for atom in complex_.GetAtoms():
        if atom.GetSymbol() == "Be":
            atom.SetAtomicNum(X_ATOMIC_NUMBER[x])
            atom.SetFormalCharge(-1)
        elif atom.GetSymbol() == "Ni":
            atom.SetFormalCharge(2)
    return {
        "smiles": Chem.MolToSmiles(Chem.RemoveHs(complex_)),  # hydrides on Ni are kept
        "charge": 0,
        "metadata": {"scaffold": scaffold, "x": x},
    }
