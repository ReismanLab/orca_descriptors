"""metal_x.py: a (multidentate) ligand -> L-M(X)n, every donor atom bound to the metal."""

from __future__ import annotations

import pytest
from rdkit import Chem

from bceg.inputs import InputMolecule
from bceg.transforms import BUILTIN_TRANSFORMS, Transformation

SCRIPT = BUILTIN_TRANSFORMS / "metal_x.py"


@pytest.mark.parametrize("smiles, params, donors", [
    ("c1ccc(-c2ccccn2)nc1", {}, ["N", "N"]),                                      # bipy
    ("c1ccc(-c2cccc(-c3ccccn3)n2)nc1", {}, ["N", "N", "N"]),                     # terpy
    ("COCCOC", {"donors": "[O;X2]"}, ["O", "O"]),                                 # DME
    ("CN(C)P(=O)(N(C)C)N(C)C", {"donors": "[O;X1:1]=P"}, ["O"]),                  # HMPA: the mapped atom only
    ("C1COCCOCCOCCOCCO1", {"donors": "[O;X2]"}, ["O"] * 5),                       # 15-crown-5
])
def test_every_donor_binds_the_metal(smiles, params, donors):
    s = Transformation("SmI2", SCRIPT, {"metal": "Sm", "x": "I", "n_x": 2, **params}, multiplicity=7).apply(
        InputMolecule.from_smiles(smiles).canonical_smiles)
    assert s.charge == 0 and s.multiplicity == 7 and s.metadata["n_donors"] == len(donors)
    sm = next(a for a in s.mol.GetAtoms() if a.GetSymbol() == "Sm")
    assert sm.GetFormalCharge() == 2
    assert sorted(n.GetSymbol() for n in sm.GetNeighbors()) == sorted(donors + ["I", "I"])
    assert len(s.elements) == Chem.AddHs(Chem.MolFromSmiles(smiles)).GetNumAtoms() + 3
    assert InputMolecule.from_smiles(smiles).inchikey                               # the ligand names the molecule


def test_too_few_or_too_many_donors_skip_the_ligand():
    t = Transformation("SmI2", SCRIPT, {"min_donors": 2})
    assert t.apply("c1ccncc1") is None and t.apply("c1ccccc1") is None
    assert Transformation("SmI2", SCRIPT, {"max_donors": 2}).apply("c1ccc(-c2cccc(-c3ccccn3)n2)nc1") is None
    s = Transformation("NiBr2", SCRIPT, {"metal": "Ni", "x": "Br"}).apply("c1ccc(-c2ccccn2)nc1")
    assert s.charge == 0 and sorted(a.GetSymbol() for a in s.mol.GetAtoms() if a.GetSymbol() in ("Ni", "Br")) == ["Br", "Br", "Ni"]
    with pytest.raises(ValueError, match="not a valid SMARTS"):
        Transformation("SmI2", SCRIPT, {"donors": "[O"}).apply("COCCOC")
