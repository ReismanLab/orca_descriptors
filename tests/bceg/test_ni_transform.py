from __future__ import annotations

from pathlib import Path

import pytest
from rdkit import Chem

from bceg.inputs import InputMolecule
from bceg.transforms import BUILTIN_TRANSFORMS, Transformation

SCRIPT = BUILTIN_TRANSFORMS / "ni_x2.py"

LIGANDS = {
    "bipy": "c1ccc(-c2ccccn2)nc1",
    "pyox": "CC(C)(C)[C@@H]1COC(c2ccccn2)=N1",
    "box": "CC1(C)COC(CC2=NC(C)(C)CO2)=N1",
    "biox": "C1COC(C2=NCCO2)=N1",
    "bilm": "C1CNC(C2=NCCN2)=N1",
    "phen": "c1cnc2c(c1)ccc1cccnc12",
    "pyim": "CC(C)(C)[C@@H]1CN(c2ccccc2)C(c2ccc(C(F)(F)F)cn2)=N1",
    "quinim": "CC(C)(C)[C@H]1CN(c2ccccc2)C(c2cccc3cccnc23)=N1",
}


@pytest.mark.parametrize("x", ["F", "H"])
@pytest.mark.parametrize("scaffold,smiles", LIGANDS.items())
def test_ni_x2_complex(scaffold, smiles, x):
    t = Transformation(f"Ni{x}2", SCRIPT, {"x": x})
    lig = InputMolecule.from_smiles(smiles)
    s = t.apply(lig.canonical_smiles)
    assert s.metadata == {"scaffold": scaffold, "x": x}
    assert s.charge == 0 and s.multiplicity == 1

    ligand_atoms = Chem.AddHs(Chem.MolFromSmiles(smiles)).GetNumAtoms()
    assert len(s.elements) == ligand_atoms + 3  # Ni + 2 X
    ni = next(a for a in s.mol.GetAtoms() if a.GetSymbol() == "Ni")
    assert ni.GetFormalCharge() == 2
    assert sorted(n.GetSymbol() for n in ni.GetNeighbors()) == sorted(["N", "N", x, x])
    # stereocentres of the ligand survive
    assert len(Chem.FindMolChiralCenters(s.mol)) == len(Chem.FindMolChiralCenters(Chem.MolFromSmiles(smiles)))


def test_declared_identity_replaces_the_file_hash():
    import hashlib

    t = Transformation("NiH2", SCRIPT, {"x": "H"})
    assert t.declared_identity() == t.sha256 == "09430bac6eb225569e2a784a0930dda360ce4f591c8b1b37ef4de11e3a61f3c3"
    assert t.sha256 != hashlib.sha256(SCRIPT.read_bytes()).hexdigest()  # the file grew since, the identity did not


def test_script_without_declared_identity_is_hashed(tmp_path):
    import hashlib

    script = tmp_path / "t.py"
    script.write_text("def transform(smiles):\n    return smiles\n")
    assert Transformation("t", script).sha256 == hashlib.sha256(script.read_bytes()).hexdigest()


def test_phenanthroline_is_phen_not_bipy_and_bipy_still_bipy():
    t = Transformation("NiH2", SCRIPT, {"x": "H"})
    assert t.apply("c1cnc2c(c1)ccc1cccnc12").metadata["scaffold"] == "phen"
    assert t.apply("c1ccc(-c2ccccn2)nc1").metadata["scaffold"] == "bipy"


def test_unknown_ligand_is_skipped():
    assert Transformation("NiF2", SCRIPT, {"x": "F"}).apply("CN(C)CCN(C)C") is None


def test_params_are_part_of_identity():
    a = Transformation("NiF2", SCRIPT, {"x": "F"}).identity()
    b = Transformation("NiF2", SCRIPT, {"x": "F"}, multiplicity=3).identity()
    assert a["script_sha256"] == b["script_sha256"] and a != b


def test_multiplicity_override():
    s = Transformation("NiF2", SCRIPT, {"x": "F"}, multiplicity=3).apply("c1ccc(-c2ccccn2)nc1")
    assert s.multiplicity == 3 and s.uhf == 2
    with pytest.raises(ValueError, match="impossible"):
        Transformation("NiF2", SCRIPT, {"x": "F"}, multiplicity=2).apply("c1ccc(-c2ccccn2)nc1")


def test_auto_start_geometry_keeps_hydrides_apart():
    import numpy as np

    from bceg.chem import embed

    s = Transformation("NiH2", SCRIPT, {"x": "H"}).apply("c1ccc(-c2ccccn2)nc1")
    hydrides = [n.GetIdx() for a in s.mol.GetAtoms() if a.GetSymbol() == "Ni" for n in a.GetNeighbors() if n.GetSymbol() == "H"]
    for seed in (1, 2, 3):
        c = embed(s.mol, seed, ff="auto")
        assert np.linalg.norm(c[hydrides[0]] - c[hydrides[1]]) > 1.5


def test_stored_smiles_has_no_explicit_carbon_hydrogens():
    s = Transformation("NiH2", SCRIPT, {"x": "H"}).apply("c1ccc(-c2ccccn2)nc1")
    assert s.smiles.count("[H") == 2 and "[H]" not in s.smiles  # only the two [H-] hydrides
    ni = next(a for a in s.mol.GetAtoms() if a.GetSymbol() == "Ni")
    assert sorted(n.GetSymbol() for n in ni.GetNeighbors()) == ["H", "H", "N", "N"]
