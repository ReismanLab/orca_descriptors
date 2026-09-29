from __future__ import annotations

import numpy as np
import pytest

from bceg.inputs import InputMolecule, load_inputs
from bceg.layout import method_dir
from bceg.provenance import ParamsMismatch, check_or_write, make_params
from bceg.search import SEARCHERS
from bceg.search.crest import Crest, to_toml
from bceg.structure import Structure
from bceg.xyz import Conformer, read_xyz, write_xyz


def test_inchikey_from_canonical_input():
    a = InputMolecule.from_smiles("OCC")
    b = InputMolecule.from_smiles("C(O)C")
    assert a.canonical_smiles == b.canonical_smiles == "CCO"
    assert a.inchikey == b.inchikey == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"


def test_duplicate_inputs_dropped(tmp_path):
    f = tmp_path / "in.csv"
    f.write_text("smiles\nCCO\nOCC\nCCCC\n")
    with pytest.warns(UserWarning):
        mols = load_inputs(f)
    assert len(mols) == 2


def test_layout_names(tmp_path):
    md = method_dir(tmp_path, "NiF2", "KEY", "crest_gfn2+uma-opt")
    assert md.path == tmp_path / "NiF2" / "KEY" / "crest_gfn2+uma-opt"
    assert md.raw(2).name == "raw_2.xyz"
    assert md.refined(2, "uma-s-1p2").name == "raw_2_uma-s-1p2.xyz"
    assert md.params("uma-s-1p2").name == "params_uma-s-1p2.json"
    with pytest.raises(ValueError):
        method_dir(tmp_path, "Ni F2", "KEY", "x")


def test_xyz_roundtrip(tmp_path):
    confs = [Conformer(np.random.rand(3, 3), -1.5, {"converged": True}), Conformer(np.random.rand(3, 3))]
    write_xyz(tmp_path / "a.xyz", ["O", "H", "H"], confs)
    els, back = read_xyz(tmp_path / "a.xyz")
    assert els == ["O", "H", "H"]
    assert back[0].energy == pytest.approx(-1.5) and back[1].energy is None
    assert np.allclose(back[0].coords, confs[0].coords, atol=1e-7)


def test_params_coercion_and_unknown_keys():
    assert Crest(threads=4, toml_extra={"cregen": {"ewin": 6}}).params["threads"] == 4
    assert SEARCHERS["rdkit_etkdg"](prune_rms=1).params["prune_rms"] == 1.0
    with pytest.raises(ValueError, match="unknown parameter"):
        Crest(ewin=6)
    with pytest.raises(TypeError):
        Crest(start_max_attempts="8")
    with pytest.raises(TypeError):
        SEARCHERS["rdkit_etkdg"](n_confs=True)


def test_resource_keys_not_compared():
    a, b = Crest(threads=4), Crest(threads=16, mlip_device="cpu")
    assert a.compared_params() == b.compared_params()
    assert Crest(refine="opt").resource_class() == "gpu"
    assert Crest().resource_class() == "cpu"


CREST_GFN2 = '''input = "input.xyz"
runtype = "imtd-gc"
threads = 8

[calculation]

[[calculation.level]]
method = "gfn2"
charge = 0
uhf = 0
'''

CREST_UMA_OPT = CREST_GFN2 + '''
[[calculation.level]]
method = "mlip"
mlip_backend = "uma"
mlip_uma_task = "omol"
mlip_uma_model = "uma-s-1p2"
charge = 0
uhf = 0
refine = "opt"
mlip_device = "cuda"
'''


@pytest.mark.parametrize("params,expected", [
    ({}, CREST_GFN2),
    ({"refine": "opt"}, CREST_UMA_OPT),
    ({"refine": "sp"}, CREST_UMA_OPT.replace('refine = "opt"', 'refine = "sp"')),
])
def test_crest_toml(params, expected):
    assert to_toml(Crest(threads=8, **params).build_input(Structure("CCO"))) == expected


def test_crest_toml_charge_mult_and_extra():
    toml = to_toml(Crest(toml_extra={"cregen": {"ewin": 6.0}}).build_input(Structure("[NH4+]", multiplicity=1)))
    assert "charge = 1" in toml and "[cregen]\newin = 6.0" in toml


def test_params_log_rules(tmp_path):
    p = tmp_path / "params.json"
    a = make_params({"x": 1, "nested": {"y": [1, 2]}}, {"threads": 4})
    assert check_or_write(p, a) == "new"
    assert check_or_write(p, make_params({"nested": {"y": [1, 2]}, "x": 1}, {"threads": 64})) == "match"
    with pytest.raises(ParamsMismatch, match=r"~ nested.y: \[1, 2\] -> \[1, 3\]"):
        check_or_write(p, make_params({"x": 1, "nested": {"y": [1, 3]}}))


def test_spin_parity_checked():
    assert Structure("CCCC", multiplicity=3).uhf == 2
    with pytest.raises(ValueError, match="impossible"):
        Structure("CCCC", multiplicity=2)
    assert "uhf = 2" in to_toml(Crest().build_input(Structure("CCCC", multiplicity=3)))


def test_select_multiplicity_rename():
    from bceg.config import select

    raw = {"transformations": {"none": {"script": None}, "NiF2": {"script": "x.py", "multiplicity": 3}}}
    out = select(raw, transformations=["none", "NiF2"], multiplicity=3)
    assert list(out["transformations"]) == ["none_m3", "NiF2"]
    assert out["transformations"]["none_m3"]["multiplicity"] == 3


def test_text_inputs_with_names(tmp_path):
    f = tmp_path / "in.txt"
    f.write_text("# header\nCCO  ethanol\nCCCC\n")
    mols = load_inputs(f)
    assert [(m.canonical_smiles, m.name) for m in mols] == [("CCO", "ethanol"), ("CCCC", None)]
