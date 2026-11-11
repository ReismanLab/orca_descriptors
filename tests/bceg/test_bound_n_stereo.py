"""start_bound_n_stereo: each replicate starts from its own configuration of the metal-bound amine N."""
from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from bceg import geometry
from bceg.search.base import START_DEFAULTS
from bceg.search.crest import Crest
from bceg.transforms import BUILTIN_TRANSFORMS, Transformation

# the donor SMARTS of configs/smI2_goat_constrain_ISmI180.yaml: amines, pyridines, ether/alcohol O beta to an N
DONORS = "[$([NX3;!a;!$(N[C,S]=O);!$(Nc)]),$([nX2]),$([OX2;!$(O[a]);$(O[CX4][CX4][NX3,nX2])])]"
SMI2 = Transformation("SmI2", BUILTIN_TRANSFORMS / "metal_x.py", {"metal": "Sm", "x": "I", "n_x": 2, "donors": DONORS},
                      multiplicity=7)
# N,N'-dibenzyl ethylenediamine with an OH arm and an OMe arm: two metal-bound N, both stereocentres
TETRADENTATE = "CO[C@@H](CN(CCN(Cc1ccccc1)C[C@H](O)c1ccccc1)Cc1ccccc1)c1ccccc1"
# the second N carries two benzyls: not a stereocentre
NBN2 = "O[C@@H](CN(CCN(Cc1ccccc1)Cc1ccccc1)Cc1ccccc1)c1ccccc1"


@pytest.fixture
def no_coordination_check(monkeypatch):
    """Raw ETKDG embeddings of SmI2 complexes clash (I...I, C...Sm) until the pre-optimisation, which
    these tests skip; they are about the N configurations, so the coordination check is switched off."""
    monkeypatch.setattr(geometry, "_coordination_issues", lambda *a, **k: [])


def params(**kw):
    return {**Crest.defaults, "start_preopt": "none", "start_bound_n_stereo": "enumerate",
            "max_metal_ligand_distance": 3.6, "start_allow_decoordination": True, **kw}


def test_stereocentres_skip_symmetric_amines():
    assert len(geometry.bound_amine_stereocentres(SMI2.apply(TETRADENTATE).mol)) == 2
    assert len(geometry.bound_amine_stereocentres(SMI2.apply(NBN2).mol)) == 1


def test_tag_from_coords_agrees_with_rdkit_on_carbon():
    for smi in ("F[C@](Cl)(Br)I", "F[C@@](Cl)(Br)I", "C[C@H](O)CC", "O[C@H]1CC[C@@H](C)CC1"):
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(m, randomSeed=3)
        xyz = m.GetConformer().GetPositions()
        for a in m.GetAtoms():
            tag = a.GetChiralTag()
            if tag in (Chem.CHI_TETRAHEDRAL_CW, Chem.CHI_TETRAHEDRAL_CCW):
                assert geometry.chiral_tag_from_coords(m, a.GetIdx(), xyz) == ("CW" if tag == Chem.CHI_TETRAHEDRAL_CW else "CCW")


def test_replicates_cover_every_configuration(tmp_path, no_coordination_check):
    s = SMI2.apply(TETRADENTATE)
    seen = []
    for replicate in range(1, 6):
        coords, info = geometry.make_start_geometry(s, replicate, tmp_path, params())
        rec = info["bound_n_stereo"]
        assert len(rec["centres"]) == 2
        got = [geometry.chiral_tag_from_coords(s.mol, i, coords) for i in rec["centres"]]
        assert got == rec["configuration"]
        seen.append(tuple(got))
    n = len(info["bound_n_stereo"]["feasible"])
    assert n >= 2 and len(set(seen[:n])) == n      # replicates 1..n: every feasible configuration once
    assert seen[4] == seen[(5 - 1) % n]            # then they cycle


def test_seed_offset_keeps_replicate_numbering(tmp_path, no_coordination_check):
    s = SMI2.apply(TETRADENTATE)
    _, a = geometry.make_start_geometry(s, 1, tmp_path, params())
    _, b = geometry.make_start_geometry(s, 101, tmp_path, params(seed_offset=100))
    assert a["bound_n_stereo"]["index"] == b["bound_n_stereo"]["index"] == 0


def test_off_by_default_and_out_of_the_hash(tmp_path, no_coordination_check):
    assert START_DEFAULTS["start_bound_n_stereo"] == "none"
    _, info = geometry.make_start_geometry(SMI2.apply(TETRADENTATE), 1, tmp_path, params(start_bound_n_stereo="none"))
    assert "bound_n_stereo" not in info
    assert "start_bound_n_stereo" not in Crest().compared_params()
    assert Crest(start_bound_n_stereo="enumerate").compared_params()["start_bound_n_stereo"] == "enumerate"


def test_config_key():
    from desc_utils.submit.config import _pre_refinement
    out = _pre_refinement({"method": "none", "bound_n_stereo": "enumerate"}, "pre_refinement", "crest")
    assert out["bound_n_stereo"] == "enumerate"
    assert "bound_n_stereo" not in _pre_refinement({"method": "none", "bound_n_stereo": "none"}, "pre_refinement", "crest")
    with pytest.raises(ValueError, match="bound_n_stereo"):
        _pre_refinement({"method": "none", "bound_n_stereo": "all"}, "pre_refinement", "crest")


def test_inverted_bound_n_after_preopt_is_rejected(tmp_path, monkeypatch, no_coordination_check):
    """A pre-optimisation that inverts a still-bound N rejects the seed."""
    s = SMI2.apply(TETRADENTATE)
    centres = geometry.bound_amine_stereocentres(s.mol)

    def invert(elements, coords, *args, **kwargs):  # mirror the whole geometry: every configuration flips
        out = np.array(coords, dtype=float)
        out[:, 0] *= -1
        return out, -1.0
    monkeypatch.setattr(geometry, "xtb_identity", lambda exe: ("xtb", "fake"))
    monkeypatch.setattr(geometry, "xtb_optimize", invert)
    with pytest.raises(RuntimeError, match="metal-bound N"):
        geometry.make_start_geometry(s, 1, tmp_path, params(start_preopt="gfn2", start_max_attempts=2))
    assert centres
