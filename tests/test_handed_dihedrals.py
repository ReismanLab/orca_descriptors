"""``orca.xyz.handed_dihedrals``: a dihedral around a stereocentre, made comparable across enantiomers.

An achiral SMILES embedded at random gives one enantiomer for some molecules and the other for the
rest. The raw signed dihedral then flips meaning between rows; the handed one must not.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from desc_utils import data_extraction as de
from desc_utils.spec import SpecError, validate_spec

# A benzylic chloride skeleton: Ca tetrahedral with Cl, Calk, Cipso (and an implicit H); Cipso
# carries a ring neighbour Xo in a plane that puts the Cl-Ca-Cipso-Xo dihedral near 60 degrees.
SYM = ["Cl", "C", "C", "C", "N"]
LABELS = {"Cl": 1, "Ca": 2, "Calk": 3, "Cipso": 4, "Xo": 5}
GEOM = np.array([
    [0.0, 0.0, 1.80],     # Cl
    [0.0, 0.0, 0.0],      # Ca
    [1.45, 0.0, -0.50],   # Calk
    [-0.72, 1.26, -0.50], # Cipso
    [-0.30, 2.50, 0.00],  # Xo
])
ENTRY = {"atoms": ["Cl", "Ca", "Cipso", "Xo"], "chirality": ["Ca", "Cl", "Calk", "Cipso"]}
DIH = "Dihed_Cl_Ca_Cipso_Xo"


def _describe(tmp_path, geom, name):
    folder = tmp_path / name / "conf_1"
    folder.mkdir(parents=True)
    lines = [str(len(SYM)), "energy=-1.0 units=hartree"]
    lines += [f"{s} {x:.8f} {y:.8f} {z:.8f}" for s, (x, y, z) in zip(SYM, geom)]
    (folder / "geometry.xyz").write_text("\n".join(lines) + "\n", encoding="utf-8")
    df = pd.DataFrame([{"path": str(folder), **LABELS}])
    out = de.get_xyz_descriptors(df, atoms=["Ca"], bonds=[], angles=[], dihedrals=[ENTRY["atoms"]],
                                 handed_dihedrals=[ENTRY], overwrite_json=True, progress=False,
                                 geometry_file="geometry.xyz")
    return out[next(iter(out))]


def test_the_mirror_image_flips_the_raw_dihedral_but_not_the_handed_one(tmp_path):
    a = _describe(tmp_path, GEOM, "A")
    b = _describe(tmp_path, GEOM * np.array([1.0, 1.0, -1.0]), "B")
    assert abs(a[DIH]) > 10
    assert b[DIH] == pytest.approx(-a[DIH], abs=1e-6)
    assert a["Handedness_Ca_Cl_Calk_Cipso"] == -b["Handedness_Ca_Cl_Calk_Cipso"]
    assert a[DIH + "_handed"] == pytest.approx(b[DIH + "_handed"], abs=1e-6)
    assert a[DIH + "_handed"] == pytest.approx(a[DIH] * a["Handedness_Ca_Cl_Calk_Cipso"])


def test_a_planar_center_has_no_hand(tmp_path):
    flat = GEOM.copy()
    flat[:4, 2] = 0.0  # Cl, Ca, Calk, Cipso coplanar: no stereocentre to refer to
    res = _describe(tmp_path, flat, "F")
    assert math.isnan(res["Handedness_Ca_Cl_Calk_Cipso"])
    assert math.isnan(res[DIH + "_handed"])


def _spec(entries):
    return {
        "name": "t", "data": {"root": "DATA/x", "molecule_id_column": "inchikey"},
        "atoms": {k: None for k in LABELS},
        "numbering": {"backend": "first_n", "first_n": {"n": 5},
                      "energy": {"source": "xyz_comment", "key": "energy", "units": "hartree",
                                 "column": "uma_energy_hartree"}},
        "descriptors": {"orca": {"xyz": {"atoms": ["Ca"], "handed_dihedrals": entries}}},
    }


def test_the_spec_accepts_a_well_formed_entry_and_defaults_to_none():
    spec = validate_spec(_spec([ENTRY]))
    assert spec.orca["xyz"]["handed_dihedrals"] == [ENTRY]
    spec = validate_spec(_spec(None))
    assert spec.orca["xyz"]["handed_dihedrals"] == []


@pytest.mark.parametrize("bad, msg", [
    ({"atoms": ["Cl", "Ca", "Cipso"], "chirality": ENTRY["chirality"]}, "exactly 4"),
    ({"atoms": ENTRY["atoms"], "chirality": ["Ca", "Cl", "Calk", "Nope"]}, "unknown atom label"),
    ({**ENTRY, "sign": 1}, "unknown setting"),
    (["Cl", "Ca", "Cipso", "Xo"], "expected"),
])
def test_the_spec_refuses_malformed_entries(bad, msg):
    with pytest.raises(SpecError, match=msg):
        validate_spec(_spec([bad]))
