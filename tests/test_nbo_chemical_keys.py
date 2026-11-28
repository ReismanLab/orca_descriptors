"""NBO descriptors keyed by type, sigma/pi class and atoms (``desc_utils.nbo_keys``).

The rank naming put the BnCl C1-Cl sigma* anywhere from ``hm24`` to ``lp4`` and kept it in 47 of
279 conformers; these keys give it one name everywhere. The rules tested here: one-centre hybrids
are parsed, atoms are written in spec order, a bond polarised onto one labelled atom is that atom's
lone pair, pi NBOs are also keyed per atom with any partner, interchangeable NBOs are grouped, and
a metal's d lone pairs are named by their d orbital in the spec frame, with x always towards the
frame's second atom and y towards the third.
"""
import copy
import math

import numpy as np
import pytest

from desc_utils import nbo_keys as nk
from desc_utils.orca_parse import NBOProps
from desc_utils.spec import SpecError, validate_spec

DETAILS = """
     (Occupancy)   Bond orbital / Coefficients / Hybrids
 -------------------------------------------------------------------------------
 ---------------- Lewis --------------------------------------------------------
  29. (1.99083) LP ( 1)Ni  2            s(  1.76%)p 0.00(  0.00%)d55.92( 98.24%)
                                                  f 0.00(  0.00%)
                                         0.0000  0.1277  0.0000  0.8981  0.0000
  30. (1.88446) LP ( 2) O  3            s(  0.00%)p 1.00( 99.79%)d 0.00(  0.20%)
                                         0.0000  0.7728 -0.0143
  47. (1.90129) BD ( 1)Ni  2- H  4
               ( 36.54%)   0.6045*Ni  2 s( 48.94%)p 0.01(  0.40%)d 1.04( 50.65%)
                                                  f 0.00(  0.00%)
                                         0.0000  0.6991  0.0000 -0.1505  0.0000
               ( 63.46%)   0.7966* H  4 s( 99.90%)p 0.00(  0.10%)
                                         0.9995  0.0031
 ---------------- non-Lewis ----------------------------------------------------
"""


def test_one_centre_hybrids_and_metal_nao_coefficients_are_parsed():
    rows = {r["bond_num"]: r for r in NBOProps._parse_orbital_details(DETAILS)}
    lp = rows[29]["contributions"]
    assert len(lp) == 1 and lp[0]["one_centre"] and lp[0]["percentage"] == 100.0
    assert lp[0]["s_percent"] == 1.76 and lp[0]["d_percent"] == 98.24 and lp[0]["f_percent"] == 0.0
    assert lp[0]["nao_coefficients"] == [0.0, 0.1277, 0.0, 0.8981, 0.0]
    o = rows[30]["contributions"][0]
    assert o["p_percent"] == 99.79 and "nao_coefficients" not in o      # no d: nothing stored
    ni, h = rows[47]["contributions"]
    assert ni["d_percent"] == 50.65 and ni["nao_coefficients"][1] == 0.6991
    assert h["s_percent"] == 99.9 and "nao_coefficients" not in h


def test_frame_axes_follow_the_labels():
    centre, n1, n2 = np.zeros(3), np.array([0.0, 2.0, 0.0]), np.array([-1.9, 0.3, 0.0])
    R = nk.frame_axes(centre, n1, n2)
    assert np.allclose(R[0], [0, 1, 0])                      # x towards N1
    assert R[1] @ (n2 - centre) > 0 and abs(R[1] @ R[0]) < 1e-12   # y on N2's side, orthogonal
    assert np.allclose(R[2], np.cross(R[0], R[1]))
    with pytest.raises(ValueError):
        nk.frame_axes(centre, n1, 2 * n1)


def test_d_character_does_not_depend_on_the_input_orientation():
    """A pure dxy rotated 45 degrees about z is a pure dx2-y2 (one shell, NAO labels in NBO's
    order). The two d shells of a hybrid are rotated separately and their squares added."""
    labels = ["s", "dxy", "dxz", "dyz", "dx2y2", "dz2", "dxy", "dxz", "dyz", "dx2y2", "dz2"]
    coefs = [0.1, 0.8, 0, 0, 0, 0, 0.6, 0, 0, 0, 0]
    c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
    rot = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    assert nk.d_character(coefs, labels, np.eye(3))["dxy"] == pytest.approx(1.0)
    assert nk.d_character(coefs, labels, rot)["dx2y2"] == pytest.approx(1.0)
    w = nk.d_character([0, 1, 0, 0, 0, 0], ["s", "dz2", "dxy", "dxz", "dyz", "dx2y2"],
                       np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))       # z -> y
    assert sum(w.values()) == pytest.approx(1.0) and w["dz2"] < 1


def _results(nbos, details, sop=()):
    return {"natural_bond_orbitals": nbos, "bond_orbital_details": details,
            "natural_atomic_orbitals": [], "second_order_perturbation": list(sop)}


def _hyb(atom, pct, s, p, d=0.0, one=False):
    h = {"atom_num": atom, "percentage": pct, "s_percent": s, "p_percent": p, "d_percent": d}
    if one:
        h["one_centre"] = True
    return h


def test_chemical_keys_merge_atom_order_bond_polarisation_and_kekule():
    atom_map = {"Cl": 1, "C1": 2, "N": 3, "A1": 4}         # spec order
    nbos = [
        {"nbo_num": 1, "nbo_type": "BD*", "atoms": [2, 1], "occupancy": 0.07, "energy": 0.07},  # C1-Cl listed C first
        {"nbo_num": 2, "nbo_type": "LP", "atoms": [1], "occupancy": 1.97, "energy": -0.31},
        {"nbo_num": 3, "nbo_type": "LP", "atoms": [1], "occupancy": 1.96, "energy": -0.33},
        {"nbo_num": 4, "nbo_type": "BD", "atoms": [3, 9], "occupancy": 1.93, "energy": -0.40},  # 95 % on N
        {"nbo_num": 5, "nbo_type": "BD", "atoms": [4, 8], "occupancy": 1.70, "energy": -0.30},  # pi, partner unlabelled
        {"nbo_num": 6, "nbo_type": "RY", "atoms": [1], "occupancy": 0.001, "energy": 0.6},
    ]
    details = [
        {"bond_num": 1, "contributions": [_hyb(2, 59.3, 15.0, 84.7), _hyb(1, 40.7, 13.7, 85.9)]},
        {"bond_num": 2, "contributions": [_hyb(1, 100.0, 0.0, 99.9, one=True)]},
        {"bond_num": 3, "contributions": [_hyb(1, 100.0, 0.1, 99.8, one=True)]},
        {"bond_num": 4, "contributions": [_hyb(3, 95.0, 35.0, 65.0), _hyb(9, 5.0, 20.0, 60.0, 20.0)]},
        {"bond_num": 5, "contributions": [_hyb(4, 50.0, 0.0, 99.9), _hyb(8, 50.0, 0.0, 99.9)]},
        {"bond_num": 6, "contributions": [_hyb(1, 100.0, 1.0, 99.0, one=True)]},
    ]
    sop = [{"donor_nbo_num": 2, "acceptor_nbo_num": 1, "e2_energy": 1.5},
           {"donor_nbo_num": 3, "acceptor_nbo_num": 1, "e2_energy": 0.5}]
    flat = nk.chemical_descriptors(_results(nbos, details, sop), atom_map)
    assert flat["nbo_BDstar_sigma_Cl_C1_energy_mean"] == 0.07          # spec order, not NBO's
    assert flat["nbo_BDstar_sigma_Cl_C1_pol_C1"] == 59.3
    assert flat["nbo_LP_p_Cl_n"] == 2                                  # the two p lone pairs
    assert flat["nbo_LP_p_Cl_energy_min"] == -0.33 and flat["nbo_LP_p_Cl_energy_max"] == -0.31
    assert flat["nbo_LP_sp_N_from_bond"] == 1                           # BD 95 % on N -> LP on N
    assert flat["nbo_BD_pi_A1_any_energy_mean"] == -0.30                # per-atom pi key
    assert not any("RY" in k for k in flat)
    assert flat["nbo_e2_LP_p_Cl_to_BDstar_sigma_Cl_C1"] == pytest.approx(2.0)   # summed over the group


def test_open_shell_keys_carry_the_spin():
    nbos = [{"nbo_num": 1, "nbo_type": "LV", "atoms": [1], "occupancy": 0.16, "energy": -0.05, "spin": "beta"},
            {"nbo_num": 1, "nbo_type": "LP", "atoms": [1], "occupancy": 0.98, "energy": -0.20, "spin": "alpha"}]
    details = [{"bond_num": 1, "spin": "beta", "contributions": [_hyb(1, 100.0, 0.0, 100.0, one=True)]},
               {"bond_num": 1, "spin": "alpha", "contributions": [_hyb(1, 100.0, 0.0, 100.0, one=True)]}]
    flat = nk.chemical_descriptors(_results(nbos, details), {"C1": 1})
    assert flat["nbo_LV_p_C1_beta_occupancy_mean"] == 0.16
    assert flat["nbo_LP_p_C1_alpha_occupancy_mean"] == 0.98


def test_metal_d_lone_pairs_are_named_by_their_frame_d_orbital():
    """Ni at the origin, N1 on +y of the input axes, N2 on -x: the frame puts x on N1, so a lone
    pair that is pure dz2 in the input axes stays dz2 and one that is pure dxz becomes dyz."""
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [-2.0, 0.0, 0.0]])
    labels = ["dxy", "dxz", "dyz", "dx2y2", "dz2"]
    results = {"natural_atomic_orbitals": [{"atom_num": 1, "angular": a} for a in labels],
               "natural_bond_orbitals": [
                   {"nbo_num": 1, "nbo_type": "LP", "atoms": [1], "occupancy": 1.99, "energy": -0.22},
                   {"nbo_num": 2, "nbo_type": "LP", "atoms": [1], "occupancy": 1.98, "energy": -0.23}],
               "bond_orbital_details": [
                   {"bond_num": 1, "contributions": [{**_hyb(1, 100.0, 0.0, 0.0, 100.0, one=True),
                                                       "nao_coefficients": [0, 0, 0, 0, 1.0]}]},
                   {"bond_num": 2, "contributions": [{**_hyb(1, 100.0, 0.0, 0.0, 100.0, one=True),
                                                       "nao_coefficients": [0, 1.0, 0, 0, 0]}]}],
               "second_order_perturbation": []}
    flat = nk.chemical_descriptors(results, {"Ni": 1, "N1": 2, "N2": 3}, ["Ni", "N1", "N2"], coords)
    assert flat["nbo_LP_dz2_Ni_dz2_Ni"] == pytest.approx(1.0)
    assert flat["nbo_LP_dyz_Ni_dyz_Ni"] == pytest.approx(1.0)
    flat = nk.chemical_descriptors(results, {"Ni": 1, "N1": 2, "N2": 3})   # no frame: just "d"
    assert flat["nbo_LP_d_Ni_n"] == 2


def test_an_old_condensed_file_is_recognised_by_its_missing_hybrids():
    two_centre_only = {"bond_orbital_details": [{"bond_num": 1, "contributions": [_hyb(1, 50, 30, 70)]}]}
    assert not nk.has_hybrids(two_centre_only)
    assert nk.has_hybrids({"bond_orbital_details": [{"contributions": [_hyb(1, 100, 0, 100, one=True)]}]})


MINIMAL = {
    "name": "mini",
    "data": {"root": "DATA/x"},
    "atoms": {"A": "C", "B": "N", "M": None},
    "numbering": {"backend": "first_n"},
    "descriptors": {"orca": {"nbo": {"atoms": ["A", "B", "M"]}}},
}


def test_spec_naming_defaults_to_rank_and_checks_the_frame():
    spec = validate_spec(MINIMAL)
    assert spec.orca["nbo"]["naming"] == "rank" and spec.orca["nbo"]["frame"] is None
    raw = copy.deepcopy(MINIMAL)
    raw["descriptors"]["orca"]["nbo"].update(naming="chemical", frame=["M", "A", "B"])
    assert validate_spec(raw).orca["nbo"]["frame"] == ["M", "A", "B"]
    for bad in ({"naming": "energy"}, {"naming": "chemical", "frame": ["M", "A"]},
                {"naming": "chemical", "frame": ["M", "A", "Zz"]}, {"frame": ["M", "A", "B"]}):
        raw = copy.deepcopy(MINIMAL)
        raw["descriptors"]["orca"]["nbo"].update(bad)
        with pytest.raises(SpecError):
            validate_spec(raw)


NLMO = """
 NATURAL LOCALIZED MOLECULAR ORBITAL (NLMO) ANALYSIS:
 1 pass over DM reduced the maximum off-diagonal element to 0.20E-09
 Hybridization/Polarization Analysis of NLMOs in NAO Basis:
 NLMO / Occupancy / Percent from Parent NBO / Atomic Hybrid Contributions
 -------------------------------------------------------------------------------
  33. (2.00000)  89.4141% LP ( 1) N  4
                           4.667% Ni  2 s( 36.41%)p 0.03(  1.26%)d 1.71( 62.32%)
                                                  f 0.00(  0.02%)
                           2.838%  H  3 s( 99.90%)p 0.00(  0.10%)
                          89.424%  N  4 s( 34.52%)p 1.89( 65.42%)d 0.00(  0.05%)
                                                  f 0.00(  0.01%)
  47. (2.00000)  95.0395% BD ( 1)Ni  2- H  3
                          37.800% Ni  2 s( 24.23%)p 0.01(  0.29%)d 3.12( 75.49%)
                          59.700%  H  3 s( 99.90%)p 0.00(  0.10%)
                           0.439%  N  4 s( 11.42%)p 7.68( 87.73%)d 0.07(  0.83%)
 125. (2.00000)  99.0000% BD ( 1) C 99- C100
                          99.000%  C100 s( 30.00%)p 2.33( 70.00%)

 NBO analysis completed in 55.08 CPU seconds (116 wall seconds)
"""


def test_nlmo_section_is_parsed():
    p = NBOProps.__new__(NBOProps)
    p.spin_sections, p.spin, p.results = {None: NLMO}, "alpha", {}
    rows = {r["nlmo_num"]: r for r in p.get_NLMO()}
    assert sorted(rows) == [33, 47, 125]
    lp = rows[33]
    assert lp["parent_type"] == "LP" and lp["parent_percent"] == 89.4141 and lp["occupancy"] == 2.0
    assert lp["atoms"] == {2: 4.667, 3: 2.838, 4: 89.424}
    assert rows[47]["parent_type"] == "BD" and rows[47]["parent_atoms"] == "Ni  2- H  3"
    assert rows[125]["atoms"] == {100: 99.0}                 # atom numbers past 99 run into the symbol


def test_nlmo_columns_reuse_the_nbo_keys_and_check_the_parent():
    atom_map = {"Ni": 2, "H2": 3, "N1": 4}
    nbos = [{"nbo_num": 33, "nbo_type": "LP", "atoms": [4], "occupancy": 1.9, "energy": -0.4},
            {"nbo_num": 47, "nbo_type": "BD", "atoms": [2, 3], "occupancy": 1.9, "energy": -0.3}]
    details = [{"bond_num": 33, "contributions": [_hyb(4, 100.0, 34.5, 65.4, one=True)]},
               {"bond_num": 47, "contributions": [_hyb(2, 38.0, 24.2, 0.3, 75.5), _hyb(3, 62.0, 99.9, 0.1)]}]
    p = NBOProps.__new__(NBOProps)
    p.spin_sections, p.spin, p.results = {None: NLMO}, "alpha", {}
    flat = nk.nlmo_descriptors(p.get_NLMO(), _results(nbos, details), atom_map)
    assert flat["nbo_LP_sp_N1_nlmo_parent"] == 89.4141
    assert flat["nbo_LP_sp_N1_nlmo_on_Ni"] == 4.667 and flat["nbo_LP_sp_N1_nlmo_on_H2"] == 2.838
    assert "nbo_LP_sp_N1_nlmo_on_N1" not in flat                       # own atom: that is the parent
    assert flat["nbo_BD_sigma_Ni_H2_nlmo_on_N1"] == 0.439          # spec order: Ni before H2
    # NBO 33 an LV in this run: the NLMO (parent LP) must not be attached to it
    nbos[0]["nbo_type"] = "LV"
    flat = nk.nlmo_descriptors(p.get_NLMO(), _results(nbos, details), atom_map)
    assert not any(k.startswith("nbo_LV") for k in flat)


def test_absent_e2_pair_is_zero_only_when_both_orbitals_exist():
    import pandas as pd
    df = pd.DataFrame({
        "nbo_LP_sp_N1_n": [1, 1, None, 1],
        "nbo_BDstar_sigma_H2_Ni_n": [1, 1, None, None],
        "nbo_e2_LP_sp_N1_to_BDstar_sigma_H2_Ni": [52.0, None, None, None],
        "nbo_LV_p_C1_beta_n": [1, None, 1, 1],
        "nbo_LP_p_C1_beta_n": [1, 1, 1, 1],
        "nbo_e2_LP_p_C1_to_LV_p_C1_beta": [None, None, 3.0, None],
    })
    out = nk.fill_absent_e2(df)
    assert out["nbo_e2_LP_sp_N1_to_BDstar_sigma_H2_Ni"].tolist()[:2] == [52.0, 0.0]
    assert out["nbo_e2_LP_sp_N1_to_BDstar_sigma_H2_Ni"].isna().tolist()[2:] == [True, True]
    assert out["nbo_e2_LP_p_C1_to_LV_p_C1_beta"].tolist()[0] == 0.0
    assert math.isnan(out["nbo_e2_LP_p_C1_to_LV_p_C1_beta"].tolist()[1])   # no LV in that conformer
    assert out["nbo_e2_LP_p_C1_to_LV_p_C1_beta"].tolist()[2:] == [3.0, 0.0]


def test_spec_nlmo_basename_needs_chemical_naming():
    raw = copy.deepcopy(MINIMAL)
    raw["descriptors"]["orca"]["nbo"].update(naming="chemical", nlmo_basename="nlmo")
    assert validate_spec(raw).orca["nbo"]["nlmo_basename"] == "nlmo"
    for bad in ({"nlmo_basename": "nlmo"}, {"naming": "chemical", "nlmo_basename": "a/nlmo"}):
        raw = copy.deepcopy(MINIMAL)
        raw["descriptors"]["orca"]["nbo"].update(bad)
        with pytest.raises(SpecError):
            validate_spec(raw)
