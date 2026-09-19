"""Open-shell NBO 7 blocks: one spin set per ``NBOProps`` (``spin`` option).

NBO 7 prints, for a UHF/UKS block, an alpha analysis (bond orbital details, NHO
directionality, second-order table, summary) followed by a beta analysis, and only the
beta summary is followed by ``$CHOOSE``. The summary slice used to run from the alpha
header to ``$CHOOSE`` and therefore swallowed the whole beta set.
"""
from pathlib import Path

import pytest

from desc_utils import orca_descriptors as od
from desc_utils.data_extraction import _collect_selected_nbo_results

from conftest import LNIH2_ROOT

REAL_OUT = LNIH2_ROOT / "AATATIAROAAKIN-SNVBAGLBSA-N" / "conf_1" / "calcParam.out"

NPA = """
 NATURAL POPULATIONS:  Natural atomic orbital occupancies

  NAO Atom No lang   Type(AO)    Occupancy      Energy
 -------------------------------------------------------
   1    H  1  s      Val( 1s)     1.18360      -0.32349

 Summary of Natural Population Analysis:

                                       Natural Population
                Natural    ---------------------------------------------
    Atom No    Charge        Core      Valence    Rydberg      Total
 --------------------------------------------------------------------
    H  1   -0.18360      0.00000     1.18360    0.00000     1.18360
   Ni  2    0.60000     18.00000     9.30000    0.10000    27.40000
    H  3   -0.20000      0.00000     1.19000    0.01000     1.20000
 ====================================================================
   * Total *  0.21640     18.00000    11.67360    0.11000    29.78360

                                 Natural Population
 --------------------------------------------------------
   Atom No         Natural Electron Configuration
"""


def _banner(label: str) -> str:
    # label is "Alpha" or "Beta " exactly as NBO 7 prints them (padded to 5 chars)
    return f"""
 ***************************************************
 *******         {label} spin orbitals         *******
 ***************************************************
"""


def _spin_set(spin: str, occ_45: str, energy_45: str, nho_theta: str, e2: str, occ_87: str) -> str:
    """One complete per-spin NBO analysis with distinguishable numbers."""
    return f"""
 NATURAL BOND ORBITAL ANALYSIS, {spin} spin orbitals:

     (Occupancy)   Bond orbital / Coefficients / Hybrids
 ------------------ Lewis ------------------------------------------------------
  45. ({occ_45}) BD ( 1) H  1-Ni  2
                ( 50.00%)   0.7071* H  1 s( 99.91%)p 0.00(  0.09%)
                ( 50.00%)   0.7071*Ni  2 s( 30.00%)p 0.50( 15.00%)d 1.83( 55.00%)
  46. (0.90000) LP ( 1) N  4            s( 20.00%)p 4.00( 80.00%)
 ---------------- non-Lewis ----------------------------------------------------
  87. ({occ_87}) BD*( 1) H  1-Ni  2
                ( 50.00%)   0.7071* H  1 s( 99.91%)p 0.00(  0.09%)


 NHO DIRECTIONALITY AND BOND BENDING (deviation from line of nuclear centers at
                                      the position of maximum hybrid amplitude)

         [Thresholds for printing:  angular deviation  >  1.0 degree]
                                    p- or d-character  > 25.0%
                                    orbital occupancy  >  0.10e

                        Line of Centers        Hybrid 1             Hybrid 2
                        ---------------  -------------------  ------------------
           NBO            Theta   Phi    Theta   Phi    Dev   Theta   Phi    Dev
 ===============================================================================
  45. BD ( 1) H  1-Ni  2   {nho_theta}   53.4    79.7   50.0   3.5   102.2  234.7   1.6


 SECOND ORDER PERTURBATION THEORY ANALYSIS OF FOCK MATRIX IN NBO BASIS

     Threshold for printing:   0.50 kcal/mol
    (Intermolecular threshold: 0.05 kcal/mol)
                                                          E(2) E(NL)-E(L) F(L,NL)
      Donor (L) NBO              Acceptor (NL) NBO      kcal/mol   a.u.    a.u.
 ===============================================================================

 within unit  1
   45. BD ( 1) H  1-Ni  2      87. BD*( 1) H  1-Ni  2      {e2}    1.03   0.032


 NATURAL BOND ORBITALS (Summary):

                                                     Principal Delocalizations
           NBO                 Occupancy    Energy   (geminal,vicinal,remote)
 ===============================================================================
 Molecular unit  1  (H2Ni)
 ------ Lewis --------------------------------------
   45. BD ( 1) H  1-Ni  2       {occ_45}    {energy_45}  87(g)
   46. LP ( 1) N  4             0.90000    -0.30000
 ------ non-Lewis ----------------------------------
   87. BD*( 1) H  1-Ni  2       {occ_87}     0.14172
          -------------------------------
                 Total Lewis    1.88000  ( 98.2733%)
           Valence non-Lewis    0.02000  (  1.5628%)
          -------------------------------
               Total unit  1    1.90000  (100.0000%)
              Charge unit  1    0.10000
"""


ALPHA = dict(occ_45="0.98000", energy_45="-0.35419", nho_theta="97.1", e2="0.63", occ_87="0.01000")
BETA = dict(occ_45="0.95164", energy_45="-0.48630", nho_theta="95.0", e2="0.71", occ_87="0.00243")

CHOOSE = """
 $CHOOSE
  ALPHA
   LONE 2 1 END
   BOND S 1 2 END
  END
  BETA
   LONE 2 1 END
   BOND S 1 2 END
  END
 $END

 Checkpoint orbital transformation(s) sent to host program

 NBO analysis completed in 36.86 CPU seconds (38 wall seconds)
"""

OPEN_SHELL_BLOCK = (
    "Now starting NBO....\n" + NPA
    + _banner("Alpha") + NPA + _spin_set("alpha", **ALPHA)
    + _banner("Beta ") + NPA + _spin_set("beta", **BETA)
    + CHOOSE + "\n *** returned from  NBO  program ***\n"
)
CLOSED_SHELL_BLOCK = (
    "Now starting NBO....\n" + NPA + _spin_set("", **ALPHA).replace(",  spin orbitals", "")
    + "\n $CHOOSE\n $END\n\n Checkpoint orbital transformation(s) sent to host program\n"
    + "\n *** returned from  NBO  program ***\n"
)


def _write(tmp_path: Path, text: str) -> Path:
    out = tmp_path / "fake.out"
    out.write_text(text)
    return out


def _parse(path: Path, **kwargs) -> od.NBOProps:
    reader = od.NBOProps(path, **kwargs)
    reader.get_NPA()
    reader.get_NAO()
    reader.get_NBO_summary()
    reader.get_orbital_details()
    reader.get_NHO_directionality()
    reader.get_second_order_perturbation()
    return reader


def _by_num(rows, key="nbo_num"):
    return {r[key]: r for r in rows}


def test_open_shell_default_returns_alpha_set_only(tmp_path):
    reader = _parse(_write(tmp_path, OPEN_SHELL_BLOCK))
    res = reader.results
    assert not reader.is_closed_shell and res["spin"] == "alpha"

    nbo = res["natural_bond_orbitals"]
    assert [e["nbo_num"] for e in nbo] == [45, 46, 87]          # each NBO once, no beta duplicates
    assert all(e["spin"] == "alpha" for e in nbo)
    assert _by_num(nbo)[45]["occupancy"] == 0.98 and _by_num(nbo)[45]["energy"] == -0.35419
    assert _by_num(nbo)[87]["occupancy"] == 0.01                # beta BD* 87 (0.00243) does not leak in
    assert max(e["occupancy"] for e in nbo) <= 2.0               # no NHO row parsed as "occupancy 97.1"

    details = res["bond_orbital_details"]
    assert [e["bond_num"] for e in details] == [45, 46, 87]
    assert details[0]["occupancy"] == 0.98 and details[0]["spin"] == "alpha"
    assert details[0]["contributions"][0]["atom_num"] == 1

    nho = res["nho_directionality"]
    assert len(nho) == 1 and nho[0]["nbo_num"] == 45 and nho[0]["line_theta"] == 97.1 and nho[0]["spin"] == "alpha"

    sop = res["second_order_perturbation"]
    assert len(sop) == 1 and sop[0]["e2_energy"] == 0.63 and sop[0]["spin"] == "alpha"

    # NPA / NAO always come from the spin-summed tables, so all three atoms are present once.
    assert sorted(res["natural_population_analysis"]["atoms"]) == [1, 2, 3]
    assert res["natural_population_analysis"]["totals"]["charge"] == 0.2164
    assert len(res["natural_atomic_orbitals"]) == 1


def test_open_shell_beta_set(tmp_path):
    res = _parse(_write(tmp_path, OPEN_SHELL_BLOCK), spin="beta").results
    assert res["spin"] == "beta"
    nbo = _by_num(res["natural_bond_orbitals"])
    assert sorted(nbo) == [45, 46, 87] and all(e["spin"] == "beta" for e in nbo.values())
    assert nbo[45]["occupancy"] == 0.95164 and nbo[45]["energy"] == -0.4863 and nbo[87]["occupancy"] == 0.00243
    assert res["bond_orbital_details"][0]["occupancy"] == 0.95164
    assert res["nho_directionality"][0]["line_theta"] == 95.0
    assert res["second_order_perturbation"][0]["e2_energy"] == 0.71


def test_open_shell_both_sets_are_tagged(tmp_path):
    res = _parse(_write(tmp_path, OPEN_SHELL_BLOCK), spin="both").results
    assert res["spin"] == "both"
    nbo = res["natural_bond_orbitals"]
    assert [(e["spin"], e["nbo_num"]) for e in nbo] == [
        ("alpha", 45), ("alpha", 46), ("alpha", 87), ("beta", 45), ("beta", 46), ("beta", 87),
    ]
    assert [e["occupancy"] for e in nbo if e["nbo_num"] == 45] == [0.98, 0.95164]
    assert [e["spin"] for e in res["bond_orbital_details"]] == ["alpha"] * 3 + ["beta"] * 3
    assert [e["line_theta"] for e in res["nho_directionality"]] == [97.1, 95.0]
    assert [e["e2_energy"] for e in res["second_order_perturbation"]] == [0.63, 0.71]

    # The flattener keeps both spins apart with its "_2" duplicate suffix -- only when asked for both.
    flat = _collect_selected_nbo_results(res, {"H1": 1, "Ni": 2, "H2": 3})
    assert flat["nbo_hm1_H1_Ni_occupancy"] == 0.98 and flat["nbo_hm1_2_H1_Ni_occupancy"] == 0.95164


def test_open_shell_alpha_flattens_without_duplicate_columns(tmp_path):
    res = _parse(_write(tmp_path, OPEN_SHELL_BLOCK)).results
    flat = _collect_selected_nbo_results(res, {"H1": 1, "Ni": 2, "H2": 3})
    assert not any("_2_" in c for c in flat if c.startswith("nbo_"))
    assert flat["nbo_hm1_H1_Ni_occupancy"] == 0.98
    assert flat["nbo_lp41_H1_Ni_occupancy"] == 0.01
    assert flat["nho_directionality_hm1_H1_Ni_line_theta"] == 97.1
    assert flat["second_order_hm1_H1_Ni_to_lp41_H1_Ni_e2_energy"] == 0.63


def test_closed_shell_block_ignores_spin_and_is_untagged(tmp_path):
    path = _write(tmp_path, CLOSED_SHELL_BLOCK)
    for spin in ("alpha", "beta", "both"):
        reader = _parse(path, spin=spin)
        assert reader.is_closed_shell and reader.results["spin"] is None
        nbo = reader.results["natural_bond_orbitals"]
        assert [e["nbo_num"] for e in nbo] == [45, 46, 87]
        assert not any("spin" in e for e in nbo)
        assert not any("spin" in e for e in reader.results["second_order_perturbation"])


def test_invalid_spin_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="spin must be one of"):
        od.NBOProps(_write(tmp_path, CLOSED_SHELL_BLOCK), spin="gamma")


@pytest.mark.skipif(not REAL_OUT.exists(), reason="needs DATA/")
def test_real_orca61_open_shell_block1():
    """Block 1 of the real file is the UKS cation: 736 NBOs per spin, $CHOOSE only after beta."""
    alpha = _parse(REAL_OUT, block_index=1).results
    nbo = alpha["natural_bond_orbitals"]
    nums = [e["nbo_num"] for e in nbo]
    assert len(nums) == len(set(nums)) == 691                    # was ~1494 with the beta set swallowed
    assert max(e["occupancy"] for e in nbo) <= 1.0                # UKS: one electron per spin orbital
    assert {e["spin"] for e in nbo} == {"alpha"}
    by_num = _by_num(nbo)
    assert by_num[47]["atoms"] == [1, 3] and by_num[47]["occupancy"] == 0.78712   # alpha-only H1-H3 bond

    beta = _parse(REAL_OUT, block_index=1, spin="beta").results
    assert len(beta["natural_bond_orbitals"]) == 691 and {e["spin"] for e in beta["natural_bond_orbitals"]} == {"beta"}
    assert _by_num(beta["natural_bond_orbitals"])[45]["atoms"] == [1, 2]          # beta Lewis structure has Ni-H bonds
    assert len(beta["second_order_perturbation"]) != len(alpha["second_order_perturbation"])

    flat = _collect_selected_nbo_results(alpha, {"H1": 1, "Ni": 2, "H2": 3})
    assert not any("_2_" in c for c in flat if c.startswith("nbo_"))

    closed = _parse(REAL_OUT, block_index=0).results
    assert closed["spin"] is None and len(closed["natural_bond_orbitals"]) == 691
