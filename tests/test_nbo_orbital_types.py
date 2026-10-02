"""Orbital types accepted by the NBO summary / detail / directionality parsers (``NBOProps``).

Two things used to drop rows from ``results['natural_bond_orbitals']``:
  * the type alternation was ``(BD*?|LP|CR|RY)`` so LV, LP*, RY*, 3C, 3Cn and 3C* rows were skipped;
  * ``")\\s+"`` demanded whitespace before the atom list, but a two-letter element abuts the
    parenthesis ("LP ( 1)Ni  2"), so *every* one-centre orbital on Ni (CR, LP, RY, LV) was skipped.
"""
from pathlib import Path

import pytest

from desc_utils import orca_descriptors as od
from desc_utils.data_extraction import _collect_selected_nbo_results

from tests.conftest import LNIH2_ROOT

REAL_OUT = LNIH2_ROOT / "AATATIAROAAKIN-SNVBAGLBSA-N" / "conf_1" / "calcParam.out"

DETAILS = """
     (Occupancy)   Bond orbital / Coefficients / Hybrids
 -------------------------------------------------------------------------------
 ---------------- Lewis --------------------------------------------------------
   1. (2.00000) CR ( 1)Ni  2            s(100.00%)
                                         1.0000  0.0000
  29. (1.99018) LP ( 1)Ni  2            s(  0.00%)p 0.00(  0.00%)d 1.00(100.00%)
                                         0.0000  0.0000  0.0000  0.0000  0.0000
  46. (1.88964) BD ( 1) H  1-Ni  2
               ( 61.50%)   0.7842* H  1 s( 99.93%)p 0.00(  0.07%)
                                         1.0000  0.0000
               ( 38.50%)   0.6205*Ni  2 s( 41.05%)p 0.01(  0.30%)d 1.43( 58.65%)
                                         0.0000  0.0000  0.6407  0.0000  0.0000
  50. (1.95000) 3C ( 1) C  5- C  6- C  7
               ( 33.00%)   0.5745* C  5 s( 33.00%)p 2.03( 67.00%)
                                         0.0000  0.5745  0.0000
 ---------------- non-Lewis ----------------------------------------------------
  87. (0.26397) LV ( 1)Ni  2            s( 99.10%)p 0.00(  0.31%)d 0.01(  0.59%)
                                                  f 0.00(  0.00%)
                                         0.0000  0.0000  0.0000  0.9955 -0.0060
  88. (0.05000) BD*( 1)Ni  2- H  3
               ( 38.50%)   0.6205* H  3 s( 99.93%)p 0.00(  0.07%)
                                         1.0000  0.0000
 128. (0.00100) RY ( 1) H  1            s(  0.20%)p99.99( 99.80%)
                                         0.0000  0.0447  0.0000
 129. (0.00050) RY*( 1) H  1            s( 50.00%)p 1.00( 50.00%)
                                         0.0000  0.0447  0.0000
 130. (0.01000) LP*( 1) N  4            s( 30.00%)p 2.33( 70.00%)
                                         0.0000  0.0447  0.0000
 131. (0.00200) 3Cn( 1) C  5- C  6- C  7
                                         0.0000  0.0447  0.0000
 132. (0.00100) 3C*( 1) C  5- C  6- C  7
                                         0.0000  0.0447  0.0000

"""

DIRECTIONALITY = """
 NHO DIRECTIONALITY AND BOND BENDING (deviation from line of nuclear centers at
                                    the position of maximum hybrid amplitude)

         [Thresholds for printing:  angular deviation  >  1.0 degree]
                                    hybrid p-character > 25.0%
                                    orbital occupancy  >  0.10e

                        Line of Centers        Hybrid 1             Hybrid 2
                        ---------------  -------------------  ------------------
           NBO            Theta   Phi    Theta   Phi    Dev   Theta   Phi    Dev
 ===============================================================================
  29. LP ( 1)Ni  2         --     --     90.0   180.0    --
  46. BD ( 1) H  1-Ni  2    90.0  254.9    91.0  253.7    1.3    --     --     --
  87. LV ( 1)Ni  2         --     --     90.0    90.0    --
"""

SUMMARY = """

 NATURAL BOND ORBITALS (Summary):

                                                     Principal Delocalizations
           NBO                 Occupancy    Energy   (geminal,vicinal,remote)
 ===============================================================================
 Molecular unit  1  (C10H14N2NiO)
 ------ Lewis --------------------------------------
    1. CR ( 1)Ni  2             2.00000  -292.68044
   29. LP ( 1)Ni  2             1.99018    -0.20356  168(r),454(r),141(g),128(v)
   46. BD ( 1) H  1-Ni  2       1.88964    -0.24567  87(g),88(g)
   50. 3C ( 1) C  5- C  6- C  7  1.95000    -0.40000
 ------ non-Lewis ----------------------------------
   87. LV ( 1)Ni  2             0.26397    -0.07374
   88. BD*( 1)Ni  2- H  3       0.05000     0.20000
  108. BD*( 2) C 14- N 15       0.19591    -0.19089
  128. RY ( 1) H  1             0.00100     0.80000
  129. RY*( 1) H  1             0.00050     0.90000
  130. LP*( 1) N  4             0.01000     0.30000
  131. 3Cn( 1) C  5- C  6- C  7  0.00200     0.50000
  132. 3C*( 1) C  5- C  6- C  7  0.00100     0.60000

 $CHOOSE
"""


def _fake_nbo_file(tmp_path: Path) -> Path:
    out = tmp_path / "fake.out"
    out.write_text("Now starting NBO....\n" + DETAILS + DIRECTIONALITY + SUMMARY + "\n *** returned from  NBO  program ***\n")
    return out


def _parse(path: Path, block_index: int = 0):
    reader = od.NBOProps(path, block_index=block_index)
    reader.get_NBO_summary()
    reader.get_orbital_details()
    reader.get_NHO_directionality()
    return reader


def _by_num(entries, key="nbo_num"):
    return {e[key]: e for e in entries}


def test_summary_accepts_every_orbital_type_and_ni_without_space(tmp_path):
    nbos = _parse(_fake_nbo_file(tmp_path)).results["natural_bond_orbitals"]
    by_num = _by_num(nbos)
    assert sorted(by_num) == [1, 29, 46, 50, 87, 88, 108, 128, 129, 130, 131, 132]
    assert {e["nbo_type"] for e in nbos} == {"CR", "LP", "BD", "3C", "LV", "BD*", "RY", "RY*", "LP*", "3Cn", "3C*"}
    # one-centre Ni orbitals have no space between ")" and "Ni"
    assert by_num[1] == {"nbo_num": 1, "nbo_type": "CR", "bond_index": 1, "atoms": [2], "occupancy": 2.0, "energy": -292.68044}
    assert by_num[29]["atoms"] == [2] and by_num[29]["occupancy"] == 1.99018
    assert by_num[87] == {"nbo_num": 87, "nbo_type": "LV", "bond_index": 1, "atoms": [2], "occupancy": 0.26397, "energy": -0.07374}
    assert by_num[50]["atoms"] == [5, 6, 7] and by_num[131]["nbo_type"] == "3Cn" and by_num[132]["nbo_type"] == "3C*"
    assert by_num[129]["nbo_type"] == "RY*" and by_num[130]["nbo_type"] == "LP*"


def test_orbital_details_accept_lv_and_ni_without_space(tmp_path):
    details = _by_num(_parse(_fake_nbo_file(tmp_path)).results["bond_orbital_details"], key="bond_num")
    assert sorted(details) == [1, 29, 46, 50, 87, 88, 128, 129, 130, 131, 132]
    lv = details[87]
    assert (lv["bond_type"], lv["bond_index"], lv["atoms"], lv["occupancy"]) == ("LV", 1, [2], 0.26397)
    assert lv["contributions"] == []          # one-centre orbitals carry no per-atom contribution block
    assert details[29]["bond_type"] == "LP" and details[29]["atoms"] == [2]
    assert details[46]["atoms"] == [1, 2] and [c["atom_num"] for c in details[46]["contributions"]] == [1, 2]
    assert details[50]["bond_type"] == "3C" and details[50]["atoms"] == [5, 6, 7]


def test_directionality_accepts_lp_and_lv_on_ni(tmp_path):
    dirs = _by_num(_parse(_fake_nbo_file(tmp_path)).results["nho_directionality"])
    assert sorted(dirs) == [29, 46, 87]
    assert dirs[87]["nbo_type"] == "LV" and dirs[87]["atoms"] == [2]
    assert dirs[87]["line_theta"] is None and dirs[87]["hybrid_theta"] == 90.0
    assert dirs[46]["deviation"] == 1.3


def test_lv_reaches_nbo_columns_without_moving_homo_reference(tmp_path):
    reader = _parse(_fake_nbo_file(tmp_path))
    flat = _collect_selected_nbo_results(reader.results, {"H1": 1, "Ni": 2, "H2": 3})
    # HOMO reference = highest nbo_num with occupancy >= 0.1. LV 87 (0.26) qualifies but BD*(2) 108
    # (0.196) is higher, so the reference stays at 108 and the LV becomes hm21 (108 - 87).
    assert flat["nbo_hm21_Ni_type"] == "LV"
    assert flat["nbo_hm21_Ni_bond_index"] == 1
    assert flat["nbo_hm21_Ni_occupancy"] == 0.26397
    assert flat["nbo_hm21_Ni_energy"] == -0.07374
    assert flat["bond_orbital_details_hm21_Ni_bond_type"] == "LV"
    assert flat["nho_directionality_hm21_Ni_type"] == "LV"
    # the other Ni one-centre orbitals now reach the columns too
    assert flat["nbo_hm107_Ni_type"] == "CR" and flat["nbo_hm79_Ni_type"] == "LP"
    assert flat["nbo_hm62_H1_Ni_type"] == "BD" and flat["nbo_lp20_H1_type"] == "RY"


def test_lv_becomes_homo_reference_when_it_is_the_highest_occupied(tmp_path):
    # Without the BD*(2) 108 row the LV (0.26 >= 0.1) is the highest-numbered "occupied" NBO
    # and the rank tokens shift: the LV itself is "homo", BD 46 goes from hm62 to hm41.
    out = tmp_path / "fake.out"
    text = "Now starting NBO....\n" + DETAILS + DIRECTIONALITY + SUMMARY + "\n *** returned from  NBO  program ***\n"
    out.write_text(text.replace("  108. BD*( 2) C 14- N 15       0.19591    -0.19089\n", ""))
    reader = _parse(out)
    flat = _collect_selected_nbo_results(reader.results, {"H1": 1, "Ni": 2, "H2": 3})
    assert flat["nbo_homo_Ni_type"] == "LV" and flat["nbo_homo_Ni_occupancy"] == 0.26397
    assert flat["nbo_hm41_H1_Ni_type"] == "BD"
    assert flat["nbo_lp1_Ni_H2_type"] == "BD*"


@pytest.mark.skipif(not REAL_OUT.exists(), reason="needs DATA/")
def test_real_orca61_output_ni_orbitals():
    # block 0 (closed shell): 43 one-centre Ni orbitals that used to be dropped, no LV
    nbos0 = _parse(REAL_OUT, block_index=0).results["natural_bond_orbitals"]
    ni0 = [e for e in nbos0 if e["atoms"] == [2]]
    assert len(ni0) == 43 and {e["nbo_type"] for e in ni0} == {"CR", "LP", "RY"}
    assert not any(e["nbo_type"] == "LV" for e in nbos0)
    assert max(e["nbo_num"] for e in nbos0 if e["occupancy"] >= 0.1) == 108

    # block 1 (open shell, alpha summary): "87. LV ( 1)Ni  2   0.26397  -0.07374"
    reader1 = _parse(REAL_OUT, block_index=1)
    lv = [e for e in reader1.results["natural_bond_orbitals"] if e["nbo_type"] == "LV" and e["nbo_num"] == 87]
    assert lv and lv[0]["atoms"] == [2] and lv[0]["occupancy"] == 0.26397 and lv[0]["energy"] == -0.07374
    assert any(e["bond_type"] == "LV" and e["bond_num"] == 87 for e in reader1.results["bond_orbital_details"])
    # LV 87 qualifies for the >= 0.1 HOMO heuristic but BD*(2) C14-N15 (108, 0.196) still wins
    assert max(e["nbo_num"] for e in reader1.results["natural_bond_orbitals"] if e["occupancy"] >= 0.1) == 108
    flat = _collect_selected_nbo_results(reader1.results, {"H1": 1, "Ni": 2, "H2": 3})
    assert flat["nbo_hm21_Ni_type"] == "LV" and flat["nbo_hm21_Ni_occupancy"] == 0.26397
