"""Second-order perturbation table of the NBO 7 block (``NBOProps.get_second_order_perturbation``)."""
from pathlib import Path

import pytest

from desc_utils import orca_descriptors as od
from desc_utils.data_extraction import _collect_selected_nbo_results

from tests.conftest import LNIH2_ROOT

REAL_OUT = LNIH2_ROOT / "AATATIAROAAKIN-SNVBAGLBSA-N" / "conf_1" / "calcParam.out"

HEADER = """
 SECOND ORDER PERTURBATION THEORY ANALYSIS OF FOCK MATRIX IN NBO BASIS

     Threshold for printing:   0.50 kcal/mol
    (Intermolecular threshold: 0.05 kcal/mol)
                                                          E(2) E(NL)-E(L) F(L,NL)
      Donor (L) NBO              Acceptor (NL) NBO      kcal/mol   a.u.    a.u.
 ===============================================================================
"""
NBO7_TABLE = HEADER + """
 within unit  1
   29. LP ( 1)Ni  2           128. RY ( 1) H  1            0.88    0.85   0.024
   46. BD ( 1) H  1-Ni  2      88. BD*( 1)Ni  2- H  3     29.34    0.38   0.095
   47. BD ( 1) H  1- H  3      87. LV ( 1)Ni  2          128.37    0.28   0.240

 from unit  1 to unit  2
   29. LP ( 1)Ni  2           119. BD*( 1) C 20- H 29      0.16    0.62   0.009
"""
LEGACY_TABLE = HEADER + """
  BD ( 1) H  1- H  2   ->  BD*( 1) H  3- H  4      12.34     0.56     0.078
  LP ( 1) O  5         ->  RY ( 1) C  6             0.91     1.20     0.030
"""
SUMMARY = """

 NATURAL BOND ORBITALS (Summary):

                                                     Principal Delocalizations
           NBO                 Occupancy    Energy   (geminal,vicinal,remote)
 ===============================================================================
 Molecular unit  1  (C10H14N2NiO)
 ------ Lewis --------------------------------------
   29. LP ( 1)Ni  2             1.99000    -0.50000
   46. BD ( 1) H  1-Ni  2       1.88964    -0.24567
   47. BD ( 1) H  1- H  3       1.90000    -0.30000
 ------ non-Lewis ----------------------------------
   87. BD*( 1) H  1-Ni  2       0.22750     0.14172
   88. BD*( 1)Ni  2- H  3       0.05000     0.20000
  128. RY ( 1) H  1             0.00100     0.80000

 $CHOOSE
"""


def _fake_nbo_file(tmp_path: Path, table: str) -> Path:
    out = tmp_path / "fake.out"
    out.write_text("Now starting NBO....\n" + table + SUMMARY + "\n *** returned from  NBO  program ***\n")
    return out


def _parse(path: Path):
    reader = od.NBOProps(path, block_index=0)
    reader.get_NBO_summary()
    reader.get_second_order_perturbation()
    return reader


def test_nbo7_numbered_rows_are_parsed(tmp_path):
    sop = _parse(_fake_nbo_file(tmp_path, NBO7_TABLE)).results["second_order_perturbation"]
    assert len(sop) == 4
    assert sop[0] == {
        "donor_nbo_num": 29, "donor_type": "LP", "donor_index": 1, "donor_atoms": "Ni  2", "donor_unit": 1,
        "acceptor_nbo_num": 128, "acceptor_type": "RY", "acceptor_index": 1, "acceptor_atoms": "H  1", "acceptor_unit": 1,
        "e2_energy": 0.88, "energy_diff": 0.85, "fock_element": 0.024,
    }
    lv = sop[2]
    assert (lv["donor_atoms"], lv["acceptor_type"], lv["acceptor_nbo_num"], lv["e2_energy"]) == ("H  1- H  3", "LV", 87, 128.37)
    inter = sop[3]
    assert (inter["donor_unit"], inter["acceptor_unit"], inter["acceptor_atoms"]) == (1, 2, "C 20- H 29")


def test_legacy_arrow_rows_are_parsed(tmp_path):
    sop = _parse(_fake_nbo_file(tmp_path, LEGACY_TABLE)).results["second_order_perturbation"]
    assert [(e["donor_type"], e["acceptor_type"], e["e2_energy"]) for e in sop] == [("BD", "BD*", 12.34), ("LP", "RY", 0.91)]
    assert sop[0]["donor_nbo_num"] is None and sop[0]["acceptor_nbo_num"] is None
    assert sop[0]["donor_atoms"] == "H  1- H  2" and sop[0]["acceptor_atoms"] == "H  3- H  4"


def test_second_order_columns_emitted_for_labelled_atoms(tmp_path):
    reader = _parse(_fake_nbo_file(tmp_path, NBO7_TABLE))
    flat = _collect_selected_nbo_results(reader.results, {"H1": 1, "Ni": 2, "H2": 3})
    sop_cols = sorted(c for c in flat if c.startswith("second_order_"))
    # HOMO is NBO 87 (highest nbo_num with occupancy >= 0.1): 46 -> hm41, 88 -> lp1, 128 -> lp41, 87 -> homo
    assert "second_order_hm41_H1_Ni_to_lp1_Ni_H2_e2_energy" in sop_cols
    assert flat["second_order_hm41_H1_Ni_to_lp1_Ni_H2_e2_energy"] == 29.34
    assert flat["second_order_hm41_H1_Ni_to_lp1_Ni_H2_energy_diff"] == 0.38
    assert flat["second_order_hm41_H1_Ni_to_lp1_Ni_H2_fock_element"] == 0.095
    assert flat["second_order_hm40_H1_H2_to_homo_Ni_e2_energy"] == 128.37   # LV acceptor, absent from the summary
    assert flat["second_order_hm58_Ni_to_lp41_H1_e2_energy"] == 0.88
    # the C20/H29 acceptor is not a labelled atom -> no column
    assert not any("C20" in c or "H29" in c for c in sop_cols)


@pytest.mark.skipif(not REAL_OUT.exists(), reason="needs DATA/")
def test_real_orca61_output_block0():
    sop = _parse(REAL_OUT).results["second_order_perturbation"]
    assert len(sop) == 400
    assert sop[0]["donor_nbo_num"] == 29 and sop[0]["acceptor_nbo_num"] == 128 and sop[0]["e2_energy"] == 0.88
    assert {e["donor_type"] for e in sop} == {"BD", "LP"}
    assert {e["acceptor_type"] for e in sop} == {"BD*", "RY"}
