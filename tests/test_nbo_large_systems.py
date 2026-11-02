"""NBO 7 output of large systems and how an NBO block ends (``NBOProps``, ``condense``).

NBO 7 writes atom labels as Fortran (a2,i3) and NBO / NAO numbers as i4:
  * atoms 100-999 abut the element ("H100", "C 64- H125");
  * orbitals 1000-9999 fill the i4 field, so NAO rows, hybrid-detail entries and NHO rows start in
    column 1 ("1000. (0.00071) RY ( 3) C 54"); the NAO and detail parsers used to require leading
    whitespace and dropped them (and an entry before 1000 swallowed the text of the ones after it);
  * past the field width Fortran prints asterisks ("C***" from atom 1000, "****." from orbital 10000).
The standard NBO 7 executables stop at 999 atoms (maxatm.fh) and analyse nothing beyond; that run
must be reported, not parsed as an empty block (likewise its limits on basis functions, shells and
exponents: 5000 / 2000 / 5000 in the manual). ORCA 6.1.0 and 6.1.1 close a block that has an
analysis with "*** returned from  NBO  program ***"; a block closed only by NBO's own "Stopping NBO..."
is accepted too, and a block never borrows the end line of the next one.
"""
from pathlib import Path

import pytest

from desc_utils import condense
from desc_utils import orca_descriptors as od
from desc_utils.data_extraction import _collect_selected_nbo_results, _extract_atom_numbers_from_text
from desc_utils.orca_parse import _fill_numbering, nbo_atom_numbers

START = "Now starting NBO....\n"
RETURNED = "\n                                *** returned from  NBO  program ***\n"
STOPPING = "\n NBO analysis completed in 1.00 CPU seconds (1 wall second)\nStopping NBO...Storing NBOs: singlepoint.nbo\n"

# NAO numbers past 999 start in column 1
NAO_1000 = """
 NATURAL POPULATIONS:  Natural atomic orbital occupancies

  NAO Atom No lang   Type(AO)    Occupancy      Energy
 -------------------------------------------------------
 998    H 99  s      Val( 1s)     0.77207       0.10539
 999    H 99  s      Ryd( 2s)     0.00112       0.42177

1000    H100  s      Val( 1s)     0.76039       0.13058
1001    H100  s      Ryd( 2s)     0.00206       0.52569
1002    H100  px     Ryd( 2p)     0.00028       1.95137

"""
# atoms 1..3 where atom 3 overflowed ("C***") and one NAO number overflowed ("****")
NAO_OVERFLOW = """
 NATURAL POPULATIONS:  Natural atomic orbital occupancies

  NAO Atom No lang   Type(AO)    Occupancy      Energy
 -------------------------------------------------------
   1    H  1  s      Val( 1s)     0.80000       0.10000
   2    H  1  s      Ryd( 2s)     0.00100       0.50000

   3    H  2  s      Val( 1s)     0.81000       0.10000
****    H  2  s      Ryd( 2s)     0.00100       0.50000

   5    C***  s      Cor( 1s)     1.99990     -10.10000
   6    C***  s      Val( 2s)     1.00000      -0.20000

"""
NPA_OVERFLOW = """
 Summary of Natural Population Analysis:

                                     Natural Population
             Natural    -----------------------------------------------
    Atom No  Charge        Core      Valence    Rydberg      Total
 --------------------------------------------------------------------
      H  1    0.19900      0.00000     0.80000    0.00100     0.80100
      H  2    0.18900      0.00000     0.81000    0.00100     0.81100
      C***   -0.38800      1.99990     2.38810    0.00000     4.38800
 ====================================================================
   * Total *  0.00000      1.99990     3.99810    0.00200     6.00000

 Natural Electron Configuration
"""
NPA = """
 Summary of Natural Population Analysis:

                                     Natural Population
             Natural    -----------------------------------------------
    Atom No  Charge        Core      Valence    Rydberg      Total
 --------------------------------------------------------------------
      H  1    0.00000      0.00000     1.00000    0.00000     1.00000
      H  2    0.00000      0.00000     1.00000    0.00000     1.00000
 ====================================================================
   * Total *  0.00000      0.00000     2.00000    0.00000     2.00000

 Natural Electron Configuration
"""
DETAILS = """
     (Occupancy)   Bond orbital / Coefficients / Hybrids
 -------------------------------------------------------------------------------
 ---------------- Lewis --------------------------------------------------------
 998. (1.98000) BD ( 1) C 54- H100
               ( 60.00%)   0.7746* C 54 s( 25.00%)p 3.00( 75.00%)
                                         0.0000  0.5000  0.0000
               ( 40.00%)   0.6325* H100 s( 99.90%)p 0.00(  0.10%)
                                         0.9995  0.0000
 ---------------- non-Lewis ----------------------------------------------------
 999. (0.02000) BD*( 1) C 54- H100
               ( 40.00%)   0.6325* C 54 s( 25.00%)p 3.00( 75.00%)
                                         0.0000  0.5000  0.0000
               ( 60.00%)  -0.7746* H100 s( 99.90%)p 0.00(  0.10%)
                                         0.9995  0.0000
1000. (0.00071) RY ( 3) C 54            s( 53.18%)p 0.86( 45.65%)d 0.02(  1.17%)
                                         0.0000  0.1000  0.9000
1001. (0.00050) BD*( 1) C***- H 99
               ( 50.00%)   0.7071* C*** s( 25.00%)p 3.00( 75.00%)
                                         0.0000  0.5000  0.0000
               ( 50.00%)  -0.7071* H 99 s( 99.90%)p 0.00(  0.10%)
                                         0.9995  0.0000
****. (0.00001) RY ( 1) H100            s( 90.00%)p 0.11( 10.00%)
                                         0.0000  1.0000

"""
DIRECTIONALITY = """
 NHO DIRECTIONALITY AND BOND BENDING (deviation from line of nuclear centers at
         [Thresholds for printing:  angular deviation  > 1.0 degree]
                                    hybrid p-character > 25.0%
                                    orbital occupancy  > 1.00e]

                      Line of Centers        Hybrid 1              Hybrid 2
                      ---------------  -------------------   ------------------
           NBO           Theta   Phi    Theta   Phi    Dev    Theta   Phi    Dev
 ===============================================================================
 998. BD ( 1) C 54- H100   97.1   53.4    95.0   50.0   3.5     --     --    --
1000. BD ( 1) C 54- H 99   80.0   10.0    78.0   12.0   2.5     --     --    --

"""
SECOND_ORDER = """
 SECOND ORDER PERTURBATION THEORY ANALYSIS OF FOCK MATRIX IN NBO BASIS

     Threshold for printing:   0.50 kcal/mol
                                                          E(2) E(NL)-E(L) F(L,NL)
      Donor (L) NBO              Acceptor (NL) NBO      kcal/mol   a.u.    a.u.
 ===============================================================================

 within unit  1
  998. BD ( 1) C 54- H100     1001. BD*( 1) C***- H 99     1.00    0.50   0.020
  998. BD ( 1) C 54- H100     ****. RY ( 1) H100           0.60    1.50   0.030

"""
SUMMARY = """

 NATURAL BOND ORBITALS (Summary):

                                                     Principal Delocalizations
           NBO                 Occupancy    Energy   (geminal,vicinal,remote)
 ===============================================================================
 Molecular unit  1  (CH4)
 ------ Lewis --------------------------------------
  998. BD ( 1) C 54- H100       1.98000    -0.60000
 ------ non-Lewis ----------------------------------
  999. BD*( 1) C 54- H100       0.02000     0.50000
 1000. RY ( 3) C 54             0.00071     0.80000
 1001. BD*( 1) C***- H 99       0.00050     0.40000
 ****. RY ( 1) H100             0.00001     2.00000

 $CHOOSE
"""
LARGE = NAO_1000 + NPA + DETAILS + DIRECTIONALITY + SECOND_ORDER + SUMMARY

BANNER = """
 *********************************** NBO 7.0 ***********************************
             N A T U R A L   A T O M I C   O R B I T A L   A N D
          N A T U R A L   B O N D   O R B I T A L   A N A L Y S I S
 Job title: ORCA Job: singlepoint
"""
ATOM_LIMIT = BANNER + """
 This calculation is attempting to analyze 1008 atoms, although NBO is currently
 configured to handle up to only 999 atoms.

 If you have the NBO source distribution, you can reconfigure NBO by editing
 maxatm.fh in nbo7/src/nbo/include and rebuilding the executables.
"""


def _write(tmp_path: Path, text: str) -> Path:
    out = tmp_path / "calcParam.out"
    out.write_text(text)
    return out


def _parse(path: Path, block_index: int = 0):
    reader = od.NBOProps(path, block_index=block_index)
    reader.get_NPA()
    reader.get_NAO()
    reader.get_NBO_summary()
    reader.get_orbital_details()
    reader.get_NHO_directionality()
    reader.get_second_order_perturbation()
    return reader.results


def test_atom_label_helpers():
    assert nbo_atom_numbers("C 64- H125") == [64, 125]
    assert nbo_atom_numbers("Ni  2- H100") == [2, 100]
    assert nbo_atom_numbers("C***- H 99") == [None, 99]
    assert _extract_atom_numbers_from_text("C***- H 99") == [None, 99]
    assert _fill_numbering([998, 999, 1000, None]) == [998, 999, 1000, 1001]
    assert _fill_numbering([1, None, 3]) == [1, 2, 3]
    assert _fill_numbering([1, None, 5]) == [1, None, 5]          # the gap does not fit one row
    assert _fill_numbering([7, None], trailing=False) == [7, None]


def test_nao_rows_past_999_start_in_column_1(tmp_path):
    nao = _parse(_write(tmp_path, START + LARGE + RETURNED))["natural_atomic_orbitals"]
    assert [r["orbital_num"] for r in nao] == [998, 999, 1000, 1001, 1002]
    assert [r["atom_num"] for r in nao] == [99, 99, 100, 100, 100]
    assert nao[2]["orbital_type"] == "Val" and nao[2]["occupancy"] == 0.76039


def test_overflowed_numbers_are_recovered_from_table_order(tmp_path):
    res = _parse(_write(tmp_path, START + NAO_OVERFLOW + NPA_OVERFLOW + RETURNED))
    nao = res["natural_atomic_orbitals"]
    assert [r["orbital_num"] for r in nao] == [1, 2, 3, 4, 5, 6]
    assert [r["atom_num"] for r in nao] == [1, 1, 2, 2, 3, 3]
    npa = res["natural_population_analysis"]["atoms"]
    assert sorted(npa) == [1, 2, 3] and npa[3]["atom"] == "C" and npa[3]["charge"] == -0.388


def test_detail_entries_past_999_are_separate(tmp_path):
    det = _parse(_write(tmp_path, START + LARGE + RETURNED))["bond_orbital_details"]
    assert [d["bond_num"] for d in det] == [998, 999, 1000, 1001, 1002]
    # 999 keeps its own two hybrids and no longer swallows the entries after it
    assert [c["atom_num"] for c in det[1]["contributions"]] == [54, 100]
    assert det[2]["bond_type"] == "RY" and det[2]["atoms"] == [54]
    assert det[3]["atoms"] == [None, 99]
    assert [c["atom_num"] for c in det[3]["contributions"]] == [None, 99]
    assert det[4]["atoms"] == [100] and det[4]["occupancy"] == 0.00001


def test_summary_nho_and_second_order_past_999(tmp_path):
    res = _parse(_write(tmp_path, START + LARGE + RETURNED))
    summary = res["natural_bond_orbitals"]
    assert [r["nbo_num"] for r in summary] == [998, 999, 1000, 1001, 1002]   # "****." filled: single unit
    assert summary[3]["atoms"] == [None, 99]
    nho = res["nho_directionality"]
    assert [(r["nbo_num"], r["atoms"]) for r in nho] == [(998, [54, 100]), (1000, [54, 99])]
    sop = res["second_order_perturbation"]
    assert [(r["donor_nbo_num"], r["acceptor_nbo_num"], r["acceptor_atoms"]) for r in sop] == [
        (998, 1001, "C***- H 99"), (998, None, "H100")]


def test_rows_with_an_unknown_atom_never_count_as_selected_atoms(tmp_path):
    res = _parse(_write(tmp_path, START + LARGE + RETURNED))
    flat = _collect_selected_nbo_results(res, {"C1": 54, "H1": 99, "H2": 100})
    assert flat["nbo_lp1_C1_H2_type"] == "BD*"                 # 999 (HOMO reference: 998), all atoms known
    assert not any("C1_H1" in k or "H1_" in k for k in flat if k.startswith(("nbo_", "bond_orbital")))
    assert not any(k.startswith("second_order_") and "to_" in k and "H1" in k.split("to_")[1] for k in flat)


@pytest.mark.parametrize("end", [RETURNED, STOPPING, STOPPING + RETURNED], ids=["returned", "stopping", "both"])
def test_block_end_marker_of_either_orca_version(tmp_path, end):
    reference = _parse(_write(tmp_path, START + LARGE + RETURNED))
    reader = od.NBOProps(_write(tmp_path, START + LARGE + end + "\n CHELPG CHARGES GENERATION\n"))
    assert len(reader.nbo_blocks) == 1 and reader.error is None
    assert _parse(reader.path) == {**reference, "source_file": str(reader.path)}


def test_failed_block_does_not_borrow_the_next_blocks_analysis(tmp_path):
    # block 0: NBO ran without an analysis (ORCA 6.1.1: "Stopping NBO..." and no "returned" line)
    text = START + BANNER + "Stopping NBO...\n" + START + NPA + RETURNED
    reader = od.NBOProps(_write(tmp_path, text), block_index=0)
    assert len(reader.nbo_blocks) == 2
    assert "without an analysis" in reader.error and "ulimit -s unlimited" in reader.error
    assert "Summary of Natural Population Analysis" not in reader.nbo_section
    assert od.NBOProps(reader.path, block_index=1).error is None
    assert len(_parse(reader.path, block_index=1)["natural_population_analysis"]["atoms"]) == 2


def test_aborted_run_without_end_line_is_left_out(tmp_path):
    reader = od.NBOProps(_write(tmp_path, START + NPA + RETURNED + START + BANNER))
    assert len(reader.nbo_blocks) == 1


def test_nbo_atom_limit_is_reported(tmp_path, capsys):
    out = _write(tmp_path, START + ATOM_LIMIT + "Stopping NBO...-------\n")
    reader = od.NBOProps(out)
    assert "1008 atoms" in reader.error and "at most 999" in reader.error
    assert "at most 999" in capsys.readouterr().out
    with pytest.raises(condense.CondenseError, match="at most 999"):
        condense._nbo_blocks(out, out.read_text(), (0,))


# The other fixed dimensions of NBO 7, as its format statements print them
DIM_LIMITS = {
    "basis functions": " This calculation is attempting to analyze 6120 basis functions, although\n"
                       " NBO is currently configured to handle up to only 5000 functions.\n",
    "shells": " This calculation is attempting to analyze 2400 shells, although NBO is currently\n"
              " configured to handle up to only 2000 shells.\n",
    "exponents": " This calculation is attempting to analyze 5300 exponents, although NBO is\n"
                 " currently configured to handle up to only 5000 exponents.\n",
}


@pytest.mark.parametrize("what", DIM_LIMITS)
def test_other_nbo_dimension_limits_are_reported(tmp_path, what):
    reader = od.NBOProps(_write(tmp_path, START + BANNER + DIM_LIMITS[what] + "Stopping NBO...\n"))
    assert f" {what}, this NBO build handles at most" in reader.error


def test_failed_later_block_is_stored_as_failed(tmp_path):
    out = _write(tmp_path, START + NPA + RETURNED + START + BANNER + "Stopping NBO...\n")
    blocks = condense._nbo_blocks(out, out.read_text(), (0,))
    assert [b["level"] for b in blocks] == ["full", "failed"]
    assert "without an analysis" in blocks[1]["error"]
    cond = {"nbo": {"blocks": blocks}}
    assert len(condense.nbo_results(cond, 0)["natural_population_analysis"]["atoms"]) == 2
    with pytest.raises(ValueError, match="without an analysis"):
        condense.nbo_results(cond, 1, require_full=False)
