"""``desc_utils.orca_parse`` and ``desc_utils.condense`` run on the cluster with a bare Python: they
must stay standard-library only and Python 3.8 compatible, and the text parsers for the frequency
tables must agree with ORCA's own JSON."""
import ast
import subprocess
import sys
from pathlib import Path

import pytest

from desc_utils import condense as c
from desc_utils import orca_parse as op

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULES = ["desc_utils/orca_parse.py", "desc_utils/condense.py"]
FIXTURE = REPO_ROOT / "DATA" / "LNiH2_library_001_to_download_2026_03_24" / "AATATIAROAAKIN-SNVBAGLBSA-N" / "conf_1"
TEST_MOL = REPO_ROOT / "DATA" / "test_molecules" / "UHOVQNZJYSORNB-UHFFFAOYSA-N" / "conf_1"

SYNTHETIC_OUT = """
                         Program Version 6.1.0  -  RELEASE   -
-----------------------
VIBRATIONAL FREQUENCIES
-----------------------

Scaling factor for frequencies =  1.000000000  (already applied!)

     0:       0.00 cm**-1
     1:       0.00 cm**-1
     2:       0.00 cm**-1
     3:       0.00 cm**-1
     4:       0.00 cm**-1
     5:    -100.50 cm**-1  ***imaginary mode***


------------
NORMAL MODES
------------

These modes are the Cartesian displacements weighted by the diagonal matrix
M(i,i)=1/sqrt(m[i]) where m[i] is the mass of the displaced atom
Thus, these vectors are normalized but *not* orthogonal

                  0          1          2          3          4          5    
      0       0.000000   0.000000   0.000000   0.000000   0.000000   0.100000
      1       0.000000   0.000000   0.000000   0.000000   0.000000   0.200000
      2       0.000000   0.000000   0.000000   0.000000   0.000000   0.300000
      3       0.000000   0.000000   0.000000   0.000000   0.000000  -0.100000
      4       0.000000   0.000000   0.000000   0.000000   0.000000  -0.200000
      5       0.000000   0.000000   0.000000   0.000000   0.000000  -0.300000


-----------
IR SPECTRUM
-----------

 Mode   freq       eps      Int      T**2         TX        TY        TZ
       cm**-1   L/(mol*cm) km/mol    a.u.
----------------------------------------------------------------------------
  5:   -100.50   0.000145   12.50  0.001474  ( 0.000592 -0.003018  0.038266)
--------------------------

The first frequency considered to be a vibration is 5
 *********************************** NBO 7.0 ***********************************
"""


@pytest.mark.parametrize("rel", MODULES)
def test_python38_syntax(rel):
    ast.parse((REPO_ROOT / rel).read_text(), feature_version=(3, 8))


def test_imports_without_third_party_packages():
    code = (
        "import sys\n"
        "for m in ('numpy', 'pandas', 'morfeus', 'dbstep', 'rdkit', 'yaml', 'tqdm', 'scipy', 'pyarrow', 'openbabel'):\n"
        "    sys.modules[m] = None\n"
        "import desc_utils.orca_parse, desc_utils.condense\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_reexports_keep_legacy_names():
    from desc_utils import orca_descriptors as od

    assert od.OrcaReader is op.OrcaReader and od.NBOProps is op.NBOProps and od.RedoxSet is op.RedoxSet
    assert od.CONDENSED_FILENAME == c.CONDENSED_FILENAME == "orca_condensed.json"


def test_text_parsers_on_synthetic_output():
    freqs = op.parse_text_frequencies(SYNTHETIC_OUT)
    modes = op.parse_text_normal_modes(SYNTHETIC_OUT)
    assert freqs == [0.0, 0.0, 0.0, 0.0, 0.0, -100.5]
    assert len(modes) == 6 and modes[5] == [0.1, 0.2, 0.3, -0.1, -0.2, -0.3]
    assert op.drop_zero_modes(freqs, modes) == [5]
    assert op.ir_intensities(SYNTHETIC_OUT) == {5: 12.5}
    assert op.orca_version(SYNTHETIC_OUT) == "6.1.0" and op.nbo_program(SYNTHETIC_OUT) == "NBO 7.0"


def test_helpers():
    assert op._flat([[1.0], [2.0]]) == [1.0, 2.0] and op._flat([1.0, 2.0]) == [1.0, 2.0] and op._flat(None) is None
    assert op.SYMBOL_TO_Z["Ni"] == 28 and op.SYMBOL_TO_Z["H"] == 1 and len(op.ELEMENTS) == 118
    assert op.polar_anisotropy([1.0, 1.0, 1.0]) == 0.0 and op.polar_anisotropy(None) is None


@pytest.mark.skipif(not (FIXTURE / "calcParam.out").is_file(), reason="needs the tracked fixture conformer")
def test_fixture_frequency_tables():
    text = (FIXTURE / "calcParam.out").read_text()
    freqs, modes = op.parse_text_frequencies(text), op.parse_text_normal_modes(text)
    assert len(freqs) == 111 == len(modes) and all(len(m) == 111 for m in modes)
    keep = op.drop_zero_modes(freqs, modes)
    assert len(keep) == 105 and keep[0] == 6 and freqs[6] == -16.71
    ir = op.ir_intensities(text)
    assert 6 not in ir and ir[7] == 0.73 and len(ir) == 104
    assert op.orca_version(text) == "6.1.0" and op.nbo_program(text) == "NBO 7.0"
    f2, m2, src = op.frequencies_and_modes(FIXTURE, text)
    assert src == "calcParam.out" and f2 == freqs and m2 == modes


@pytest.mark.skipif(not (TEST_MOL / "frequency.property.json").is_file(), reason="needs DATA/test_molecules")
def test_json_and_text_modes_agree():
    freqs_j, modes_j, src = op.frequencies_and_modes(TEST_MOL)
    assert src == "frequency.property.json"
    text = (TEST_MOL / "calcParam.out").read_text()
    freqs_t, modes_t = op.parse_text_frequencies(text), op.parse_text_normal_modes(text)
    assert len(freqs_j) == len(freqs_t) == len(modes_j) == len(modes_t)
    assert max(abs(a - b) for a, b in zip(freqs_j, freqs_t)) < 0.01  # the text has two decimals
    assert max(abs(a - b) for mj, mt in zip(modes_j, modes_t) for a, b in zip(mj, mt)) < 1e-6
