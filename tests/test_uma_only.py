"""Geometry-only descriptors from an xyz that ORCA never saw (UMA / ASE extended xyz): geometry file
pass-through, energy from the comment line with an explicit provenance column, no ORCA bookkeeping."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from desc_utils.numbering.io import HARTREE_IN, to_hartree, xyz_comment_energy
from desc_utils.spec import SpecError, load_spec, validate_spec

REPO = Path(__file__).resolve().parents[1]

# acetylene as UMA writes it: extended-xyz header, forces after the coordinates
UMA_XYZ = """4
Properties=species:S:1:pos:R:3:forces:R:3 charge=0 spin=1 energy=-2104.1093822195417 free_energy=-2104.2 stress="-0.01 0 0 0 0 0 0 0 0" pbc="F F F"
C       -0.59775945      -0.00000177      -0.00000196      -0.00035322      -0.00003532      -0.00003651
C        0.59775945       0.00000177       0.00000196       0.00035322       0.00003532       0.00003651
H       -1.66399745      -0.00000508      -0.00000527       0.00001234       0.00000123       0.00000127
H        1.66399745       0.00000508       0.00000527      -0.00001234      -0.00000123      -0.00000127
"""

UMA_SPEC = {
    "name": "uma_test",
    "data": {"root": "DATA/x", "geometry_file": "optimized_uma.xyz", "condensed_file": None},
    "atoms": {"at1": None, "at2": None},
    "numbering": {"backend": "first_n",
                  "energy": {"source": "xyz_comment", "key": "energy", "units": "eV", "column": "uma_energy_hartree"}},
    "descriptors": {"orca": {"xyz": {"atoms": ["at1", "at2"], "bonds": [["at1", "at2"]]}}},
}


def _library(root: Path, n_conf=2) -> Path:
    for i in range(1, n_conf + 1):
        d = root / "MOL" / f"conf_{i}"
        d.mkdir(parents=True)
        (d / "optimized_uma.xyz").write_text(UMA_XYZ.replace("-2104.1093822195417", f"-2104.{i}"))
    return root


# --------------------------------------------------------------------------- spec

def test_shipped_uma_spec_validates(repo_root):
    spec = load_spec(repo_root / "specs" / "test_molecules_uma_only.yaml")
    assert spec.numbering["energy"]["source"] == "xyz_comment"
    assert spec.aggregate["energy_column"] == spec.numbering["energy"]["column"] == "uma_energy_hartree"
    assert list(spec.orca) == ["xyz"] and spec.descripytor is None


def test_energy_source_defaults_to_orca_and_scf_energy():
    raw = copy.deepcopy(UMA_SPEC)
    del raw["numbering"]["energy"]
    spec = validate_spec(raw)
    assert spec.numbering["energy"]["source"] == "orca"
    assert spec.numbering["energy"]["column"] == spec.aggregate["energy_column"] == "SCF_energy"


def test_xyz_comment_needs_an_explicit_non_scf_column_name():
    raw = copy.deepcopy(UMA_SPEC)
    del raw["numbering"]["energy"]["column"]
    with pytest.raises(SpecError, match="column is required with source xyz_comment"):
        validate_spec(raw)
    raw["numbering"]["energy"]["column"] = "SCF_energy"
    with pytest.raises(SpecError, match="may not be 'SCF_energy'"):
        validate_spec(raw)
    raw["numbering"]["energy"].update(column="uma_energy_hartree", units="Ry", key="free energy")
    with pytest.raises(SpecError) as exc:
        validate_spec(raw)
    assert "units must be one of" in str(exc.value) and "key must be an extended-xyz key" in str(exc.value)
    raw["numbering"]["energy"]["source"] = "gaussian"
    with pytest.raises(SpecError, match="source must be one of"):
        validate_spec(raw)


def test_aggregate_energy_column_follows_numbering_unless_set():
    spec = validate_spec(UMA_SPEC)
    assert spec.aggregate["energy_column"] == "uma_energy_hartree"
    raw = copy.deepcopy(UMA_SPEC)
    raw["aggregate"] = {"energy_column": "SCF_energy"}      # explicit: e.g. the charges family supplies it
    assert validate_spec(raw).aggregate["energy_column"] == "SCF_energy"


# --------------------------------------------------------------------------- comment-line energy

def test_xyz_comment_energy_reads_key_and_converts(tmp_path):
    (tmp_path / "g.xyz").write_text(UMA_XYZ)
    e = xyz_comment_energy(tmp_path, "g.xyz")
    assert e == pytest.approx(-2104.1093822195417 / 27.211386245988)
    assert xyz_comment_energy(tmp_path, "g.xyz", key="free_energy") == pytest.approx(-2104.2 / HARTREE_IN["eV"])
    assert xyz_comment_energy(tmp_path, "g.xyz", units="hartree") == pytest.approx(-2104.1093822195417)
    with pytest.raises(ValueError, match="no 'enthalpy=<number>'"):
        xyz_comment_energy(tmp_path, "g.xyz", key="enthalpy")
    assert to_hartree(627.5094740631, "kcal/mol") == pytest.approx(1.0)
    with pytest.raises(ValueError, match="unknown energy units"):
        to_hartree(1.0, "Ry")


def test_xyz_comment_energy_handles_quotes_exponents_and_plain_xyz(tmp_path):
    (tmp_path / "q.xyz").write_text('1\nenergy="-1.5e2" free_energy=-3\nH 0 0 0\n')
    assert xyz_comment_energy(tmp_path, "q.xyz", units="hartree") == -150.0
    (tmp_path / "orca.xyz").write_text("1\nCoordinates from ORCA-job optimization E -77.3\nH 0 0 0\n")
    with pytest.raises(ValueError):
        xyz_comment_energy(tmp_path, "orca.xyz")


# --------------------------------------------------------------------------- morfeus reader

def test_properties_xyz_reads_the_named_geometry_and_caches_apart(tmp_path):
    import desc_utils.orca_descriptors as od

    (tmp_path / "optimized_uma.xyz").write_text(UMA_XYZ)
    (tmp_path / "xyz.json").write_text(json.dumps({"path": "stale", "SASA": 1.0}))   # ORCA-geometry cache
    P = od.PropertiesXYZ(str(tmp_path), geometry_file="optimized_uma.xyz")
    assert P.n_atoms == 4 and P.elements == ["C", "C", "H", "H"]
    assert P.coordinates[0] == pytest.approx([-0.59775945, -0.00000177, -0.00000196])   # forces ignored
    assert P.cache_file == "xyz_optimized_uma.json" and "SASA" not in P.results          # xyz.json not loaded
    assert od.PropertiesXYZ.cache_name() == "xyz.json"
    P.results["SASA"] = 2.0
    P.save_results_to_json()
    assert json.loads((tmp_path / "xyz_optimized_uma.json").read_text())["SASA"] == 2.0
    assert json.loads((tmp_path / "xyz.json").read_text())["SASA"] == 1.0
    with pytest.raises(FileNotFoundError):
        od.PropertiesXYZ(str(tmp_path))          # the default optimization.xyz is not there


def test_properties_xyz_rejects_truncated_file(tmp_path):
    import desc_utils.orca_descriptors as od

    (tmp_path / "g.xyz").write_text("\n".join(UMA_XYZ.splitlines()[:4]) + "\n")
    with pytest.raises(ValueError, match="announces 4 atoms, 2 read"):
        od.PropertiesXYZ(str(tmp_path), geometry_file="g.xyz")


# --------------------------------------------------------------------------- pipeline

def test_geometry_only_pipeline_on_uma_xyz(tmp_path):
    """number -> extract -> aggregate on a folder that holds nothing but the UMA xyz."""
    from desc_utils.aggregate import aggregate_conformers
    from desc_utils.extract import needs_orca_output, run_extraction
    from desc_utils.numbering import build_numbering_table

    root = _library(tmp_path / "lib")
    raw = copy.deepcopy(UMA_SPEC)
    raw["data"]["root"] = str(root)
    spec = validate_spec(raw)
    assert not needs_orca_output(spec)

    num = build_numbering_table(spec, repo_root=tmp_path)
    assert list(num["numbering_status"]) == ["ok", "ok"] and list(num["xyz_name"]) == ["optimized_uma"] * 2
    assert num["uma_energy_hartree"].tolist() == pytest.approx([-2104.1 / HARTREE_IN["eV"], -2104.2 / HARTREE_IN["eV"]])
    assert "SCF_energy" not in num.columns
    assert not any("energy" in str(n) for n in num["numbering_notes"])

    df = run_extraction(spec, num, repo_root=tmp_path, progress=False)
    assert "orca_job_status" not in df.columns and "n_conformer_folders" not in df.columns
    assert df["Bond_at1_at2"].tolist() == pytest.approx([1.1955189] * 2, abs=1e-6)
    assert "%Vbur_atom_at1_radius_3.5" in df.columns and "SASA" in df.columns
    assert (root / "MOL" / "conf_1" / "xyz_optimized_uma.json").is_file()
    assert not (root / "MOL" / "conf_1" / "xyz.json").exists()

    agg = aggregate_conformers(df, spec)
    assert len(agg) == 1 and agg.loc[0, "n_conformers"] == 2
    assert "all_conformers_completed" not in agg.columns
    assert agg.loc[0, "Bond_at1_at2_boltz"] == pytest.approx(1.1955189, abs=1e-6)
    assert np.isclose(agg.loc[0, "uma_energy_hartree_min"], -2104.2 / HARTREE_IN["eV"])


def test_orca_bookkeeping_kept_when_an_orca_family_is_requested():
    from desc_utils.extract import needs_orca_output

    raw = copy.deepcopy(UMA_SPEC)
    raw["descriptors"]["orca"]["charges"] = {"atoms": ["at1"]}
    spec = validate_spec(raw)
    assert needs_orca_output(spec) and needs_orca_output(spec, ["orca.charges"])
    assert not needs_orca_output(spec, ["orca.xyz"])


# --------------------------------------------------------------------------- pyramidalization connectivity

# Pyridine-type N bound to Ni as UMA places it: the Ni-N distance (2.22 A) lies between the reader's cutoff
# (covalent radii x 1.25 = 2.44 A) and morfeus' own (Pyykko x 1.2 = 2.17 A). Letting morfeus redo the
# neighbour search raised "2 neighbors. 3 expected." and aborted the whole extraction.
BORDERLINE_XYZ = """4
N with two C at 1.33 A and one Ni at 2.22 A, slightly pyramidal
N    0.000000   0.000000   0.000000
C    1.330000   0.000000   0.000000
C   -0.665000   1.151800   0.000000
Ni  -1.100000  -1.905300   0.300000
"""


def test_pyramidalization_uses_the_readers_neighbours(tmp_path):
    import numpy as np
    from morfeus.utils import get_connectivity_matrix
    import desc_utils.orca_descriptors as od

    (tmp_path / "g.xyz").write_text(BORDERLINE_XYZ)
    P = od.PropertiesXYZ(str(tmp_path), geometry_file="g.xyz")
    P.define_atom_list({"N1": 1})
    assert sorted(P.atom_connectivity[1]) == [2, 3, 4]
    morfeus_n = int(get_connectivity_matrix(np.array(P.coordinates), elements=P.elements,
                                            radii_type="pyykko", scale_factor=1.2)[0].sum())
    assert morfeus_n == 2                      # the disagreement that used to raise
    P.get_pyramidalizations()
    assert np.isfinite(P.results["Pyr_N1"]) and P.results["Pyr_N1"] > 0
    assert np.isfinite(P.results["Pyr_angle_N1"])


def test_pyramidalization_failure_leaves_nan_instead_of_raising(tmp_path, monkeypatch):
    import desc_utils.orca_descriptors as od

    (tmp_path / "g.xyz").write_text(BORDERLINE_XYZ)
    P = od.PropertiesXYZ(str(tmp_path), geometry_file="g.xyz")
    P.define_atom_list({"N1": 1})

    def boom(*a, **k):
        raise RuntimeError("morfeus said no")

    monkeypatch.setattr(od, "Pyramidalization", boom)
    P.get_pyramidalizations()                  # must not raise
    assert "Pyr_N1" not in P.results           # -> NaN column for this conformer in the table
