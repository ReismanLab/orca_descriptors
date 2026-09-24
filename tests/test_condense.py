"""Condensed per-conformer file (``desc_utils.condense``): producer, checker, consumer adapters and
the round trip raw ORCA files -> orca_condensed.json -> identical in-house descriptor columns."""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from desc_utils import condense as c
from desc_utils import data_extraction as de
from desc_utils.numbering.io import scf_energy

REPO_ROOT = Path(__file__).resolve().parents[1]
LIB = REPO_ROOT / "DATA" / "LNiH2_library_001_to_download_2026_03_24"
FIXTURES = [LIB / "AATATIAROAAKIN-SNVBAGLBSA-N" / f"conf_{i}" for i in (1, 2)] + \
           [LIB / "CTDSXZLFAXPRHX-MRVPVSSYSA-N" / f"conf_{i}" for i in (1, 2, 3, 4)]
HAVE_FIXTURES = all((d / "calcParam.out").is_file() and (d / "singlepoint.json").is_file() for d in FIXTURES)
needs_fixtures = pytest.mark.skipif(not HAVE_FIXTURES, reason="needs the tracked fixture conformers under DATA/")
LABELS = ["H1", "Ni", "H2", "N1", "C1", "C2", "N2"]
TOL = 1e-8


@pytest.fixture(scope="module")
def condensed_tree(tmp_path_factory):
    """Copies of the fixtures holding only optimization.xyz + orca_condensed.json (what rsync pulls)."""
    if not HAVE_FIXTURES:
        pytest.skip("needs the tracked fixture conformers under DATA/")
    root = tmp_path_factory.mktemp("condensed")
    pairs = []
    for raw in FIXTURES:
        target = root / raw.parent.name / raw.name
        target.mkdir(parents=True)
        shutil.copy(raw / "optimization.xyz", target / "optimization.xyz")
        c.write_condensed(target, c.build_condensed(raw))
        pairs.append((raw, target))
    return pairs


@pytest.fixture(scope="module")
def numbering():
    return pd.read_csv(REPO_ROOT / "tests" / "reference" / "atom_numbering_results.csv")


def _rows(numbering, dirs):
    keys = [d.relative_to(REPO_ROOT).as_posix() + "/" for d in dirs]
    sub = numbering[numbering["path"].isin(keys)].set_index("path").loc[keys].reset_index()
    assert len(sub) == len(dirs)
    return sub


# --------------------------------------------------------------------------- pure functions

def test_pack_unpack_roundtrip():
    rows = [{"nbo_num": 1, "atoms": [1, 2], "energy": -0.5}, {"nbo_num": 2, "atoms": [3], "energy": None}]
    packed = c.pack_table(rows)
    assert packed["columns"] == ["nbo_num", "atoms", "energy"] and c.unpack_table(packed) == rows
    assert c.pack_table([]) == [] and c.pack_table({}) == {} and c.unpack_table({"atoms": {}}) == {"atoms": {}}


def test_nbo_results_spin_filter_and_levels():
    rows = [{"nbo_num": 1, "spin": "alpha"}, {"nbo_num": 1, "spin": "beta"}]
    block = {"index": 0, "level": "full", "results": {"spin": "both", "natural_bond_orbitals": c.pack_table(rows),
                                                      "natural_population_analysis": {"atoms": {}, "totals": {}}}}
    cond = {"nbo": {"blocks": [block]}}
    alpha = c.nbo_results(cond, 0, spin="alpha")
    assert alpha["spin"] == "alpha" and [r["spin"] for r in alpha["natural_bond_orbitals"]] == ["alpha"]
    assert [r["spin"] for r in c.nbo_results(cond, 0, spin="beta")["natural_bond_orbitals"]] == ["beta"]
    assert len(c.nbo_results(cond, 0, spin="both")["natural_bond_orbitals"]) == 2
    assert cond["nbo"]["blocks"][0]["results"]["spin"] == "both"  # the stored dict is not mutated
    block["level"] = "npa"
    with pytest.raises(ValueError, match="npa"):
        c.nbo_results(cond, 0)
    assert c.nbo_results(cond, 0, require_full=False)["spin"] == "alpha"
    with pytest.raises(ValueError):
        c.nbo_results(cond, 0, spin="gamma", require_full=False)


def test_views_of_missing_state_are_empty():
    cond = {"states": {"singlepoint": {"scf_energy_eh": -1.0, "charges": {}, "mo": None}}, "molecule": {"atoms": []}}
    assert c.property_json_view(cond, "nmr") == {} and c.gbw_view(cond, "singlepoint") == {}
    assert c.property_json_view(cond, "singlepoint")["Geometries"][0]["DFT_Energy"]["finalEn"] == -1.0
    assert c.scf_energy(cond) == -1.0 and c.scf_energy(cond, "nmr") is None


# --------------------------------------------------------------------------- producer on the fixtures

@needs_fixtures
def test_build_schema_and_size(condensed_tree):
    raw, cond_dir = condensed_tree[0]
    path = cond_dir / c.CONDENSED_FILENAME
    assert c.check_condensed(path) == []
    assert path.stat().st_size < 600_000
    head = path.read_bytes()[:8192].decode("utf-8")
    assert '"molecule"' in head and '"atoms"' in head  # DescriPyTor folder sniffer
    data = c.load_condensed(path)
    assert data["schema_version"] == 1 and data["molecule"]["n_atoms"] == 37
    assert data["molecule"]["charge"] == 0 and data["molecule"]["multiplicity"] == 1
    vib = data["vibrations"]
    assert len(vib["modes"]) == 3 * 37 - 6 and vib["n_dropped_zero_modes"] == 6 and vib["n_imaginary"] == 1
    assert vib["modes"][0]["frequency_cm-1"] == -16.71 and vib["modes"][0]["ir_intensity_km_per_mol"] is None
    assert vib["modes"][1]["ir_intensity_km_per_mol"] == 0.73 and len(vib["modes"][1]["displacements"]) == 37
    npa = data["nbo"]["blocks"][0]["results"]["natural_population_analysis"]["atoms"]
    assert data["charges"]["nbo"][1] == npa["2"]["charge"] and data["charges"]["cm5"] == data["charges"]["hirshfeld"]
    assert data["properties"]["dipole"]["total"] == pytest.approx(2.9809792900395555 * c.AU_TO_DEBYE)
    assert data["properties"]["dipole"]["units"] == "debye" and data["properties"]["polarizability"]["iso"] > 0
    assert data["states"]["singlepoint"]["scf_energy_eh"] == -2497.275318481515
    assert data["states"]["oxidation"]["mo"] is None and len(data["states"]["singlepoint"]["mo"]["energies_eh"]) == 736
    assert data["states"]["oxidation"]["multiplicity"] == 2 and data["states"]["oxidation"]["charge"] == 1
    assert data["states"]["nmr"]["nmr"]["siso_ppm"][1] < -1000  # Ni shielding
    assert [b["level"] for b in data["nbo"]["blocks"]] == ["full", "npa", "npa"]
    assert [b["closed_shell"] for b in data["nbo"]["blocks"]] == [True, False, False]
    assert data["producer"]["orca_version"] == "6.1.0" and data["producer"]["sources"]["calcParam.out"]["found"]


@needs_fixtures
def test_check_flags_inconsistencies(condensed_tree):
    _, cond_dir = condensed_tree[0]
    data = c.load_condensed(cond_dir / c.CONDENSED_FILENAME)
    broken = dict(data, charges=dict(data["charges"], nbo=data["charges"]["nbo"][:-1]))
    p = c.write_condensed(cond_dir, broken, name="broken.json")
    assert any("charges.nbo" in msg for msg in c.check_condensed(p))
    p.unlink()


@needs_fixtures
def test_cli_check_and_skip(condensed_tree):
    _, cond_dir = condensed_tree[0]
    r = subprocess.run([sys.executable, "-m", "desc_utils.condense", "--conf-dir", str(cond_dir), "--check"],
                       cwd=REPO_ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    # an existing file is skipped without --overwrite (the copy has no raw files, a rebuild would fail)
    r = subprocess.run([sys.executable, "-m", "desc_utils.condense", "--conf-dir", str(cond_dir)],
                       cwd=REPO_ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and "skipped" in r.stdout, r.stdout + r.stderr
    r = subprocess.run([sys.executable, "-m", "desc_utils.condense", "--conf-dir", str(cond_dir), "--overwrite"],
                       cwd=REPO_ROOT, capture_output=True, text=True)
    assert r.returncode == 1 and "failed" in r.stdout


# --------------------------------------------------------------------------- round trip

def _assert_same(a: pd.DataFrame, b: pd.DataFrame, family: str):
    assert list(a.columns) == list(b.columns), family
    for col in a.columns:
        x, y = a[col], b[col]
        if pd.api.types.is_numeric_dtype(x) or pd.api.types.is_numeric_dtype(y):
            x, y = pd.to_numeric(x), pd.to_numeric(y)
            assert (x.isna() == y.isna()).all(), (family, col)
            if x.notna().any():
                assert np.nanmax(np.abs(x.astype(float).values - y.astype(float).values)) <= TOL, (family, col)
        else:
            assert x.fillna("<na>").tolist() == y.fillna("<na>").tolist(), (family, col)


@needs_fixtures
def test_roundtrip_inhouse_orca_columns(condensed_tree, numbering):
    raw_dirs = [r for r, _ in condensed_tree]
    cond_dirs = [t for _, t in condensed_tree]
    df_raw = _rows(numbering, raw_dirs)
    df_cond = df_raw.copy()
    df_raw["path"] = [str(d) for d in raw_dirs]
    df_cond["path"] = [str(d) for d in cond_dirs]

    def run(df, condensed_file):
        # condensed_file=None forces the raw ORCA files even when the fixture folder holds a condensed file
        charges = pd.DataFrame.from_dict(de.get_all_charges(df, LABELS, condensed_file=condensed_file), orient="index")
        nbo = de.get_all_nbo_properties(df, LABELS, condensed_file=condensed_file)
        redox = pd.DataFrame.from_dict(
            de.get_redox_descriptors(df, LABELS, descriptorPrefixes=["nbo_charge", "cm5"],
                                     nbo_block_map={"singlepoint": 1, "oxidation": 2, "reduction": 3},
                                     condensed_file=condensed_file),
            orient="index")
        return {"charges": charges, "nbo": nbo, "redox": redox}

    raw, cond = run(df_raw, None), run(df_cond, c.CONDENSED_FILENAME)
    assert "nbo_extraction_error" not in cond["nbo"].columns
    assert cond["charges"]["HOMO_energy"].notna().all() and cond["redox"]["Vertical_Ionization_Potential"].notna().all()
    assert cond["nbo"].filter(like="second_order_").shape[1] > 0
    for family in raw:
        _assert_same(raw[family].drop(columns=["path"]), cond[family].drop(columns=["path"]), family)


@needs_fixtures
def test_reader_source_and_scf_energy(condensed_tree):
    raw, cond_dir = condensed_tree[0]
    reader = de.od.OrcaReader(str(cond_dir))
    reader.parse_property_json()
    assert reader.source == "condensed" and reader.propertyjson["Geometries"][0]["DFT_Energy"]["finalEn"] == -2497.275318481515
    reader = de.od.OrcaReader(str(raw), condensed_file=None)
    reader.parse_property_json()
    assert reader.source == "raw"
    forced_raw = de.od.OrcaReader(str(cond_dir), condensed_file=None)
    forced_raw.parse_property_json()
    assert forced_raw.source == "raw" and forced_raw.propertyjson == {}
    assert scf_energy(cond_dir) == scf_energy(raw, condensed_file=None) == -2497.275318481515


# --------------------------------------------------------------------------- DescriPyTor on the same file

@pytest.mark.slow
@needs_fixtures
def test_descripytor_reads_condensed_json(condensed_tree, numbering):
    from desc_utils.extract.descripytor import _import_molecules, _quiet_cwd, extract_one, label_columns
    from desc_utils.spec import load_spec

    spec = load_spec(REPO_ROOT / "specs" / "lnih2_pyox.yaml")
    raw, cond_dir = condensed_tree[0]
    Molecules = _import_molecules(REPO_ROOT)
    with _quiet_cwd():
        mols = Molecules(molecules_dir_name=str(cond_dir))
        assert len(mols.molecules) == 1
        row = _rows(numbering, [raw]).iloc[0]
        vals, notes = extract_one(mols, spec.descripytor, row)
    assert set(label_columns(spec.descripytor)) <= set(vals), notes
    assert np.isfinite(vals["vib_amp_Ni-H1"]) and vals["vib_amp_Ni-H1"] != 0.0
    data = c.load_condensed(cond_dir / c.CONDENSED_FILENAME)
    assert vals["charge_NBO_Ni"] == pytest.approx(data["charges"]["nbo"][int(row["Ni"]) - 1], abs=1e-9)
    assert vals["dipole"] == pytest.approx(data["properties"]["dipole"]["total"], abs=1e-6)
