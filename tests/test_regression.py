"""Slow regression tests: the spec-driven pipeline must reproduce the historical outputs.

Reference data (committed):
* ``atom_numbering_results.csv``                      numbering table of the LNiH2 library
* ``review/results/library_per_conformer_upstream_f6bc24c.csv``  DescriPyTor block, per conformer
* ``20260918_LNiH2_library_001_to_download_2026_03_24_descriptors.csv``  aggregated + filtered
* ``20260918_test_molecules_descriptors.csv``         aggregated + filtered, test molecules

Both aggregated references were regenerated on 2026-09-18 with the current DescriPyTor submodule
and the corrected IP/EA sign convention, so every numeric column is compared.

Run with ``pytest -m slow`` (several minutes). Skipped automatically when DATA/ is absent.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from desc_utils.aggregate import aggregate_conformers, apply_filter
from desc_utils.extract import run_extraction
from desc_utils.extract.descripytor import extract_descripytor
from desc_utils.numbering import build_numbering_table, read_numbering_table
from desc_utils.spec import load_spec

pytestmark = pytest.mark.slow
LABELS = ["H1", "Ni", "H2", "N1", "C1", "C2", "N2"]
TOL = 1e-8


@pytest.fixture(scope="module")
def lnih2(repo_root):
    return load_spec(repo_root / "specs" / "lnih2_pyox.yaml")


@pytest.fixture(scope="module")
def numbering_ref(repo_root):
    return pd.read_csv(repo_root / "atom_numbering_results.csv")


def _assert_close(ref: pd.DataFrame, new: pd.DataFrame, cols, tol=TOL):
    a, b = ref[cols].astype(float), new[cols].astype(float)
    assert int((a.isna() != b.isna()).sum().sum()) == 0, "NaN pattern differs"
    diff = (a - b).abs().max()
    assert diff.max() <= tol, f"max |diff| {diff.max():.3e} in {diff.idxmax()}"


def test_smarts_numbering_reproduces_table(lnih2, numbering_ref, repo_root):
    df = build_numbering_table(lnih2, repo_root, progress=False).set_index("path")
    ref = numbering_ref.set_index("path")
    assert set(df.index) == set(ref.index)
    df = df.reindex(ref.index)
    assert (df["numbering_status"] == "ok").all()
    for c in LABELS:
        assert (df[c].astype(int) == ref[c].astype(int)).all(), c
    for c in ["smiles", "inchi", "ligand_type", "ligand_class"]:
        assert (df[c].astype(str) == ref[c].astype(str)).all(), c
    assert (df["SCF_energy"] - ref["SCF_energy"]).abs().max() < 1e-9   # CSV round-trip of the float


def test_descripytor_block_reproduces_reference(lnih2, numbering_ref, repo_root):
    ref = pd.read_csv(repo_root / "review/results/library_per_conformer_upstream_f6bc24c.csv").set_index("feather_dir")
    out = extract_descripytor(numbering_ref, lnih2.descripytor, repo_root, progress=False).set_index("path").reindex(ref.index)
    cols = [c for c in ref.columns if c != "n_loaded"]
    assert not [c for c in cols if c not in out.columns]
    _assert_close(ref, out, cols)


def test_full_pipeline_reproduces_aggregated_csv(lnih2, repo_root):
    ref = pd.read_csv(repo_root / "20260918_LNiH2_library_001_to_download_2026_03_24_descriptors.csv").set_index("inchi")
    num = read_numbering_table(repo_root / "atom_numbering_results.csv", lnih2)
    per_conf = run_extraction(lnih2, num, repo_root, progress=False)
    agg = aggregate_conformers(per_conf, lnih2)
    new = agg.set_index("inchi").reindex(ref.index)
    numeric = [c for c in ref.columns if pd.api.types.is_numeric_dtype(ref[c])]
    missing = [c for c in numeric if c not in new.columns]
    assert not missing, f"{len(missing)} reference columns missing, e.g. {missing[:5]}"
    _assert_close(ref, new, numeric)
    filtered, report = apply_filter(agg, lnih2)
    assert report["final_numeric_count"] > 300
    assert {"inchi", "inchi_folder", "smiles", "ligand_class"} <= set(filtered.columns)


def test_test_molecules_pipeline(repo_root):
    spec = load_spec(repo_root / "specs" / "test_molecules_at1_at2.yaml")
    ref = pd.read_csv(repo_root / "20260918_test_molecules_descriptors.csv").set_index("inchi")
    num = build_numbering_table(spec, repo_root, progress=False)
    assert (num[["at1", "at2"]].astype(int).values == [1, 2]).all()
    per_conf = run_extraction(spec, num, repo_root, progress=False)
    agg = aggregate_conformers(per_conf, spec)
    new = agg.set_index("inchi").reindex(ref.index)
    numeric = [c for c in ref.columns if pd.api.types.is_numeric_dtype(ref[c])]
    missing = [c for c in numeric if c not in new.columns]
    assert not missing, f"{len(missing)} reference columns missing, e.g. {missing[:5]}"
    _assert_close(ref, new, numeric)
    assert (new["Vertical_Ionization_Potential_boltz"] > 0).all()   # IP = E(cation) - E(neutral)
