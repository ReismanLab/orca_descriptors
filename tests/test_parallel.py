"""``--jobs N``: the parallel numbering and extraction must give the serial tables, row for row."""
import pickle

import numpy as np
import pandas as pd
import pytest

from desc_utils.numbering.base import make_backend
from desc_utils.parallel import parallel_map, resolve_jobs
from desc_utils.spec import load_spec


def _square(x):
    return x * x


def test_resolve_jobs():
    assert resolve_jobs(None) == 1
    assert resolve_jobs(1) == 1
    assert resolve_jobs(3) == 3
    assert resolve_jobs(0) >= 1
    assert resolve_jobs(-1) == resolve_jobs(0)


def test_parallel_map_keeps_order_serial_and_parallel():
    items = list(range(7))
    assert parallel_map(_square, items, 1, progress=False) == [x * x for x in items]
    assert parallel_map(_square, items, 2, progress=False) == [x * x for x in items]
    assert parallel_map(_square, [], 2, progress=False) == []


def test_smarts_backend_survives_pickling(repo_root):
    """The backend travels to every worker; RDKit query molecules must round-trip."""
    spec = load_spec(repo_root / "specs" / "aryl_alkynes.yaml")
    backend = make_backend(spec.numbering, spec.labels)
    clone = pickle.loads(pickle.dumps(backend))
    from rdkit import Chem

    mol = Chem.AddHs(Chem.MolFromSmiles("C#Cc1ccccc1"))
    ctx = None
    assert clone.number(mol, ctx) == backend.number(mol, ctx)


def _same_table(a: pd.DataFrame, b: pd.DataFrame):
    assert list(a.columns) == list(b.columns)
    assert len(a) == len(b)
    for c in a.columns:
        if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
            x, y = a[c].astype(float).to_numpy(), b[c].astype(float).to_numpy()
            assert np.array_equal(np.isnan(x), np.isnan(y)), c
            assert np.allclose(x[~np.isnan(x)], y[~np.isnan(y)], rtol=0, atol=1e-12), c
        else:
            assert (a[c].astype(str) == b[c].astype(str)).all(), c


@pytest.mark.slow
def test_parallel_pipeline_matches_serial(repo_root):
    from desc_utils.extract import run_extraction
    from desc_utils.numbering import build_numbering_table

    spec = load_spec(repo_root / "specs" / "test_molecules_at1_at2.yaml")
    num_serial = build_numbering_table(spec, repo_root, progress=False)
    num_parallel = build_numbering_table(spec, repo_root, progress=False, jobs=2)
    _same_table(num_serial, num_parallel)

    serial = run_extraction(spec, num_serial, repo_root, progress=False)
    parallel = run_extraction(spec, num_serial, repo_root, progress=False, jobs=2)
    _same_table(serial, parallel)
