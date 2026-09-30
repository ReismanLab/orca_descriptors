import numpy as np
import pandas as pd
import pytest

from desc_utils.aggregate import aggregate_conformers, aggregate_frame, boltzmann_weights, molecule_folder
from desc_utils.spec import validate_spec

SPEC = validate_spec({
    "name": "t", "data": {"root": "DATA/x"}, "atoms": {"A": None, "B": None},
    "numbering": {"backend": "first_n"},
    "descriptors": {"descripytor": {"charges": ["A"]}},
    "aggregate": {"carry_first": ["smiles"]},
})


def test_boltzmann_weights_basic_and_nan():
    kB, T = 3.166811563e-6, 298.15
    e = pd.Series([-100.0, -100.0 + 0.001, np.nan])   # 1 mEh above -> exp(-1e-3/(kB T))
    w = boltzmann_weights(e, T, kB)
    assert np.isnan(w.iloc[2])
    assert w.iloc[:2].sum() == pytest.approx(1.0)
    assert w.iloc[1] / w.iloc[0] == pytest.approx(np.exp(-0.001 / (kB * T)))


def test_molecule_folder():
    assert molecule_folder("DATA/lib/ID/conf_12/") == "DATA/lib/ID"
    assert molecule_folder("DATA/lib/ID") == "DATA/lib/ID"


def test_aggregate_frame_stats():
    df = pd.DataFrame({
        "inchikey": ["m", "m", "n"], "SCF_energy": [-10.0, -10.0, -5.0],
        "x": [1.0, 3.0, 7.0], "y": [np.nan, 2.0, np.nan], "smiles": ["s1", "s1", "s2"],
    })
    agg = aggregate_frame(df, "inchikey", "SCF_energy", carry_first=["smiles"]).set_index("inchikey")
    assert agg.loc["m", "x_min"] == 1.0 and agg.loc["m", "x_max"] == 3.0
    assert agg.loc["m", "x_boltz"] == pytest.approx(2.0)       # equal energies -> plain mean
    assert agg.loc["m", "x_lowE"] == 1.0                        # first minimum wins (idxmin)
    assert agg.loc["m", "y_boltz"] == 2.0                       # NaN excluded from the weighted mean
    assert np.isnan(agg.loc["n", "y_boltz"])
    assert agg.loc["m", "n_conformers"] == 2 and agg.loc["m", "smiles"] == "s1"


def test_aggregate_conformers_excludes_label_columns_and_adds_folder():
    df = pd.DataFrame({
        "path": ["DATA/x/M1/conf_1", "DATA/x/M1/conf_2"], "inchikey": ["M1", "M1"],
        "SCF_energy": [-1.0, -1.0], "A": [1, 1], "B": [2, 2], "charge_NBO_A": [0.1, 0.3], "smiles": ["s", "s"],
    })
    agg = aggregate_conformers(df, SPEC)
    assert list(agg.columns[:3]) == ["inchikey_folder", "inchikey", "smiles"]
    assert agg.loc[0, "inchikey_folder"] == "DATA/x/M1"
    assert "A_min" not in agg.columns and "charge_NBO_A_boltz" in agg.columns
    assert agg.loc[0, "charge_NBO_A_boltz"] == pytest.approx(0.2)
