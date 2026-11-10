"""Duplicate descriptor columns are dropped only when the values agree.

* DescriPyTor columns that repeat a morfeus/NBO column (``dist_`` = ``Bond_``, ``angle_`` = ``Angle_``,
  ``charge_NBO_`` = ``npa_*_charge``).
* Direction-free descriptors written both ways: the orientation with ascending atom indices is kept.
"""
import numpy as np
import pandas as pd

from desc_utils.extract import drop_duplicate_descriptors, duplicate_twin

LABELS = ["Ni", "N1", "C1", "H1"]


def _frame(**cols):
    # atom indices: Ni=1, N1=5, C1=3, H1=2 in both rows
    base = {"path": ["a", "b"], "Ni": [1, 1], "N1": [5, 5], "C1": [3, 3], "H1": [2, 2]}
    return pd.DataFrame({**base, **cols})


def test_identical_twins_dropped_and_differing_kept():
    df = _frame(**{
        "Bond_Ni_H1": [1.5, 1.6], "dist_Ni-H1": [1.5, 1.6],               # identical -> drop dist_
        "Angle_N1_Ni_C1": [80.0, 81.0], "angle_C1-Ni-N1": [80.0, 81.0],   # reversed order still a twin
        "npa_Ni_charge": [0.2, np.nan], "charge_NBO_Ni": [0.2, np.nan],
        "Bond_C1_N1": [1.3, 1.4], "dist_C1-N1": [1.3, 1.5],               # differs -> both kept
        "B1_Ni-H1": [1.97, 1.97], "Sterimol_Ni_H1_Bmin": [2.0, 2.1],       # Sterimol is never collapsed
    })
    out = drop_duplicate_descriptors(df, quiet=True)
    assert not {"dist_Ni-H1", "angle_C1-Ni-N1", "charge_NBO_Ni"} & set(out.columns)
    assert {"dist_C1-N1", "Bond_C1_N1", "B1_Ni-H1", "Bond_Ni_H1", "npa_Ni_charge"} <= set(out.columns)


def test_reversed_pairs_keep_ascending_atom_order():
    df = _frame(**{
        "Bond_N1_C1": [1.3, 1.4], "Bond_C1_N1": [1.3, 1.4],               # C1=3 < N1=5 -> keep Bond_C1_N1
        "vib_freq_N1-C1": [1600.0, 1610.0], "vib_freq_C1-N1": [1600.0, 1610.0],
        "charge_NBO_diff_N1-C1": [0.4, 0.5], "charge_NBO_diff_C1-N1": [-0.4, -0.5],   # signed: negated twin
        "Angle_N1_Ni_C1": [80.0, 81.0], "Angle_C1_Ni_N1": [80.0, 81.0],   # ends C1=3 < N1=5
        "Sterimol_N1_C1_L": [4.0, 4.1], "Sterimol_C1_N1_L": [3.0, 3.1],   # direction dependent: both kept
        "vib_amp_N1-C1": [0.3, 0.4], "vib_amp_C1-N1": [0.3, 0.9],         # disagree -> both kept
    })
    out = drop_duplicate_descriptors(df, LABELS, quiet=True)
    assert {"Bond_C1_N1", "vib_freq_C1-N1", "charge_NBO_diff_C1-N1", "Angle_C1_Ni_N1"} <= set(out.columns)
    assert not {"Bond_N1_C1", "vib_freq_N1-C1", "charge_NBO_diff_N1-C1", "Angle_N1_Ni_C1"} & set(out.columns)
    assert {"Sterimol_N1_C1_L", "Sterimol_C1_N1_L", "vib_amp_N1-C1", "vib_amp_C1-N1"} <= set(out.columns)


def test_majority_orientation_when_atom_order_varies():
    df = pd.DataFrame({"C1": [3, 3, 9], "N1": [5, 5, 5],
                       "Bond_N1_C1": [1.3, 1.4, 1.5], "Bond_C1_N1": [1.3, 1.4, 1.5]})
    out = drop_duplicate_descriptors(df, ["C1", "N1"], quiet=True)
    assert "Bond_C1_N1" in out.columns and "Bond_N1_C1" not in out.columns


def test_twin_carries_aggregation_suffix_and_sign():
    cols = ["Bond_Ni_H1_boltz", "npa_Ni_charge_min", "charge_NBO_diff_C1-N1_max", "Bond_C1_N1_lowE"]
    assert duplicate_twin("dist_Ni-H1_boltz", cols) == ("Bond_Ni_H1_boltz", 1)
    assert duplicate_twin("charge_NBO_Ni_min", cols) == ("npa_Ni_charge_min", 1)
    assert duplicate_twin("charge_NBO_diff_C1-N1", cols) is None
    # reversed: min of the negated column is minus the max of the kept one
    assert duplicate_twin("charge_NBO_diff_N1-C1_min", cols, LABELS) == ("charge_NBO_diff_C1-N1_max", -1)
    # reversed DescriPyTor copy: dist_N1-C1 -> Bond_C1_N1
    assert duplicate_twin("dist_N1-C1_lowE", cols, LABELS) == ("Bond_C1_N1_lowE", 1)
