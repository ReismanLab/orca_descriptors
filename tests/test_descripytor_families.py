import numpy as np
import pandas as pd

from desc_utils.extract.descripytor import FAMILIES, FAMILY_ORDER, extract_one, label_columns, to_schema_b
from desc_utils.spec import DESCRIPYTOR_TERM_ARITY

CFG = {
    "charges": ["Ni", "N1"],
    "sterimol": [["Ni", "N1"]],
    "bond_length": [["Ni", "N1"]],
    "stretching_vibration": [["Ni", "N1"]],
    "bending_vibration": [],
    "charge_diff": [["Ni", "N1"]],
    "dipole": [["Ni", "N2", "N1"]],
    "bond_angle": [["N1", "Ni", "N2"]],
    "stretch_threshold": 1400, "upper_stretch_threshold": 3500, "bend_threshold": 1300,
    "radii": "CPK", "drop_atoms": [],
}
ROW = {"Ni": 2, "N1": 12, "N2": 4}


class FakeMolecules:
    """Returns DescriPyTor-shaped frames with index-based column names."""

    def _frame(self, cols):
        return pd.DataFrame([{c: float(i) for i, c in enumerate(cols, 1)}], index=["mol"])

    def get_charge_df_dict(self, atom_indices):
        return self._frame([f"nbo_atom_{i}" for i in atom_indices])

    def get_sterimol_dict(self, atom_indices, radii="CPK", **kw):
        assert "drop_atoms" not in kw, "empty drop_atoms must not be forwarded"
        return self._frame([f"{k}_{a}-{b}" for a, b in atom_indices for k in ("B1", "B5", "L", "loc_B5", "B1_B5_angle")])

    def get_bond_length_dict(self, atom_pairs):
        return self._frame([f"0_bond_length_{a}-{b}" for a, b in atom_pairs])

    def get_stretch_vibration_dict(self, atom_pairs, threshold, upper_threshold):
        assert (threshold, upper_threshold) == (1400, 3500)
        return self._frame([f"{k}_{a}_{b}" for a, b in atom_pairs for k in ("Frequency_Stretch", "Amplitude_Stretch")])

    def get_charge_diff_df_dict(self, atom_indices):
        return self._frame([f"nbo_diff_{a}-{b}" for a, b in atom_indices])

    def get_dipole_dict(self, atom_indices):
        a, b, c = atom_indices
        return self._frame([f"{k}_{a}-{b}-{c}" for k in ("total_dipole", "dipole_x", "dipole_y", "dipole_z")])

    def get_bond_angle_dict(self, atom_indices):
        return self._frame([f"0_angle_[{a}, {b}, {c}]" for a, b, c in atom_indices])


def test_families_match_spec_arity_and_order():
    assert {k: f.arity for k, f in FAMILIES.items()} == DESCRIPYTOR_TERM_ARITY
    assert sorted(FAMILY_ORDER) == sorted(FAMILIES)


def test_label_columns_use_legacy_names():
    cols = label_columns(CFG)
    assert cols[:2] == ["charge_NBO_Ni", "charge_NBO_N1"]
    assert "B1-B5_angle_Ni-N1" in cols and "dist_Ni-N1" in cols and "vib_freq_Ni-N1" in cols
    assert "dipole" in cols and "dipole_x_Ni-N2-N1" in cols      # single dipole keeps the legacy total name
    assert "angle_N1-Ni-N2" in cols


def test_extract_one_renames_index_columns_to_labels():
    out, notes = extract_one(FakeMolecules(), CFG, ROW)
    assert notes == []
    assert set(out) == set(label_columns(CFG))
    assert out["charge_NBO_Ni"] == 1.0 and out["charge_NBO_N1"] == 2.0
    assert out["dipole"] == 1.0 and out["dipole_z_Ni-N2-N1"] == 4.0


def test_missing_output_column_gives_nan_and_note():
    class Broken(FakeMolecules):
        def get_bond_length_dict(self, atom_pairs):
            return pd.DataFrame([{"unexpected": 1.0}], index=["mol"])

    out, notes = extract_one(Broken(), CFG, ROW)
    assert np.isnan(out["dist_Ni-N1"])
    assert any("bond_length" in n for n in notes)


def test_to_schema_b_resolves_indices():
    sb = to_schema_b(CFG, ROW)
    assert sb["charges"] == [2, 12]
    assert sb["sterimol"] == [[2, 12]]
    assert sb["dipole"] == [[2, 4, 12]]
    assert sb["bond_angle"] == [[12, 2, 4]]
    assert sb["stretch_threshold"] == 1400 and sb["drop_atoms"] == []
