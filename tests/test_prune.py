"""The laptop-side conformer prune (``desc_utils.prune``, the spec's ``prune`` block).

Each test names the failure it guards against. The two that matter most are the first and the last:
a de-duplication that could drop a molecule's lowest-energy conformer would silently change every
``_lowE`` descriptor, and an RMSD that is not invariant under rigid motion would call two copies of
one minimum two different structures, which is the whole point of the step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from desc_utils import prune as pr
from desc_utils.spec import SpecError, validate_spec

SYM = ["C", "C", "O", "H", "H"]
GEOM = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.1, 1.2, 0.0],
                 [-0.5, 0.9, 0.0], [-0.5, -0.9, 0.3]])


def _rotated(X, angle=0.7, shift=(3.0, 1.0, -2.0)):
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return X @ R.T + np.asarray(shift)


def _write(folder, X, energy, sym=SYM, name="geometry.xyz"):
    folder.mkdir(parents=True, exist_ok=True)
    lines = [str(len(sym)), f"{energy} energy={energy} units=hartree"]
    lines += [f"{s} {x:.8f} {y:.8f} {z:.8f}" for s, (x, y, z) in zip(sym, X)]
    (folder / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _table(tmp_path, conformers, molecule="MOL1"):
    rows = []
    for name, X, e in conformers:
        _write(tmp_path / molecule / name, X, e)
        rows.append({"path": str(tmp_path / molecule / name), "inchikey": molecule,
                     "uma_energy_hartree": e})
    return pd.DataFrame(rows)


def _cfg(**over):
    cfg = {"energy_column": "uma_energy_hartree", "energy_window_kcal_mol": None,
           "rmsd_threshold_angstrom": None, "energy_gate_kcal_mol": None, "max_automorphisms": 1000}
    cfg.update(over)
    return cfg


def _run(df, **over):
    return pr.prune(df, _cfg(**over), molecule_column="inchikey",
                    geometry_file="geometry.xyz", progress=False)


def test_the_lowest_energy_conformer_is_never_de_duplicated_away(tmp_path):
    """Leader clustering must visit conformers in ascending energy order.

    Any other order could drop a molecule's lowest conformer in favour of a near-identical higher
    one, which would move every `_lowE` descriptor of that molecule with no symptom in the tables.
    """
    df = _table(tmp_path, [("conf_1", _rotated(GEOM), -100.0000),
                           ("conf_2", GEOM, -100.0010),
                           ("conf_3", GEOM + 1e-5, -100.0005)])
    keep, record = _run(df, rmsd_threshold_angstrom=0.01)
    lowest = df.loc[df["uma_energy_hartree"].idxmin(), "path"]
    assert lowest in set(keep["path"])
    assert record.loc[record["path"] == lowest, "prune_status"].item() == pr.KEPT


def test_two_copies_of_one_minimum_collapse_however_they_are_oriented(tmp_path):
    """The RMSD must be invariant under rotation and translation.

    The same minimum found by two searches arrives in two frames; comparing coordinates as written
    would call them different structures and the pooled library would carry every minimum several
    times over, which is exactly what pooling seven ensembles makes likely.
    """
    df = _table(tmp_path, [("conf_1", GEOM, -100.0010),
                           ("conf_2", _rotated(GEOM), -100.0009)])
    keep, record = _run(df, rmsd_threshold_angstrom=0.01)
    assert len(keep) == 1
    dropped = record[record["prune_status"] == pr.DUPLICATE].iloc[0]
    assert dropped["prune_leader"] == str(tmp_path / "MOL1" / "conf_1")
    assert dropped["prune_rmsd_to_leader"] < 1e-6


def test_a_genuinely_different_conformer_survives_the_threshold(tmp_path):
    """A prune that kept only one structure per molecule would pass every test above."""
    moved = GEOM.copy()
    moved[2] += [0.0, 0.9, 0.4]
    df = _table(tmp_path, [("conf_1", GEOM, -100.0010), ("conf_2", moved, -100.0005)])
    keep, _ = _run(df, rmsd_threshold_angstrom=0.01)
    assert len(keep) == 2


def test_the_energy_window_is_measured_per_molecule(tmp_path):
    """Against the whole table's minimum instead, every conformer of a less stable molecule would be
    dropped at once - the molecule would simply vanish from the library."""
    df = pd.concat([_table(tmp_path, [("conf_1", GEOM, -100.0), ("conf_2", GEOM, -99.98)], "MOL1"),
                    _table(tmp_path, [("conf_1", GEOM, -50.0), ("conf_2", GEOM, -49.98)], "MOL2")])
    keep, record = _run(df, energy_window_kcal_mol=6.0)
    assert set(keep["inchikey"]) == {"MOL1", "MOL2"}
    assert len(keep) == 2                                   # the 12.55 kcal/mol one of each
    assert (record["prune_status"] == pr.ABOVE_WINDOW).sum() == 2


def test_the_window_runs_before_the_rmsd_pass(tmp_path):
    """Otherwise a conformer above the window could be the leader of one inside it, and the record
    would name a leader the library does not contain."""
    df = _table(tmp_path, [("conf_1", GEOM, -100.0000),     # 0.0 kcal/mol, inside
                           ("conf_2", GEOM, -99.9700)])     # 18.8 kcal/mol, outside
    _, record = _run(df, energy_window_kcal_mol=6.0, rmsd_threshold_angstrom=0.01)
    outside = record[record["path"].str.endswith("conf_2")].iloc[0]
    assert outside["prune_status"] == pr.ABOVE_WINDOW
    assert outside["prune_leader"] == ""


def test_pruning_disabled_changes_nothing(tmp_path):
    """`enabled: false` must be the identity, so a spec can carry the block while a run ignores it."""
    df = _table(tmp_path, [("conf_1", GEOM, -100.0), ("conf_2", GEOM, -99.0)])
    keep, record = _run(df)
    assert len(keep) == len(df)
    assert set(record["prune_status"]) == {pr.KEPT}


def test_a_conformer_without_an_energy_is_recorded_not_silently_kept(tmp_path):
    """It cannot be placed in the window or in the clustering order, so it is named rather than
    being carried into the library as though it had been judged."""
    df = _table(tmp_path, [("conf_1", GEOM, -100.0)])
    df.loc[len(df)] = {"path": df.loc[0, "path"], "inchikey": "MOL1",
                       "uma_energy_hartree": float("nan")}
    keep, record = _run(df, energy_window_kcal_mol=6.0)
    assert pr.NO_ENERGY in set(record["prune_status"])
    assert len(keep) == 1


def test_conformers_with_different_atom_orders_are_refused_by_name(tmp_path):
    """An RMSD between two atom orders compares different atoms and returns a plausible number."""
    df = _table(tmp_path, [("conf_1", GEOM, -100.0)])
    other = tmp_path / "MOL1" / "conf_2"
    order = [2, 1, 0, 3, 4]                                 # O and C swapped: a different atom order
    _write(other, GEOM[order], -99.999, sym=[SYM[i] for i in order])
    df.loc[len(df)] = {"path": str(other), "inchikey": "MOL1", "uma_energy_hartree": -99.999}
    with pytest.raises(ValueError, match="atom order"):
        _run(df, rmsd_threshold_angstrom=0.01)


# --------------------------------------------------------------------------- the spec block
def _spec(prune_block):
    return {
        "name": "t", "data": {"root": "DATA/x", "molecule_id_column": "inchikey"},
        "atoms": {"C1": "C"},
        "numbering": {"backend": "first_n", "first_n": {"n": 1},
                      "energy": {"source": "xyz_comment", "key": "energy", "units": "hartree",
                                 "column": "uma_energy_hartree"}},
        "descriptors": {"orca": {"xyz": {"atoms": ["C1"]}}},
        "prune": prune_block,
    }


def test_the_prune_energy_column_defaults_to_the_numbering_energy():
    """The prune, `_lowE` and the Boltzmann weights must be on one energy unless the spec says
    otherwise; guessing a different one would prune on one quantity and condense on another."""
    spec = validate_spec(_spec({"enabled": True, "energy_window_kcal_mol": 6.0}))
    assert spec.prune["energy_column"] == "uma_energy_hartree"
    assert spec.aggregate["energy_column"] == "uma_energy_hartree"


def test_a_loose_rmsd_threshold_with_the_default_energy_gate_is_refused():
    """The gate is an empirical shortcut measured at tight thresholds only. Applied above them it
    would quietly call distant-in-energy duplicates distinct, so it has to be chosen explicitly."""
    with pytest.raises(SpecError, match="energy_gate_kcal_mol"):
        validate_spec(_spec({"enabled": True, "rmsd_threshold_angstrom": 0.5}))
    validate_spec(_spec({"enabled": True, "rmsd_threshold_angstrom": 0.5,
                         "energy_gate_kcal_mol": None}))


def test_an_enabled_prune_with_no_criterion_is_refused():
    with pytest.raises(SpecError, match="would do nothing"):
        validate_spec(_spec({"enabled": True}))


def test_an_unknown_prune_key_is_refused():
    """A misspelt key would otherwise be dropped and the run would prune on the default."""
    with pytest.raises(SpecError, match="rmsd_threshold"):
        validate_spec(_spec({"enabled": True, "rmsd_threshold": 0.01}))


# --------------------------------------------------------------------------- empty-family report
def test_descriptor_columns_that_are_empty_for_every_conformer_are_named(capsys):
    """`aggregate` drops all-NaN columns, so a descriptor family that produced nothing simply does
    not appear in the spreadsheet and the run looks clean. That is how a g-xTB library was
    extracted with all 21 Fukui columns empty - the redox family's `cm5` prefix cannot fill on
    g-xTB output (no Hirshfeld charges to derive CM5 from) while the IP/EA/hardness columns of the
    same family filled normally and hid it.
    """
    import sys as _sys
    from types import SimpleNamespace

    from desc_utils.run import _report_empty_columns

    df = pd.DataFrame({
        "path": ["a", "b"], "inchikey": ["M", "M"],
        "%Vbur_atom_Ni_radius_3.5": [41.2, 40.8],
        "fplus_cm5_Ni": [np.nan, np.nan],
        "fminus_cm5_Ni": [np.nan, np.nan],
    })
    _report_empty_columns(df, SimpleNamespace(labels=["Ni"]))
    err = capsys.readouterr().err
    assert "2 of 3" in err
    assert "fplus_cm5_Ni" in err and "fminus_cm5_Ni" in err
    assert "%Vbur" not in err


def test_nothing_is_reported_when_every_descriptor_column_has_a_value(capsys):
    from types import SimpleNamespace

    from desc_utils.run import _report_empty_columns

    df = pd.DataFrame({"path": ["a"], "inchikey": ["M"], "%Vbur_atom_Ni_radius_3.5": [41.2]})
    _report_empty_columns(df, SimpleNamespace(labels=["Ni"]))
    assert capsys.readouterr().err == ""
