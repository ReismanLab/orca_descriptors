"""BUNNY's N1/N2 convention (``desc_utils.hemisphere``, the spec's ``hemisphere`` block).

Each test names the failure it guards against. The one that matters most is the first:
the labels decide which atom every site-labelled descriptor is computed on, so getting
the side wrong renames `NMR_shift_N1` to the other nitrogen's value across a whole
library, with nothing anywhere to show it happened.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from desc_utils import hemisphere as hs

# A square-planar L-Ni(H)2 sketch: Ni at the origin, the two donors up and out, the two
# hydrides down. The bulk is a block of carbons parked over ONE donor, so which donor is
# the bulkier one is a fact about the geometry and not about the labelling.
BASE_SYM = ["Ni", "N", "N", "H", "H"]
BASE_XYZ = np.array([[0.0, 0.0, 0.0],      # Ni
                     [-1.9, 0.0, 0.9],     # the donor that starts as N1
                     [1.9, 0.0, 0.9],      # the donor that starts as N2
                     [-1.0, 0.0, -1.3],    # H1
                     [1.0, 0.0, -1.3]])    # H2
BULK = np.array([[dx, dy, 2.6] for dx in (-0.8, 0.0, 0.8) for dy in (-0.8, 0.0, 0.8)])


def _geometry(bulk_over, tmp_path, name="MOL1", energy=-1.0, conf="conf_1"):
    """Write one conformer with the bulk sitting over the named donor. Returns its row."""
    shift = np.array([-2.4 if bulk_over == "N1" else 2.4, 0.0, 0.0])
    xyz = np.vstack([BASE_XYZ, BULK + shift])
    sym = BASE_SYM + ["C"] * len(BULK)
    folder = tmp_path / name / conf
    folder.mkdir(parents=True, exist_ok=True)
    lines = [str(len(sym)), f"energy={energy} units=hartree"]
    lines += [f"{s} {x:.8f} {y:.8f} {z:.8f}" for s, (x, y, z) in zip(sym, xyz)]
    (folder / "geometry.xyz").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # 1-based, as the numbering tables are
    return {"path": str(folder), "inchikey": name, "uma_energy_hartree": energy,
            "ligand_class": "pyr6", "Ni": 1, "N1": 2, "N2": 3, "H1": 4, "H2": 5,
            "C1": 6, "C2": 7}


def _cfg(**over):
    cfg = {"centre": "Ni", "pairs": [["N1", "N2"], ["C1", "C2"]],
           "exclude": ["H1", "H2"], "radius": 6.5, "skip_classes": ["pyox"],
           "class_column": "ligand_class", "energy_column": "uma_energy_hartree"}
    cfg.update(over)
    return cfg


def _run(df, **over):
    return hs.relabel(df, _cfg(**over), molecule_column="inchikey",
                      geometry_file="geometry.xyz")


@pytest.mark.parametrize("bulky", ["N1", "N2"])
def test_the_bulkier_donor_ends_up_labelled_N2(bulky, tmp_path):
    """The whole point of the convention, asserted as chemistry rather than as a sign.

    Whichever donor carries the bulk must come out as N2, so a descriptor named for N2
    means the same position in every molecule of the library. Before this pass the label
    was whichever the SMARTS matched first, which is reproducible but arbitrary - and on
    a real library it disagreed with the steric rule for 22 of 45 molecules.
    """
    row = _geometry(bulky, tmp_path)
    bulky_atom = row[bulky]                       # the atom index carrying the bulk
    out, record = _run(pd.DataFrame([row]))
    assert record["decided"].all()
    assert out["N2"].iloc[0] == bulky_atom, "the bulkier donor is not N2"
    assert record["swapped"].iloc[0] == (bulky == "N1")


def test_the_carbons_follow_their_nitrogens(tmp_path):
    """C1/C2 are defined by which nitrogen they sit next to (SI S4.3.1), so a swap that
    moved only the nitrogens would leave every C1 descriptor on the wrong ring."""
    row = _geometry("N1", tmp_path)              # forces a swap
    out, record = _run(pd.DataFrame([row]))
    assert record["swapped"].iloc[0]
    assert (out["C1"].iloc[0], out["C2"].iloc[0]) == (row["C2"], row["C1"])


def test_a_second_pass_swaps_nothing(tmp_path):
    """The numbering table is written back, so the stage must be a fixed point: swapping
    flips the frame with it, the bulkier side becomes north, and nothing moves again. If
    it oscillated, two runs of the same spec would give two different libraries."""
    out1, rec1 = _run(pd.DataFrame([_geometry("N1", tmp_path)]))
    assert rec1["swapped"].iloc[0]
    out2, rec2 = _run(out1)
    assert not rec2["swapped"].any()
    assert out2[["N1", "N2", "C1", "C2"]].equals(out1[["N1", "N2", "C1", "C2"]])


def test_every_conformer_of_a_molecule_moves_together(tmp_path):
    """The decision is made on the lowest-energy conformer and applied to the rest. Per
    conformer it would give one ligand different labels in different conformers, which is
    what `match_selection: lowest_indices` exists to prevent upstream."""
    rows = [_geometry("N1", tmp_path, energy=e, conf=f"conf_{i}")
            for i, e in enumerate([-1.0, -0.9, -0.8])]
    out, record = _run(pd.DataFrame(rows))
    assert len(record) == 1 and record["swapped"].iloc[0]
    assert out["N2"].nunique() == 1 and out["N2"].iloc[0] == rows[0]["N1"]


def test_a_pyox_ligand_keeps_its_own_convention(tmp_path):
    """For PyOx the SI fixes N1 as the aromatic nitrogen and N2 as the aliphatic one, so
    sterics get no vote. BUNNY holds them out of this step and so must we."""
    row = _geometry("N1", tmp_path) | {"ligand_class": "pyox"}
    out, record = _run(pd.DataFrame([row]))
    assert not record["decided"].iloc[0] and not record["swapped"].iloc[0]
    assert "pyox" in record["reason"].iloc[0]
    assert out["N1"].iloc[0] == row["N1"] and out["N2"].iloc[0] == row["N2"]


def test_a_molecule_with_no_usable_energy_is_recorded_not_silently_kept(tmp_path):
    """An undecided molecule keeps the numbering it had, which is right - but silence
    would make it indistinguishable from one the rule looked at and left alone."""
    row = _geometry("N1", tmp_path) | {"uma_energy_hartree": float("nan")}
    out, record = _run(pd.DataFrame([row]))
    assert not record["decided"].iloc[0]
    assert "energy" in record["reason"].iloc[0]
    assert hs.summary(record)["n_undecided"] == 1
    assert hs.summary(record)["n_swapped"] == 0, "an undecided molecule is not a swap"
