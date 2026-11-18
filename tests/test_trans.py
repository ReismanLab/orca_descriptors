"""H1/H2 by the donor each hydride is trans to (``desc_utils.trans``, the spec's ``trans`` block).

The failure this guards against is the one the pass was written for: before it, the two
hydride labels were the two atoms' positions in the xyz file, so `Ni-H1` named a
different site on every molecule and no model could use it. A pass that swapped the
wrong way, or that invented an assignment on a complex with no trans pair, would be
worse - it would wear a chemical name while still being arbitrary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from desc_utils import trans as tr

CFG = {"enabled": True, "centre": "Ni", "pairs": [["H1", "H2"]],
       "trans_to": ["N2", "N1"], "min_margin_degrees": 20.0}

# A square-planar L-Ni(H)2 sketch in the xy plane: the donors on +x and -y, each hydride
# directly opposite one of them.
SYM = ["Ni", "N", "N", "H", "H"]
XYZ = np.array([[0.0, 0.0, 0.0],       # Ni
                [1.9, 0.0, 0.0],       # the donor labelled N1
                [0.0, -1.9, 0.0],      # the donor labelled N2
                [0.0, 1.5, 0.0],       # the hydride trans to N2
                [-1.5, 0.0, 0.0]])     # the hydride trans to N1


def _row(tmp_path, h1, h2, xyz=XYZ, name="MOL1", conf="conf_1"):
    """One conformer whose H1/H2 columns are the 1-based rows given."""
    folder = tmp_path / name / conf
    folder.mkdir(parents=True, exist_ok=True)
    lines = [str(len(SYM)), "energy=-1.0 units=hartree"]
    lines += [f"{s} {x:.8f} {y:.8f} {z:.8f}" for s, (x, y, z) in zip(SYM, xyz)]
    (folder / "geometry.xyz").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"path": str(folder), "xyz_name": "geometry", "inchikey": name,
            "Ni": 1, "N1": 2, "N2": 3, "H1": h1, "H2": h2}


def _relabel(rows):
    return tr.relabel(pd.DataFrame(rows), CFG, geometry_file="geometry.xyz")


def test_the_hydride_trans_to_n2_becomes_h1_whichever_way_it_came_in(tmp_path):
    """The guard: the labels must follow the geometry, not the order the SMARTS matched."""
    right = _row(tmp_path, h1=4, h2=5, conf="a")      # already correct
    wrong = _row(tmp_path, h1=5, h2=4, conf="b")      # the other way round
    out, record = _relabel([right, wrong])
    assert list(out["H1"]) == [4, 4] and list(out["H2"]) == [5, 5]
    assert list(record["swapped"]) == [False, True]
    assert record["decided"].all()


def test_the_decision_is_the_angle_and_not_the_distance(tmp_path):
    """A distorted complex can put the hydride trans to a donor FURTHER from it.

    Deciding on the Ni-H...N distance would then swap the two labels the wrong way.
    """
    xyz = XYZ.copy()
    xyz[3] = [0.0, 2.6, 0.0]        # the trans hydride, pushed out
    xyz[4] = [-1.1, 0.0, 0.0]       # the cis one, pulled in
    assert np.linalg.norm(xyz[3] - xyz[2]) > np.linalg.norm(xyz[4] - xyz[2])
    out, record = _relabel([_row(tmp_path, h1=5, h2=4, xyz=xyz)])
    assert out["H1"].iloc[0] == 4 and out["H2"].iloc[0] == 5


def test_a_complex_with_no_trans_pair_keeps_its_labels_and_says_so(tmp_path):
    """Inventing an assignment on a tetrahedral complex would be the arbitrary label
    this pass removes, now wearing a chemical name."""
    xyz = XYZ.copy()
    xyz[3] = [0.0, 1.1, 1.1]        # both hydrides roughly 90 degrees from N1
    xyz[4] = [0.0, 1.1, -1.1]
    out, record = _relabel([_row(tmp_path, h1=4, h2=5, xyz=xyz)])
    assert out["H1"].iloc[0] == 4 and out["H2"].iloc[0] == 5
    assert not record["decided"].iloc[0] and not record["swapped"].iloc[0]
    assert "no trans pair" in record["reason"].iloc[0]
    assert tr.summary(record)["n_undecided"] == 1


def test_a_conformer_with_no_numbering_is_left_alone(tmp_path):
    """An unnumbered conformer must not be dropped or given a guess."""
    row = _row(tmp_path, h1=4, h2=5)
    row["H1"] = row["N1"] = None
    out, record = _relabel([row])
    assert not record["decided"].iloc[0]
    assert tr.summary(record)["n_without_numbering"] == 1


def test_the_pass_is_idempotent(tmp_path):
    """It is written back over the numbering table, so a re-run must swap nothing."""
    once, _ = _relabel([_row(tmp_path, h1=5, h2=4)])
    twice, record = tr.relabel(once, CFG, geometry_file="geometry.xyz")
    assert not record["swapped"].any()
    assert list(twice["H1"]) == list(once["H1"])


def test_the_spec_refuses_a_label_that_is_trans_to_itself():
    """A typo in `trans_to` must not produce a pass that compares an atom with itself."""
    from desc_utils.spec import SpecError, _validate_trans

    problems = []
    _validate_trans({"enabled": True, "pairs": [["H1", "H2"]], "trans_to": ["H1", "N1"]},
                    {"Ni", "H1", "H2", "N1", "N2"}, problems)
    assert any("trans to itself" in p for p in problems)
