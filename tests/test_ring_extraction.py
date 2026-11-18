"""The ring family as it reaches a table (``orca.xyz.rings`` / ``distances`` / ``ring_probes``).

`tests/test_rings.py` covers the mathematics on synthetic geometries. What is guarded here is
the wiring, where the failures are of a different kind and none of them raises: a ring whose
atoms this conformer does not carry has to be skipped and SAID, not left as an indistinguishable
blank; a through-space contact must not acquire a Sterimol vector or a `Bond_` name; and the
quadrant frame must not narrow the atom list that every later row is described with.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from desc_utils import data_extraction as de

# A flat benzene ring of exactly optimal aromatic bonds, plus a lone probe atom on its axis.
_R = 1.388 / (2 * math.sin(math.pi / 6))
_A = 2 * np.pi * np.arange(6) / 6
SYM = ["C"] * 6 + ["Ni"]
XYZ = np.vstack([np.c_[_R * np.cos(_A), _R * np.sin(_A), np.zeros(6)], [[0.0, 0.0, 2.0]]])
LABELS = {"C1": 1, "C2": 2, "C3": 3, "C4": 4, "C5": 5, "C6": 6, "Ni": 7}
RING = {"label": "benzene", "atoms": ["C1", "C2", "C3", "C4", "C5", "C6"], "homa": True}


def _conformer(tmp_path, labels=LABELS):
    folder = tmp_path / "MOL1" / "conf_1"
    folder.mkdir(parents=True, exist_ok=True)
    lines = [str(len(SYM)), "energy=-1.0 units=hartree"]
    lines += [f"{s} {x:.8f} {y:.8f} {z:.8f}" for s, (x, y, z) in zip(SYM, XYZ)]
    (folder / "geometry.xyz").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return pd.DataFrame([{"path": str(folder), **labels}])


def _describe(df, **over):
    over.setdefault("atoms", ["Ni"])
    over.setdefault("bonds", [])
    over.setdefault("angles", [])
    over.setdefault("dihedrals", [])
    out = de.get_xyz_descriptors(df, progress=False, geometry_file="geometry.xyz",
                                 overwrite_json=True, **over)
    return out[next(iter(out))]


def test_the_ring_family_reaches_the_table_with_its_label_on_it(tmp_path):
    res = _describe(_conformer(tmp_path), rings=[RING],
                    ring_probes=[{"ring": "benzene", "atom": "Ni"}])
    assert res["Ring_benzene_HOMA"] == pytest.approx(1.0, abs=1e-9)
    assert res["Ring_benzene_planarity_rms"] == pytest.approx(0.0, abs=1e-9)
    assert res["RingProbe_benzene_Ni_height"] == pytest.approx(2.0, abs=1e-9)
    assert res["RingProbe_benzene_Ni_centroid_distance"] == pytest.approx(2.0, abs=1e-9)


def test_a_ring_this_conformer_does_not_carry_is_skipped_and_named(tmp_path):
    """The ring stage is `required: false`, so a ligand of another class has no pyridine
    labels. Letting `int(nan)` raise would take the whole library's extraction down for one
    ligand; leaving the columns silently blank would be indistinguishable from a ring that was
    never asked for."""
    labels = dict(LABELS); labels["C4"] = float("nan")
    res = _describe(_conformer(tmp_path, labels), rings=[RING])
    assert "Ring_benzene_HOMA" not in res
    assert "C4" in res["Ring_benzene_note"]
    assert res["SASA"] > 0            # the rest of the conformer is still described


def test_a_through_space_contact_is_not_given_a_bond_name_or_a_sterimol_vector(tmp_path):
    """`bonds` entries also build Sterimol parameters, and cutting a ring bond disconnects
    nothing - the substituent walk comes back round and measures the whole molecule. So the
    ring's own contacts go through `distances`, which must produce neither."""
    res = _describe(_conformer(tmp_path), distances=[["C1", "C4"], ["C1", "C2"]])
    assert res["Dist_C1_C4"] == pytest.approx(2 * _R, abs=1e-9)       # across the ring
    assert res["Dist_C1_C2"] == pytest.approx(1.388, abs=1e-9)        # a bond, measured as one
    assert not [k for k in res if k.startswith("Bond_")]
    assert not [k for k in res if k.startswith("Sterimol_")]


def test_a_quadrant_frame_does_not_narrow_the_atoms_the_next_row_is_described_with(tmp_path):
    """`atoms` was rebound inside the quadrant branch, so from the second row on every
    conformer was described with only the quadrant's own atoms - no %Vbur, SASA or
    pyramidalization for the rest. Never fired, because every ORCA spec ships `quadrants: []`,
    and it would have produced a table that simply lacked columns rather than wrong ones."""
    # The quadrant frame is DISJOINT from the described atoms on purpose: a frame that happens
    # to contain them survives the rebinding and the test passes on the bug, which is exactly
    # what the first version of this test did.
    df = pd.concat([_conformer(tmp_path), _conformer(tmp_path)], ignore_index=True)
    out = de.get_xyz_descriptors(
        df, atoms=["C1", "C2", "Ni"], bonds=[], angles=[], dihedrals=[],
        quadrants=[{"label": "frame", "atoms": ["C3", "C4", "C5", "C6"], "radius": 5.0}],
        progress=False, geometry_file="geometry.xyz", overwrite_json=True)
    rows = list(out.values())
    assert len(rows) == 2
    for n, row in enumerate(rows):
        for atom in ("C1", "C2", "Ni"):
            assert f"SASA_{atom}" in row, (
                f"{atom} missing from row {n}: the quadrant frame narrowed the atom list")
