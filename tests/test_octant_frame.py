"""The octant/quadrant frame (the spec's ``orca.xyz.quadrants`` entries).

Both tests guard a silent failure: neither would have raised, and both would have left a
library that looks complete and describes something else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from desc_utils import data_extraction as de
from desc_utils import orca_descriptors as od

SYM = ["Ni", "N", "N", "H", "H"] + ["C"] * 9
XYZ = np.vstack([
    np.array([[0.0, 0.0, 0.0], [-1.9, 0.0, 0.9], [1.9, 0.0, 0.9],
              [-1.0, 0.0, -1.3], [1.0, 0.0, -1.3]]),
    np.array([[dx - 2.4, dy, 2.6] for dx in (-0.8, 0.0, 0.8) for dy in (-0.8, 0.0, 0.8)]),
])
LABELS = {"Ni": 1, "N1": 2, "N2": 3, "H1": 4, "H2": 5}


def _conformer(tmp_path):
    folder = tmp_path / "MOL1" / "conf_1"
    folder.mkdir(parents=True, exist_ok=True)
    lines = [str(len(SYM)), "energy=-1.0 units=hartree"]
    lines += [f"{s} {x:.8f} {y:.8f} {z:.8f}" for s, (x, y, z) in zip(SYM, XYZ)]
    (folder / "geometry.xyz").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return pd.DataFrame([{"path": str(folder), **LABELS}])


ENTRY = {"label": "Ni", "atoms": ["Ni", "N1", "N2", "N2"],
         "exclude": ["H1", "H2"], "radius": 5.0}


def _describe(df, quadrants, **over):
    """The descriptors of the single conformer in `df`. Returns the results dict itself,
    which is what `get_xyz_descriptors` gives back - one entry per row index."""
    out = de.get_xyz_descriptors(df, atoms=["Ni"], bonds=[], angles=[], dihedrals=[],
                                 quadrants=quadrants, progress=False,
                                 geometry_file="geometry.xyz", **over)
    return out[next(iter(out))]


def test_the_entrys_own_radius_and_exclusions_reach_morfeus(tmp_path):
    """`quadrant_radius` defaults to 6.5 A with nothing excluded, while BUNNY's octants
    are 5.0 A with the two hydrides taken out of the volume. Before the entry could carry
    them the class defaults won, so the columns were named for BUNNY's frame and computed
    in a different one - a library that looks right and is not comparable with theirs.
    """
    from morfeus import BuriedVolume

    got = _describe(_conformer(tmp_path), [ENTRY], overwrite_json=True)

    ref = BuriedVolume(SYM, XYZ, LABELS["Ni"], radius=5.0, include_hs=True,
                       excluded_atoms=[LABELS["H1"], LABELS["H2"]],
                       z_axis_atoms=[LABELS["N1"], LABELS["N2"]],
                       xz_plane_atoms=[LABELS["N2"]])
    ref.octant_analysis()
    assert got["%Vbur_Ni_octant_+,+,+"] == pytest.approx(
        ref.octants["percent_buried_volume"][0], abs=1e-9)

    # and the defaults would NOT have given this: a different sphere, nothing excluded
    other = BuriedVolume(SYM, XYZ, LABELS["Ni"], radius=6.5, include_hs=True,
                         z_axis_atoms=[LABELS["N1"], LABELS["N2"]],
                         xz_plane_atoms=[LABELS["N2"]])
    other.octant_analysis()
    assert got["%Vbur_Ni_octant_+,+,+"] != pytest.approx(
        other.octants["percent_buried_volume"][0], abs=1e-6), \
        "the test cannot tell the two frames apart, so it proves nothing"


def test_a_cached_geometry_is_not_trusted_when_the_octants_are_new(tmp_path):
    """The cache is trusted when it holds every column the spec asks for, and the octant
    columns were missing from that list. A library extracted once WITHOUT quadrants then
    kept every xyz.json, and adding the entry produced no octant column at all - no error,
    no warning, just a library silently missing the descriptors that were asked for.
    """
    df = _conformer(tmp_path)
    first = _describe(df.copy(), None, overwrite_json=True)        # writes the cache
    assert not [c for c in first if "octant" in c]

    second = _describe(df.copy(), [ENTRY])                          # cache must be refused
    assert [c for c in second if "octant" in c], \
        "the cached xyz.json was trusted and the octants never computed"
