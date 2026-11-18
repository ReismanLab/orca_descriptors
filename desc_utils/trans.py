"""H1/H2 by the donor each hydride sits opposite: H1 trans to N2, H2 trans to N1.

The SMARTS stages number the two hydrides in match order, which is reproducible but
chemically empty: measured on the pooled nih2 library, `(H1, Ni, H2)` was `(1, 2, 3)` in
all 82,273 numbered conformers, i.e. the label was the atom's position in the xyz file.
Nothing downstream reorders them - `hemisphere.relabel` swaps `[N1, N2]` and `[C1, C2]`
only, and it EXCLUDES the hydrides from its deciding sphere - so `fzero_mulliken_H1`,
`Ni-H1` and every other per-hydride column named one of two interchangeable atoms, and
the same column named a different site on the next molecule.

This pass gives them the one label a square-planar L-Ni(H)2 complex can carry: each
hydride is trans to one of the two donors, so once N1/N2 are fixed the hydrides are too.

Three things about it, each a deliberate choice:

* **It runs after the hemisphere pass, per CONFORMER.** N1/N2 must be final first, since
  the rule is stated in terms of them. Per conformer rather than per molecule - unlike
  the N1/N2 decision, which is a steric comparison and would give one ligand different
  labels in different conformers, "which donor is this hydride opposite" is a discrete
  geometric fact, ~180 degrees against ~90, that every conformer answers the same way
  unless the complex itself has distorted.
* **The decision is the L-Ni-N angle**, not a distance: in a distorted complex the
  hydride trans to the bulkier donor can still be the nearer one to it.
* **A conformer whose two angles are closer together than `min_margin_degrees` is
  recorded as undecided and left alone**, keeping its match-order labels. That is a
  complex far from square planar (a tetrahedral or five-coordinate geometry has no trans
  pair), and inventing an assignment there would be the arbitrary label this pass exists
  to remove, now wearing a chemical name.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .prune import read_xyz


def angle_degrees(X: np.ndarray, centre: int, a: int, b: int) -> float:
    """The a-centre-b angle in degrees, from 0-based rows of a coordinate array."""
    u = X[a] - X[centre]
    v = X[b] - X[centre]
    c = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return float(np.degrees(np.arccos(min(1.0, max(-1.0, c)))))


def _indices(row, labels, base: int):
    """0-based coordinate rows for `labels`, or None if any is missing."""
    out = []
    for l in labels:
        v = row.get(l)
        if v is None or (isinstance(v, float) and np.isnan(v)) or v == "":
            return None
        out.append(int(float(v)) - base)
    return out


def relabel(df: pd.DataFrame, cfg: dict, geometry_file: str, index_base: int = 1
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Swap each conformer's hydride labels so the first is trans to the first donor.

    Returns the re-labelled numbering table and a per-conformer record of the decision.
    """
    centre = cfg["centre"]
    first, second = list(cfg["pairs"][0])
    ref_first, ref_second = list(cfg["trans_to"])
    margin_min = float(cfg["min_margin_degrees"])

    out = df.copy()
    rows = []
    for i, row in df.iterrows():
        idx = _indices(row, [centre, first, second, ref_first, ref_second], index_base)
        if idx is None:
            rows.append({"path": row.get("path"), "xyz_name": row.get("xyz_name"),
                         "decided": False, "swapped": False,
                         "reason": "the conformer has no complete numbering",
                         "angle_first_to_ref": None, "angle_second_to_ref": None,
                         "margin": None})
            continue
        c, h_first, h_second, n_first, _ = idx
        try:
            _, X = read_xyz(f"{row['path']}/{geometry_file}")
            a_first = angle_degrees(X, c, h_first, n_first)
            a_second = angle_degrees(X, c, h_second, n_first)
        except Exception as exc:                                        # noqa: BLE001
            rows.append({"path": row.get("path"), "xyz_name": row.get("xyz_name"),
                         "decided": False, "swapped": False,
                         "reason": f"{type(exc).__name__}: {exc}",
                         "angle_first_to_ref": None, "angle_second_to_ref": None,
                         "margin": None})
            continue
        margin = abs(a_first - a_second)
        if margin < margin_min:
            rows.append({"path": row.get("path"), "xyz_name": row.get("xyz_name"),
                         "decided": False, "swapped": False,
                         "reason": f"the two {centre}-{ref_first} angles differ by "
                                   f"{margin:.1f} degrees, under the {margin_min:g} "
                                   f"required: this complex has no trans pair",
                         "angle_first_to_ref": a_first,
                         "angle_second_to_ref": a_second, "margin": margin})
            continue
        # `first` must be the one OPPOSITE `ref_first`, i.e. the larger angle to it
        swap = a_second > a_first
        if swap:
            out.loc[i, [first, second]] = out.loc[i, [second, first]].to_numpy()
        rows.append({"path": row.get("path"), "xyz_name": row.get("xyz_name"),
                     "decided": True, "swapped": bool(swap), "reason": "",
                     "angle_first_to_ref": a_first, "angle_second_to_ref": a_second,
                     "margin": margin})
    return out, pd.DataFrame(rows)


def summary(record: pd.DataFrame) -> dict:
    """Counts for the run log. A conformer that could not be decided is NOT a swap."""
    if not len(record):
        return {"n_conformers": 0, "n_decided": 0, "n_swapped": 0, "n_undecided": 0,
                "smallest_margin": None, "n_without_numbering": 0}
    decided = record[record["decided"]]
    margin = decided["margin"] if len(decided) else pd.Series(dtype=float)
    return {
        "n_conformers": int(len(record)),
        "n_decided": int(len(decided)),
        "n_swapped": int(decided["swapped"].sum()) if len(decided) else 0,
        "n_undecided": int((~record["decided"]).sum()),
        "n_without_numbering": int(record["reason"].str.contains(
            "no complete numbering", na=False).sum()),
        "smallest_margin": float(margin.min()) if len(margin) else None,
    }
