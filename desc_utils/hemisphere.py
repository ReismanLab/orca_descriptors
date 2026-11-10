"""BUNNY's N1/N2 convention: the bulkier donor is N2, decided by buried-volume hemispheres.

The SMARTS stages number a complex in whatever order the pattern matched, which is
reproducible but arbitrary - `match_selection: lowest_indices` only guarantees that one
molecule gets the SAME arbitrary labels in all of its conformers. BUNNY's library instead
fixes the labels chemically (SI S4.3.1): N2 is the sterically bulkier donor position.

This is a post-pass over the numbering table rather than a `numbering.orient` rule,
because the decision is per MOLECULE and `apply_orient` runs per conformer. Deciding per
conformer would give one ligand different labels in different conformers, which is the
failure `lowest_indices` exists to prevent.

The rule is taken from BUNNY's own code (`get_properties_functions_m.py`, the
`readjust`/hemisphere block), not from the SI prose, and three details differ from what
the prose alone would suggest:

* the hemispheres are **quadrant sums** in the morfeus octant frame -
  north = `+,+` plus `+,-`, south = `-,-` plus `-,+`;
* the deciding sphere is **6.5 A**, while the octant DESCRIPTORS are computed at 5.0 A
  (the SI quotes 5 A, which is the descriptor radius, not this one);
* the decision is made on each molecule's **lowest-energy conformer** and then applied
  to every conformer of it.

PyOx ligands are excluded, as they are in BUNNY: there N1 is the aromatic nitrogen and
N2 the aliphatic one by definition, and sterics do not get a vote.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .prune import read_xyz

# morfeus returns the four quadrants under these keys; north/south is the sign of the
# first coordinate, so the pair that shares a leading `+` is one hemisphere.
NORTH_QUADRANTS = (1, 4)
SOUTH_QUADRANTS = (3, 2)


def hemispheres(path: str, geometry_file: str, idx: dict, exclude: list[int],
                radius: float) -> tuple[float, float]:
    """(north, south) buried volume around the metal, in the N1/N2 frame."""
    from morfeus import BuriedVolume

    sym, X = read_xyz(f"{path}/{geometry_file}")
    bv = BuriedVolume(sym, X, int(idx["centre"]), radius=float(radius), include_hs=True,
                      excluded_atoms=[int(i) for i in exclude] or None,
                      z_axis_atoms=[int(idx["first"]), int(idx["second"])],
                      xz_plane_atoms=[int(idx["second"])])
    bv.octant_analysis()
    q = bv.quadrants["percent_buried_volume"]
    return (sum(q[i] for i in NORTH_QUADRANTS), sum(q[i] for i in SOUTH_QUADRANTS))


def relabel(df: pd.DataFrame, cfg: dict, molecule_column: str, geometry_file: str
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Swap each molecule's label pairs where the south hemisphere is the bulkier one.

    Returns the re-labelled numbering table and a per-molecule record of the decision.
    """
    centre, pairs = cfg["centre"], [list(p) for p in cfg["pairs"]]
    first, second = pairs[0]
    energy_col, radius = cfg["energy_column"], float(cfg["radius"])
    exclude_labels = list(cfg.get("exclude") or [])
    skip = set(cfg.get("skip_classes") or [])
    class_col = cfg.get("class_column")

    out = df.copy()
    rows = []
    for mol, g in df.groupby(molecule_column, sort=False):
        cls = str(g[class_col].iloc[0]) if class_col and class_col in g.columns else ""
        if cls in skip:
            rows.append({"molecule": mol, "ligand_class": cls, "decided": False,
                         "reason": f"ligand_class {cls!r} keeps its own convention",
                         "swapped": False, "north": None, "south": None,
                         "conformer": None})
            continue
        e = pd.to_numeric(g[energy_col], errors="coerce")
        usable = g[e.notna() & g[[centre, first, second]].notna().all(axis=1)]
        if usable.empty:
            rows.append({"molecule": mol, "ligand_class": cls, "decided": False,
                         "reason": "no conformer has both an energy and a full numbering",
                         "swapped": False, "north": None, "south": None,
                         "conformer": None})
            continue
        row = usable.loc[pd.to_numeric(usable[energy_col], errors="coerce").idxmin()]
        try:
            north, south = hemispheres(
                row["path"], geometry_file,
                {"centre": row[centre], "first": row[first], "second": row[second]},
                [row[a] for a in exclude_labels if a in row.index], radius)
        except Exception as exc:                              # noqa: BLE001
            rows.append({"molecule": mol, "ligand_class": cls, "decided": False,
                         "reason": f"{type(exc).__name__}: {exc}", "swapped": False,
                         "north": None, "south": None, "conformer": row["path"]})
            continue
        swap = south > north
        if swap:
            m = out[molecule_column] == mol
            for a, b in pairs:
                out.loc[m, [a, b]] = out.loc[m, [b, a]].to_numpy()
        rows.append({"molecule": mol, "ligand_class": cls, "decided": True, "reason": "",
                     "swapped": bool(swap), "north": float(north), "south": float(south),
                     "conformer": row["path"]})
    return out, pd.DataFrame(rows)


def summary(record: pd.DataFrame) -> dict:
    """Counts for the run log. A molecule that could not be decided is NOT a swap."""
    decided = record[record["decided"]] if len(record) else record
    margin = (decided["north"] - decided["south"]).abs() if len(decided) else pd.Series(dtype=float)
    return {
        "n_molecules": int(len(record)),
        "n_decided": int(len(decided)),
        "n_swapped": int(decided["swapped"].sum()) if len(decided) else 0,
        "n_skipped_by_class": int((~record["decided"] & record["reason"].str.contains(
            "ligand_class", na=False)).sum()) if len(record) else 0,
        "n_undecided": int((~record["decided"]).sum()) if len(record) else 0,
        # A near-tie is a real state on a symmetric ligand, where the two donors are the
        # same atom and the winner is noise. Reported rather than hidden: the labels are
        # then arbitrary, which is harmless for the symmetric case and worth knowing.
        "smallest_margin": float(margin.min()) if len(margin) else None,
        "n_within_0p5_percent": int((margin < 0.5).sum()) if len(margin) else 0,
    }
