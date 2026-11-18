"""Ring-level geometry: shape, puckering, and the HOMA aromaticity index.

Why this is a family of its own, rather than more entries in ``bonds``/``angles``/``dihedrals``:
every quantity here is a property of a SET of atoms taken together - a best-fit plane, a
centroid, a sum over all of the ring's bonds - and none of them can be written as one distance,
one angle or one torsion. The existing families each take a fixed-size tuple of labels, so a
six-membered ring's planarity has nowhere to live in them.

What is here and what is deliberately NOT here
----------------------------------------------
HOMA is a *geometric* (structural) criterion of aromaticity: it reads the ring's bond lengths
against reference lengths for an ideal aromatic bond of each element pair. It is NOT a magnetic
criterion, so it is not NICS and does not stand in for it - NICS needs the shielding at a ghost
point inside the ring, which has to be present in the quantum-chemical input and so cannot be
recovered from a finished calculation. HOMA's advantage here is the exact complement: it needs
nothing but the coordinates, so it is available for every conformer of every library, g-xTB and
ORCA alike, with nothing re-run.

HOMA is only computed for rings the spec marks ``homa: true``. On a saturated ring - an
oxazoline, whose C4 and C5 are sp3 - the formula still returns a number, a large negative one,
because the reference lengths it is measured against are aromatic ones. That number says
"these bonds are not aromatic bonds", which is true and useless, and it would sit in a model
alongside the pyridine's HOMA looking like the same measurement. So the flag is required
rather than defaulted, and a ring without it gets the shape descriptors and no HOMA.

References
----------
HOMA and its EN/GEO decomposition: Krygowski, *J. Chem. Inf. Comput. Sci.* 1993, 33, 70-78;
Krygowski & Cyranski, *Chem. Rev.* 2001, 101, 1385. Ring puckering: Cremer & Pople,
*J. Am. Chem. Soc.* 1975, 97, 1354.
"""
from __future__ import annotations

import math

import numpy as np

# Krygowski's optimal bond lengths (Å) and normalisation constants, keyed by the unordered
# element pair. alpha is chosen so that HOMA is 0 for the Kekule structure of the reference
# system and 1 for the fully delocalised one.
HOMA_PARAMETERS: dict[frozenset[str], tuple[float, float]] = {
    frozenset(("C", "C")): (1.388, 257.70),
    frozenset(("C", "N")): (1.334, 93.52),
    frozenset(("C", "O")): (1.265, 157.38),
    frozenset(("N", "N")): (1.309, 130.33),
    frozenset(("N", "O")): (1.248, 57.21),
    frozenset(("C", "P")): (1.698, 118.91),
    frozenset(("C", "S")): (1.677, 94.09),
}


def _as_array(coords) -> np.ndarray:
    X = np.asarray(coords, dtype=float)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError(f"ring coordinates must be (n, 3), got {X.shape}")
    if len(X) < 3:
        raise ValueError(f"a ring needs at least 3 atoms, got {len(X)}")
    return X


def best_plane(coords) -> tuple[np.ndarray, np.ndarray]:
    """Centroid and unit normal of the least-squares plane through the atoms.

    The normal is the singular vector of smallest singular value, i.e. the direction of least
    spread. Its SIGN is arbitrary - SVD does not fix it - so nothing downstream may read a
    signed height against this normal as chemistry without first orienting it against something
    physical. `atom_height` returns the magnitude for that reason.
    """
    X = _as_array(coords)
    centroid = X.mean(axis=0)
    _, _, vh = np.linalg.svd(X - centroid)
    return centroid, vh[2] / np.linalg.norm(vh[2])


def planarity(coords) -> dict[str, float]:
    """RMS and maximum distance of the ring atoms from their own best-fit plane, in Å."""
    X = _as_array(coords)
    centroid, normal = best_plane(X)
    d = (X - centroid) @ normal
    return {"planarity_rms": float(np.sqrt(np.mean(d ** 2))),
            "planarity_worst": float(np.max(np.abs(d)))}   # not `_max`: see `bond_statistics`


def cremer_pople(coords) -> dict[str, float]:
    """Cremer-Pople puckering coordinates for a ring given in CONNECTIVITY order.

    The atom order is the ring walk, not the order they happen to appear in the file: the
    phase angles are defined against position around the ring, so a shuffled order gives a
    number that looks fine and means nothing. `ring_descriptors` is handed the order from the
    spec, which is written as the walk.

    Returns the total puckering amplitude Q (Å) for any ring size, plus the (q2, phi2) pair
    for 5- and 6-membered rings and, for 6-membered rings only, q3 and the theta/phi pair that
    names the conformation (theta ~ 0 or 180 is a chair, ~90 a boat or twist-boat).
    """
    X = _as_array(coords)
    n = len(X)
    X = X - X.mean(axis=0)

    j = np.arange(n)
    r1 = (X * np.sin(2 * np.pi * j / n)[:, None]).sum(axis=0)
    r2 = (X * np.cos(2 * np.pi * j / n)[:, None]).sum(axis=0)
    normal = np.cross(r1, r2)
    norm = np.linalg.norm(normal)
    if norm == 0:                      # a degenerate (collinear) ring has no mean plane
        return {"pucker_Q": float("nan")}
    normal = normal / norm
    z = X @ normal

    out: dict[str, float] = {"pucker_Q": float(np.sqrt(np.sum(z ** 2)))}

    if n in (5, 6):
        c = np.sqrt(2.0 / n) * float(np.sum(z * np.cos(4 * np.pi * j / n)))
        s = -np.sqrt(2.0 / n) * float(np.sum(z * np.sin(4 * np.pi * j / n)))
        out["pucker_q2"] = float(math.hypot(c, s))
        out["pucker_phi2_deg"] = float(np.degrees(math.atan2(s, c)) % 360.0)
    if n == 6:
        q3 = float(np.sum(z * ((-1.0) ** j)) / np.sqrt(n))
        out["pucker_q3"] = q3
        Q = out["pucker_Q"]
        out["pucker_theta_deg"] = float(np.degrees(math.acos(max(-1.0, min(1.0, q3 / Q))))) if Q else float("nan")
    return out


def ring_bond_lengths(coords) -> np.ndarray:
    """The n bond lengths around a ring given in connectivity order, closing the ring."""
    X = _as_array(coords)
    return np.linalg.norm(X - np.roll(X, -1, axis=0), axis=1)


def homa(coords, symbols) -> dict[str, float]:
    """HOMA for one ring, with its EN and GEO components.

    ``HOMA = 1 - (1/n) * sum_i alpha_i * (Ropt_i - R_i)^2``

    EN/GEO split the loss into bond *elongation* away from the optimal length and bond
    *alternation* around the ring. For a ring of one bond type this is Krygowski's exact
    decomposition. For a heterocycle with several bond types - pyridine has both CC and CN -
    each type is referenced to the mean of its OWN bonds, which is the usual generalisation;
    it is reported because the split is the informative part (a ring can lose aromaticity by
    stretching uniformly or by alternating, and those are different chemistry), but note it is
    a generalisation and not the 1993 single-type formula.

    Raises if any ring bond joins a pair of elements Krygowski never parameterised, rather
    than dropping the term: a HOMA summed over 5 of a ring's 6 bonds is not a HOMA, and
    silently returning one would be a number nobody could trace.
    """
    X = _as_array(coords)
    syms = [str(s).capitalize() for s in symbols]
    if len(syms) != len(X):
        raise ValueError(f"{len(X)} ring atoms but {len(syms)} element symbols")

    lengths = ring_bond_lengths(X)
    pairs = [frozenset((syms[i], syms[(i + 1) % len(syms)])) for i in range(len(syms))]
    missing = sorted({"-".join(sorted(p)) for p in pairs if p not in HOMA_PARAMETERS})
    if missing:
        raise ValueError(
            f"HOMA has no Krygowski parameters for the ring bond type(s) {', '.join(missing)}. "
            f"Parameterised pairs are: "
            f"{', '.join(sorted('-'.join(sorted(k)) for k in HOMA_PARAMETERS))}. "
            f"Set `homa: false` for this ring, or add the parameters with their source."
        )

    ropt = np.array([HOMA_PARAMETERS[p][0] for p in pairs])
    alpha = np.array([HOMA_PARAMETERS[p][1] for p in pairs])
    n = len(lengths)

    value = 1.0 - float(np.sum(alpha * (ropt - lengths) ** 2)) / n

    # Per bond type, the mean length of that type's own bonds.
    mean_by_pair = {p: float(np.mean(lengths[[i for i, q in enumerate(pairs) if q == p]]))
                    for p in set(pairs)}
    rav = np.array([mean_by_pair[p] for p in pairs])
    en = float(np.sum(alpha * (ropt - rav) ** 2)) / n
    geo = float(np.sum(alpha * (rav - lengths) ** 2)) / n
    return {"HOMA": value, "HOMA_EN": en, "HOMA_GEO": geo}


def bond_statistics(coords) -> dict[str, float]:
    """Plain descriptive statistics of the ring's bond lengths, and its alternation."""
    lengths = ring_bond_lengths(coords)
    # Bond-length alternation: the mean absolute difference between consecutive bonds around
    # the ring. Zero for a ring of equal bonds (benzene), large for a localised Kekule ring.
    bla = float(np.mean(np.abs(lengths - np.roll(lengths, -1))))
    # `shortest`/`longest`, not `min`/`max`: these names end up as column suffixes, and `min`,
    # `max` and `avg` are the CONDENSATION vocabulary - the statistic taken over a molecule's
    # conformers. `Ring_pyridine_bond_max` in a per-conformer table would be read by
    # `strip_condensation` as the stem `Ring_pyridine_bond` condensed by `max`, which is the
    # same trap as BUNNY's `_low_E` spelling. These are statistics over a ring's own bonds.
    return {"bond_mean": float(np.mean(lengths)), "bond_shortest": float(np.min(lengths)),
            "bond_longest": float(np.max(lengths)), "bond_range": float(np.ptp(lengths)),
            "bond_alternation": bla, "perimeter": float(np.sum(lengths))}


def ring_area(coords) -> float:
    """Area (Å^2) of the ring polygon projected onto its own best-fit plane.

    Projected rather than measured in 3D because a non-planar polygon has no single area; the
    projection is the one a reader means by "how big is this ring", and it is what shrinks when
    a ring puckers.
    """
    X = _as_array(coords)
    centroid, normal = best_plane(X)
    Y = X - centroid
    Y = Y - np.outer(Y @ normal, normal)              # drop the out-of-plane component
    total = sum(np.dot(np.cross(Y[i], Y[(i + 1) % len(Y)]), normal) for i in range(len(Y)))
    return float(abs(total) / 2.0)


def atom_height(coords, probe) -> float:
    """Distance (Å) of a probe atom from the ring's best-fit plane. Unsigned: see `best_plane`."""
    centroid, normal = best_plane(coords)
    return float(abs(np.dot(np.asarray(probe, dtype=float) - centroid, normal)))


def centroid_distance(coords, probe) -> float:
    """Distance (Å) from the ring centroid to a probe atom."""
    X = _as_array(coords)
    return float(np.linalg.norm(np.asarray(probe, dtype=float) - X.mean(axis=0)))


def interplanar_angle(coords_a, coords_b) -> float:
    """Angle (degrees) between two rings' best-fit planes, folded onto 0-90.

    Folded because the plane normals have arbitrary sign, so the raw angle between them is
    only defined up to 180 - x. 0 means coplanar and 90 perpendicular, whichever way the SVD
    happened to point.
    """
    _, na = best_plane(coords_a)
    _, nb = best_plane(coords_b)
    c = abs(float(np.dot(na, nb)))
    return float(np.degrees(math.acos(max(0.0, min(1.0, c)))))


def ring_descriptors(coords, symbols, with_homa: bool) -> dict[str, float]:
    """Every ring-level quantity for one ring, as a flat dict of bare names.

    The caller prefixes them with the ring's label; nothing here knows the label, so a ring
    cannot end up with another ring's name on its numbers.
    """
    out: dict[str, float] = {}
    out.update(bond_statistics(coords))
    out.update(planarity(coords))
    out.update(cremer_pople(coords))
    out["area"] = ring_area(coords)
    if with_homa:
        out.update(homa(coords, symbols))
    return out
