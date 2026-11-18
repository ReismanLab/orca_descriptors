"""Ring-level geometry and the HOMA aromaticity index (``desc_utils.rings``).

The failure modes these guard against are all of the kind where a wrong number looks
entirely reasonable. HOMA on a saturated ring returns a confident large negative value;
Cremer-Pople puckering coordinates computed over atoms in FILE order rather than ring-walk
order return a plausible amplitude for a ring that is actually flat; an interplanar angle
taken from raw SVD normals flips between x and 180-x depending on which way the
decomposition happened to point, so the same pair of rings gives two different answers on
two conformers of one molecule. None of these raises.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from desc_utils import rings


def regular_ring(n: int, bond: float) -> np.ndarray:
    """A planar regular n-gon whose edges are exactly `bond` long."""
    r = bond / (2 * math.sin(math.pi / n))
    a = 2 * np.pi * np.arange(n) / n
    return np.c_[r * np.cos(a), r * np.sin(a), np.zeros(n)]


def test_an_ideal_aromatic_ring_scores_exactly_one():
    """HOMA is defined so a ring of optimal bonds is 1.0. If the parameters were mistyped or
    the formula's 1/n misplaced, this lands near but not on 1 and nothing would notice."""
    X = regular_ring(6, rings.HOMA_PARAMETERS[frozenset(("C", "C"))][0])
    out = rings.homa(X, ["C"] * 6)
    assert out["HOMA"] == pytest.approx(1.0, abs=1e-12)
    assert out["HOMA_EN"] == pytest.approx(0.0, abs=1e-12)
    assert out["HOMA_GEO"] == pytest.approx(0.0, abs=1e-12)


def test_bond_alternation_shows_up_as_geo_and_not_as_en():
    """The EN/GEO split is the informative half of HOMA: a ring that alternates must load GEO,
    a ring that is uniformly stretched must load EN. Swapping the two expressions leaves HOMA
    itself correct, so only this distinguishes them."""
    th, p, pts = 0.0, np.zeros(3), []
    for L in [1.34, 1.45] * 3:                      # a closed Kekule-localised hexagon
        pts.append(p.copy())
        p = p + np.array([math.cos(th), math.sin(th), 0.0]) * L
        th += math.radians(60)
    alternating = rings.homa(np.array(pts), ["C"] * 6)
    assert alternating["HOMA"] < 0.5
    assert alternating["HOMA_GEO"] > 10 * alternating["HOMA_EN"]

    stretched = rings.homa(regular_ring(6, 1.45), ["C"] * 6)     # uniform, no alternation
    assert stretched["HOMA_GEO"] == pytest.approx(0.0, abs=1e-12)
    assert stretched["HOMA_EN"] > 0.5


def test_homa_refuses_a_bond_type_it_has_no_parameters_for():
    """Dropping the unparameterised term would return a HOMA summed over 5 of 6 bonds - a
    number with the right name, the right range, and no meaning."""
    with pytest.raises(ValueError, match="no Krygowski parameters"):
        rings.homa(regular_ring(5, 1.4), ["C", "C", "C", "C", "Si"])


def test_a_flat_ring_has_no_pucker_and_a_chair_is_recognised():
    """Cremer-Pople theta near 180 (or 0) is a chair and near 90 a boat. A sign error in the
    q3 alternating sum turns one into the other."""
    assert rings.cremer_pople(regular_ring(6, 1.4))["pucker_Q"] == pytest.approx(0.0, abs=1e-12)

    a = 2 * np.pi * np.arange(6) / 6
    chair = np.c_[1.46 * np.cos(a), 1.46 * np.sin(a), 0.25 * (-1.0) ** np.arange(6)]
    out = rings.cremer_pople(chair)
    assert out["pucker_Q"] == pytest.approx(0.6124, abs=1e-3)
    assert out["pucker_q2"] == pytest.approx(0.0, abs=1e-9)
    assert min(out["pucker_theta_deg"], 180 - out["pucker_theta_deg"]) == pytest.approx(0.0, abs=1e-6)


def test_puckering_is_measured_around_the_ring_and_not_in_file_order():
    """The phase angles are defined against position around the ring, so the atom order has to
    be the connectivity walk. Shuffled, a genuinely puckered ring reports a different
    amplitude - plausible, and wrong - which is why the spec's ring atoms are written as the
    walk and this is asserted rather than described."""
    a = 2 * np.pi * np.arange(6) / 6
    chair = np.c_[1.46 * np.cos(a), 1.46 * np.sin(a), 0.25 * (-1.0) ** np.arange(6)]
    walked = rings.cremer_pople(chair)["pucker_Q"]
    shuffled = rings.cremer_pople(chair[[0, 2, 4, 1, 3, 5]])["pucker_Q"]
    assert walked == pytest.approx(0.6124, abs=1e-3)
    assert abs(shuffled - walked) > 0.1


def test_the_interplanar_angle_does_not_depend_on_the_svd_sign():
    """Plane normals from an SVD have arbitrary sign, so the raw angle between two rings is
    only defined up to 180-x; an unfolded value flips between conformers of one molecule and
    reads as a real conformational change."""
    A = regular_ring(6, 1.4)
    t = math.radians(35)
    rot = np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])
    B = A @ rot.T + np.array([5.0, 0.0, 0.0])
    assert rings.interplanar_angle(A, B) == pytest.approx(35.0, abs=1e-6)
    assert rings.interplanar_angle(A, B[::-1]) == pytest.approx(35.0, abs=1e-6)
    assert rings.interplanar_angle(A, A) == pytest.approx(0.0, abs=1e-9)


def test_the_projected_area_is_the_analytic_one_and_shrinks_when_the_ring_puckers():
    """A non-planar polygon has no single area; projecting onto the ring's own plane is the
    one a reader means, and it must reproduce the textbook hexagon exactly when flat."""
    bond = 1.4
    X = regular_ring(6, bond)
    r = bond / (2 * math.sin(math.pi / 6))
    assert rings.ring_area(X) == pytest.approx(3 * math.sqrt(3) / 2 * r ** 2, rel=1e-12)

    # A real pucker holds the BOND lengths and folds the ring, so the in-plane footprint has
    # to contract. Simply displacing z at fixed x,y is not a pucker - it stretches every bond -
    # and leaves the projected area untouched, which is correct and would pass a weaker test.
    h = 0.3
    puckered = regular_ring(6, math.sqrt(bond ** 2 - (2 * h) ** 2))
    puckered[:, 2] = h * (-1.0) ** np.arange(6)
    assert rings.ring_bond_lengths(puckered) == pytest.approx([bond] * 6, abs=1e-12)
    assert rings.ring_area(puckered) < rings.ring_area(X)


def test_a_probe_on_the_ring_axis_is_measured_from_the_plane_not_the_atoms():
    """`atom_height` must be the distance to the PLANE and `centroid_distance` to the centroid;
    for an on-axis probe they coincide, which is the only case that cannot tell them apart -
    so an off-axis probe is checked too."""
    X = regular_ring(6, 1.4)
    assert rings.atom_height(X, [0.0, 0.0, 1.7]) == pytest.approx(1.7, abs=1e-12)
    assert rings.centroid_distance(X, [0.0, 0.0, 1.7]) == pytest.approx(1.7, abs=1e-12)

    off_axis = [1.0, 0.0, 1.7]
    assert rings.atom_height(X, off_axis) == pytest.approx(1.7, abs=1e-12)
    assert rings.centroid_distance(X, off_axis) == pytest.approx(math.hypot(1.0, 1.7), abs=1e-12)


def test_a_ring_without_the_homa_flag_gets_shape_descriptors_and_no_homa():
    """An oxazoline is saturated; HOMA against aromatic reference lengths returns a large
    negative number that would sit in a model next to the pyridine's HOMA looking like the
    same measurement."""
    X = regular_ring(5, 1.5)
    with_flag = rings.ring_descriptors(X, ["C", "N", "C", "C", "O"], with_homa=False)
    assert "HOMA" not in with_flag
    assert {"planarity_rms", "planarity_worst", "pucker_Q", "area", "bond_alternation"} <= set(with_flag)
