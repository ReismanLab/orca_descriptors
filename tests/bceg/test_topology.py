from __future__ import annotations

import numpy as np
import pytest

from bceg import topology
from bceg.chem import embed
from bceg.structure import Structure
from bceg.transforms import BUILTIN_TRANSFORMS, Transformation


@pytest.fixture(scope="module")
def nih2():
    """(structure, coords, Ni index, [N, N], [H, H]): square-planar (bpy)NiH2 built from an embedding, the
    Ni-N set to 1.95 A and each hydride 1.45 A from Ni trans to an N."""
    s = Transformation("NiH2", BUILTIN_TRANSFORMS / "ni_x2.py", {"x": "H"}).apply("Cc1ccc(-c2ccccn2)nc1")
    c = embed(s.mol, 1, ff="none")
    el = s.elements
    ni = el.index("Ni")
    nbrs = [n.GetIdx() for n in s.mol.GetAtomWithIdx(ni).GetNeighbors()]
    ns, hs = [i for i in nbrs if el[i] == "N"], [i for i in nbrs if el[i] == "H"]
    unit = lambda v: v / np.linalg.norm(v)
    for n in ns:
        c[n] = c[ni] + 1.95 * unit(c[n] - c[ni])
    for h, n in zip(hs, ns[::-1]):
        c[h] = c[ni] - 1.45 * unit(c[n] - c[ni])
    return s, c, ni, ns, hs


def kinds(mol, coords, **kw):
    return [c.kind for c in topology.topology_changes(mol, coords, **kw)]


def test_intact_complex_has_no_change(nih2):
    s, c, *_ = nih2
    assert topology.topology_changes(s.mol, c) == []


def test_decoordinated_nitrogen(nih2):
    s, c, ni, ns, _ = nih2
    d = c.copy()
    d[ns[0]] = c[ni] + 3.0 * (c[ns[0]] - c[ni]) / np.linalg.norm(c[ns[0]] - c[ni])
    assert "decoordinated" in kinds(s.mol, d)


def test_reductive_elimination_of_h2_is_a_new_bond(nih2):
    """H2 still sigma-bound to Ni (both Ni-H short): only the H-H bond gives it away."""
    s, c, ni, _, hs = nih2
    d = c.copy()
    mid = (c[hs[0]] + c[hs[1]]) / 2
    u = (c[hs[0]] - c[hs[1]]) / np.linalg.norm(c[hs[0]] - c[hs[1]])
    d[hs[0]], d[hs[1]] = mid + 0.4 * u, mid - 0.4 * u
    changes = topology.topology_changes(s.mol, d)
    assert [ch.kind for ch in changes] == ["new_bond"]
    assert set(changes[0].atoms) == set(hs)
    assert topology.disallowed(changes)


def _ni_methane(ni_h: float, ni_c: float = 3.0):
    """[Ni].C with one C-H pointing at Ni: Ni at the origin, C on z, the other H tetrahedral."""
    s = Structure("[Ni].C")
    c = np.zeros((6, 3))
    c[1] = [0, 0, ni_c]
    c[2] = [0, 0, ni_c - 1.09]
    for k, phi in enumerate((0, 120, 240)):
        p = np.radians(phi)
        c[3 + k] = c[1] + 1.09 * np.array([0.943 * np.cos(p), 0.943 * np.sin(p), 0.333])
    assert abs(np.linalg.norm(c[2]) - ni_h) < 1e-6
    return s, c


def test_agostic_contact_is_allowed_by_default():
    s, c = _ni_methane(ni_h=1.91, ni_c=3.0)
    changes = topology.topology_changes(s.mol, c)
    assert [ch.kind for ch in changes] == ["agostic"]
    assert topology.disallowed(changes) == []
    assert topology.disallowed(changes, allow=()) == changes


def test_distant_ch_is_not_agostic():
    s, c = _ni_methane(ni_h=2.91, ni_c=4.0)
    assert topology.topology_changes(s.mol, c) == []


def test_new_metal_contact():
    s = Structure("[Ni].O")
    c = np.array([[0, 0, 0], [0, 0, 2.0], [0.76, 0, 2.59], [-0.76, 0, 2.59]], dtype=float)
    assert kinds(s.mol, c) == ["new_metal_contact"]


def test_nh_pointing_at_the_metal_is_judged_at_agostic_max():
    """[Ni].N with one N-H at Ni: 2.4 A (an imidazoline N-H next to Ni in real ensembles) is no change,
    2.0 A is a new metal contact (not agostic: the H is not on C)."""
    s = Structure("[Ni].N")
    for ni_h, expected in ((2.4, []), (2.0, ["new_metal_contact"])):
        n = np.array([0, 0, ni_h + 1.01])
        c = np.array([[0, 0, 0], n, [0, 0, ni_h], n + [0.95, 0, 0.34], n + [-0.48, 0.82, 0.34]], dtype=float)
        assert kinds(s.mol, c) == expected


def test_inverted_stereocentre():
    s = Structure("C[C@H](N)O")
    c = embed(s.mol, 1)
    assert topology.topology_changes(s.mol, c) == []
    assert kinds(s.mol, -c) == ["stereo"]  # the mirror image
    assert kinds(Structure("CC(N)O").mol, -c) == []  # unassigned centres are not judged


def test_atom_count_mismatch_raises(nih2):
    s, c, *_ = nih2
    with pytest.raises(ValueError, match="atoms"):
        topology.topology_changes(s.mol, c[:-1])


def test_check_params():
    assert topology.check_params(True, "t") == topology.DEFAULTS
    p = topology.check_params({"allow": "agostic", "agostic_max": 2}, "t")
    assert p["allow"] == ["agostic"] and p["agostic_max"] == 2.0
    with pytest.raises(ValueError, match="allow"):
        topology.check_params({"allow": ["nonsense"]}, "t")
    with pytest.raises(ValueError, match="unknown"):
        topology.check_params({"cutoff": 2}, "t")
    with pytest.raises(ValueError, match="positive"):
        topology.check_params({"agostic_max": -1}, "t")
