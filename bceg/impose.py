"""Move a start geometry onto imposed constraint values (``pre_refinement: constrained``), smoothly.

The targets (bceg.constraints.targets) are planes and bonds, angles and dihedrals with an explicit
``value``, e.g. I-Sm-I at 180 degrees or Ni and its first shell coplanar. Setting them in one jump on an
ETKDG geometry can push atoms through each other, so they are reached in three phases:

1. **ramp**: ``steps`` restrained minimisations, step s pulling every target a fraction s/steps of the
   way from its start value. The rest of the molecule follows through an elastic network built on the
   embedded geometry (every bond and every 1-3 distance held near its start length) plus a repulsion
   that keeps non-bonded atoms from coming closer than their covalent-radii sum. The network needs no
   force-field typing, so it works for any element (UFF has no type for Sm(III) and collapses NiH2).
   The connectivity is checked after every step.
2. **snap**: once every target is within ``SNAP_TOLERANCE`` the exact values are set by rigid moves:
   rotating (angles, dihedrals) or translating (bonds) the smaller side of the moving bond, and
   projecting each plane group onto its least-squares plane.
3. **settle**: the network is minimised once more with the held atoms fixed, so that the rest of the
   molecule absorbs the snap.

A target that the ramp cannot bring within the snap tolerance, or a step that breaks or forms a bond, is
an :class:`ImposeError`: the caller rejects the embedding and tries the next seed, never forcing it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from rdkit import Chem
from scipy.optimize import minimize

from bceg import constraints as cons

SNAP_TOLERANCE = {"bonds": 0.05, "angles": 3.0, "dihedrals": 3.0, "planes": 0.05}  # Angstrom, degrees, plane RMS
FINAL_TOLERANCE = {"bonds": 0.01, "angles": 0.5, "dihedrals": 0.5, "planes": 0.01}
K_BOND, K_13, K_REPULSION, K_TARGET = 1.0, 0.3, 10.0, 5.0  # network and restraint strengths (arbitrary units)
K_STIFF = 1000.0  # the restraint replacing a rigid snap inside a ring
SNAP_ROUNDS = 5  # a plane projection can disturb an angle that shares its atoms, and vice versa


class ImposeError(RuntimeError):
    """The targets could not be reached cleanly from this embedding."""


# --------------------------------------------------------------------------- measuring
def plane_rms(coords: np.ndarray, group: list[int]) -> float:
    pts = coords[group] - coords[group].mean(axis=0)
    return float(np.sqrt(np.linalg.svd(pts, compute_uv=False)[-1] ** 2 / len(group)))


def _plane(coords: np.ndarray, group: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """(centroid, unit normal) of the least-squares plane through the group."""
    centre = coords[group].mean(axis=0)
    return centre, np.linalg.svd(coords[group] - centre)[2][-1]


def _wrap(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def measure(coords: np.ndarray, kind: str, target: Any) -> float:
    if kind == "planes":
        return plane_rms(coords, target)
    return cons.MEASURE[len(target["atoms"])](coords, *target["atoms"])


def deviation(coords: np.ndarray, kind: str, target: Any) -> float:
    if kind == "planes":
        return plane_rms(coords, target)
    value = measure(coords, kind, target)
    return abs(_wrap(value - target["value"])) if kind == "dihedrals" else abs(value - target["value"])


def _all_targets(targets: dict):
    for kind in ("bonds", "angles", "dihedrals", "planes"):
        for t in targets.get(kind) or []:
            yield kind, t


def _within(coords: np.ndarray, targets: dict, tolerance: dict) -> list[str]:
    """Descriptions of the targets further than ``tolerance`` from their value (none: all reached)."""
    return [f"{kind[:-1]} {t if kind == 'planes' else t['atoms']}: off by {deviation(coords, kind, t):.3f}"
            for kind, t in _all_targets(targets) if deviation(coords, kind, t) > tolerance[kind]]


# --------------------------------------------------------------------------- the elastic network
class Network:
    """Bonds and 1-3 distances held at the embedded lengths, non-bonded pairs kept apart."""

    def __init__(self, mol: Chem.Mol, coords: np.ndarray, targets: dict) -> None:
        n = mol.GetNumAtoms()
        bonds = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol.GetBonds()}
        pairs13 = set()
        for atom in mol.GetAtoms():
            nbrs = [a.GetIdx() for a in atom.GetNeighbors()]
            pairs13 |= {tuple(sorted((a, b))) for i, a in enumerate(nbrs) for b in nbrs[i + 1:]}
        pairs13 -= bonds
        # distances that the targets change are left to them: the bonds themselves, the ends of an angle,
        # the 1-3 distances inside a plane group (planarising a tetrahedron changes its angles)
        free = {tuple(sorted(t["atoms"])) for t in targets.get("bonds") or []}
        for t in targets.get("angles") or []:  # the ends move around the apex, against its other ligands
            ends, apex = {t["atoms"][0], t["atoms"][2]}, t["atoms"][1]
            nbrs = {n.GetIdx() for n in mol.GetAtomWithIdx(apex).GetNeighbors()}
            free |= {tuple(sorted((e, o))) for e in ends for o in nbrs if o != e}
        for group in targets.get("planes") or []:
            g = set(group)
            free |= {p for p in pairs13 if set(p) <= g}
        bonds, pairs13 = sorted(bonds - free), sorted(pairs13 - free)
        self.springs = np.array(bonds + pairs13, dtype=int).reshape(-1, 2)
        self.k = np.array([K_BOND] * len(bonds) + [K_13] * len(pairs13))
        d = coords[self.springs[:, 0]] - coords[self.springs[:, 1]]
        self.d0 = np.linalg.norm(d, axis=1)
        pt = Chem.GetPeriodicTable()
        radii = np.array([pt.GetRcovalent(a.GetAtomicNum()) for a in mol.GetAtoms()])
        near = set(map(tuple, self.springs.tolist())) | free
        iu = [(i, j) for i in range(n) for j in range(i + 1, n) if (i, j) not in near]
        self.nonbonded = np.array(iu, dtype=int).reshape(-1, 2)
        start = np.linalg.norm(coords[self.nonbonded[:, 0]] - coords[self.nonbonded[:, 1]], axis=1)
        # never closer than the covalent-radii sum, nor than they already are when that is closer
        self.rmin = np.minimum(radii[self.nonbonded[:, 0]] + radii[self.nonbonded[:, 1]], start)

    def energy(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        grad = np.zeros_like(x)
        d = x[self.springs[:, 0]] - x[self.springs[:, 1]]
        r = np.linalg.norm(d, axis=1)
        e = float((self.k * (r - self.d0) ** 2).sum())
        g = (2 * self.k * (r - self.d0) / np.maximum(r, 1e-9))[:, None] * d
        np.add.at(grad, self.springs[:, 0], g)
        np.add.at(grad, self.springs[:, 1], -g)
        d = x[self.nonbonded[:, 0]] - x[self.nonbonded[:, 1]]
        r = np.linalg.norm(d, axis=1)
        close = r < self.rmin
        if close.any():
            dc, rc, rm = d[close], r[close], self.rmin[close]
            e += float((K_REPULSION * (rm - rc) ** 2).sum())
            g = (-2 * K_REPULSION * (rm - rc) / np.maximum(rc, 1e-9))[:, None] * dc
            np.add.at(grad, self.nonbonded[close, 0], g)
            np.add.at(grad, self.nonbonded[close, 1], -g)
        return e, grad


# --------------------------------------------------------------------------- restraints on the targets
def _bond_term(x, atoms, value, grad, k=K_TARGET):
    i, j = atoms
    d = x[i] - x[j]
    r = np.linalg.norm(d)
    g = 2 * k * (r - value) / max(r, 1e-9) * d
    grad[i] += g
    grad[j] -= g
    return k * (r - value) ** 2


def _angle_term(x, atoms, value, grad, k=K_TARGET):
    """K (d_ik - D)^2 with D = sqrt(r1^2 + r2^2 - 2 r1 r2 cos value), the i...k distance the angle gives for
    the current bond lengths: as stiff as a bond restraint and smooth at 180 degrees (D = r1 + r2), where
    the gradient of the angle itself is singular."""
    i, j, l_ = atoms
    u, v, w = x[i] - x[j], x[l_] - x[j], x[i] - x[l_]
    r1, r2, d = np.linalg.norm(u), np.linalg.norm(v), np.linalg.norm(w)
    c = np.cos(np.radians(value))
    big = np.sqrt(max(r1 ** 2 + r2 ** 2 - 2 * r1 * r2 * c, 1e-12))
    diff = d - big
    f = 2 * k * diff
    g_d = f * w / max(d, 1e-9)                                   # d(d)/dx_i = w/d, d(d)/dx_k = -w/d
    g_r1 = -f * (r1 - r2 * c) / big * u / max(r1, 1e-9)         # -dD/dr1 * dr1/dx_i
    g_r2 = -f * (r2 - r1 * c) / big * v / max(r2, 1e-9)
    grad[i] += g_d + g_r1
    grad[l_] += -g_d + g_r2
    grad[j] -= g_r1 + g_r2
    return k * diff ** 2


def _dihedral_term(x, atoms, value, grad, k=K_TARGET):
    """K (1 - cos(phi - value)), with the analytic gradient of phi (Blondel & Karplus, J. Comput. Chem. 1996)."""
    i, j, l_, m = atoms
    f, g, h = x[i] - x[j], x[j] - x[l_], x[m] - x[l_]
    a, b = np.cross(f, g), np.cross(h, g)
    aa, bb, gn = np.dot(a, a), np.dot(b, b), np.linalg.norm(g)
    if aa < 1e-10 or bb < 1e-10:
        return 0.0  # a collinear triple: the dihedral is undefined, the network carries it
    phi = np.arctan2(np.dot(np.cross(b, a), g) / gn, np.dot(a, b))
    diff = phi - np.radians(value)
    s = k * np.sin(diff)
    fg, hg = np.dot(f, g), np.dot(h, g)
    d_i = -gn / aa * a
    d_m = gn / bb * b
    d_j = gn / aa * a + fg / (aa * gn) * a - hg / (bb * gn) * b
    d_l = -gn / bb * b - fg / (aa * gn) * a + hg / (bb * gn) * b
    for idx, d in zip((i, j, l_, m), (d_i, d_j, d_l, d_m)):
        grad[idx] += s * d
    return k * (1 - np.cos(diff))


def _plane_term(x, group, heights, normal, centre, grad):
    """Each atom of the group at its target signed height above the (fixed) reference plane."""
    e = 0.0
    for a, h in zip(group, heights):
        diff = np.dot(x[a] - centre, normal) - h
        e += K_TARGET * diff ** 2
        grad[a] += 2 * K_TARGET * diff * normal
    return e


def _minimize(x0: np.ndarray, energy, fixed: list[int] | None = None) -> np.ndarray:
    n = len(x0)
    mask = np.ones(n, dtype=bool)
    if fixed:
        mask[fixed] = False

    def f(flat):
        x = x0.copy()
        x[mask] = flat.reshape(-1, 3)
        e, grad = energy(x)
        return e, grad[mask].ravel()

    res = minimize(f, x0[mask].ravel(), jac=True, method="L-BFGS-B", options={"maxiter": 2000, "gtol": 1e-6})
    x = x0.copy()
    x[mask] = res.x.reshape(-1, 3)
    return x


# --------------------------------------------------------------------------- rigid snaps
def _side(mol: Chem.Mol, start: int, blocked: int) -> set[int] | None:
    """Atoms reached from ``start`` without crossing ``blocked``; None if the bond is in a ring."""
    if _reaches(mol, start, blocked):
        return None
    seen, stack = {start}, [start]
    while stack:
        for nb in mol.GetAtomWithIdx(stack.pop()).GetNeighbors():
            a = nb.GetIdx()
            if a != blocked and a not in seen:
                seen.add(a)
                stack.append(a)
    return seen


def _reaches(mol: Chem.Mol, start: int, blocked: int) -> bool:
    """True if ``blocked`` is reachable from ``start`` without the direct start-blocked bond (a ring)."""
    seen, stack = {start}, [start]
    while stack:
        cur = stack.pop()
        for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
            a = nb.GetIdx()
            if cur == start and a == blocked:
                continue
            if a == blocked:
                return True
            if a not in seen:
                seen.add(a)
                stack.append(a)
    return False


def _rotate(x: np.ndarray, atoms: set[int], origin: np.ndarray, axis: np.ndarray, degrees: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    t = np.radians(degrees)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rot = np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * k @ k
    x = x.copy()
    idx = sorted(atoms)
    x[idx] = (x[idx] - origin) @ rot.T + origin
    return x


def _smaller_side(mol: Chem.Mol, a: int, b: int) -> tuple[int, set[int]] | None:
    """For the bond a-b: (pivot, atoms to move), the smaller of b's side and a's side; None in a ring."""
    side_b, side_a = _side(mol, b, a), _side(mol, a, b)
    if side_b is None or side_a is None:
        return None
    return (a, side_b) if len(side_b) <= len(side_a) else (b, side_a)


def _snap_one(mol: Chem.Mol, x: np.ndarray, kind: str, t: Any) -> np.ndarray | None:
    """Exact value by a rigid move; None when the moving bond is in a ring."""
    if kind == "planes":
        centre, normal = _plane(x, t)
        x = x.copy()
        x[t] -= np.outer((x[t] - centre) @ normal, normal)
        return x
    atoms, value = t["atoms"], t["value"]
    if kind == "bonds":
        i, j = atoms
        moved = _smaller_side(mol, i, j)
        if moved is None:
            return None
        pivot, side = moved
        other = j if pivot == i else i
        u = (x[other] - x[pivot]) / np.linalg.norm(x[other] - x[pivot])
        x = x.copy()
        x[sorted(side)] += (value - np.linalg.norm(x[other] - x[pivot])) * u
        return x
    if kind == "angles":
        i, j, k = atoms
        side_k, side_i = _side(mol, k, j), _side(mol, i, j)
        if side_k is None or side_i is None:
            return None
        moving, fixed_end = (k, i) if len(side_k) <= len(side_i) else (i, k)
        side = side_k if moving == k else side_i
        u, v = x[fixed_end] - x[j], x[moving] - x[j]
        axis = np.cross(u, v)
        if np.linalg.norm(axis) < 1e-6:  # already (anti)linear: any perpendicular axis
            axis = np.cross(u, [1.0, 0, 0]) if abs(u[0]) < 0.9 * np.linalg.norm(u) else np.cross(u, [0, 1.0, 0])
        current = cons.angle(x, fixed_end, j, moving)
        return _rotate(x, side, x[j], axis, value - current)
    i, j, k, l = atoms  # dihedrals: rotate about j-k
    moved = _smaller_side(mol, j, k)
    if moved is None:
        return None
    pivot, side = moved
    current = cons.dihedral(x, i, j, k, l)
    delta = _wrap(value - current)
    # rotating l's side (k's side of j-k) by +delta about j->k raises the dihedral by delta
    if pivot == j:
        return _rotate(x, side, x[k], x[k] - x[j], delta)
    return _rotate(x, side, x[j], x[k] - x[j], -delta)


# --------------------------------------------------------------------------- the procedure
def impose(mol: Chem.Mol, coords: np.ndarray, targets: dict, held: list[int], steps: int = 10,
           check=None) -> tuple[np.ndarray, dict]:
    """Return (coords, info) with every target met within FINAL_TOLERANCE.

    ``check(coords) -> list[str]`` is run after every ramp step and at the end (connectivity); any
    problem, or a target out of reach of the snap, raises :class:`ImposeError`.
    """
    x = np.asarray(coords, dtype=float).copy()
    start = {f"{kind}:{t if kind == 'planes' else t['atoms']}": round(measure(x, kind, t), 3)
             for kind, t in _all_targets(targets)}
    info: dict[str, Any] = {"steps": steps, "start": start}
    if not any(True for _ in _all_targets(targets)):
        return x, info | {"final": {}}
    network = Network(mol, x, targets)
    begin = {id(t): measure(x, kind, t) for kind, t in _all_targets(targets) if kind != "planes"}
    for s in range(1, steps + 1):
        frac = s / steps
        planes = []
        for group in targets.get("planes") or []:
            centre, normal = _plane(x, group)
            heights = (x[group] - centre) @ normal * (steps - s) / (steps - s + 1)
            planes.append((group, heights, normal, centre))

        def energy(y, frac=frac, planes=planes):
            e, grad = network.energy(y)
            for t in targets.get("bonds") or []:
                b = begin[id(t)]
                e += _bond_term(y, t["atoms"], b + frac * (t["value"] - b), grad)
            for t in targets.get("angles") or []:
                b = begin[id(t)]
                e += _angle_term(y, t["atoms"], b + frac * (t["value"] - b), grad)
            for t in targets.get("dihedrals") or []:
                b = begin[id(t)]
                e += _dihedral_term(y, t["atoms"], b + frac * _wrap(t["value"] - b), grad)
            for group, heights, normal, centre in planes:
                e += _plane_term(y, group, heights, normal, centre, grad)
            return e, grad

        x = _minimize(x, energy)
        problems = check(x) if check else []
        if problems:
            raise ImposeError(f"ramp step {s}/{steps} changed the connectivity: {problems}")
    info["after_ramp"] = {key: round(measure(x, kind, t), 3)
                          for key, (kind, t) in zip(start, _all_targets(targets))}
    far = _within(x, targets, SNAP_TOLERANCE)
    if far:
        raise ImposeError(f"after the {steps}-step ramp some targets are beyond the snap tolerance: {far}")

    ring_snaps = []
    for _ in range(SNAP_ROUNDS):
        for kind, t in _all_targets(targets):
            snapped = _snap_one(mol, x, kind, t)
            if snapped is None:
                ring_snaps.append((kind, t))
            else:
                x = snapped
        if not _within(x, targets, FINAL_TOLERANCE):
            break
    if ring_snaps:  # a target inside a ring: a stiff restrained minimisation instead of a rigid move
        terms = {"bonds": _bond_term, "angles": _angle_term, "dihedrals": _dihedral_term}

        def stiff(y):
            e, grad = network.energy(y)
            for kind, t in ring_snaps:
                e += terms[kind](y, t["atoms"], t["value"], grad, k=K_STIFF)
            return e, grad
        x = _minimize(x, stiff)

    target_atoms = {a for kind, t in _all_targets(targets) for a in (t if kind == "planes" else t["atoms"])}
    x = _minimize(x, network.energy, fixed=sorted(set(held) | target_atoms))  # settle the rest around them
    off = _within(x, targets, FINAL_TOLERANCE)
    if off:
        raise ImposeError(f"targets not met after the snap: {off}")
    problems = check(x) if check else []
    if problems:
        raise ImposeError(f"imposed geometry has a connectivity problem: {problems}")
    info["final"] = {key: round(measure(x, kind, t), 3) for key, (kind, t) in zip(start, _all_targets(targets))}
    return x, info
