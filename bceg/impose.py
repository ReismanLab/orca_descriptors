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

A linear angle between two terminal atoms (I-Sm-I) starts from the line through the apex with the most
room instead (:func:`place_axial`), so that the ramp never pulls one end through a wrapping ligand's
donors.

A target that the ramp cannot bring within the snap tolerance, or a step that breaks or forms a bond, is
an :class:`ImposeError`: the caller rejects the embedding and tries the next seed, never forcing it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from rdkit import Chem
from scipy.optimize import minimize

from bceg import constraints as cons
from bceg.chem import ORGANIC_ELEMENTS

SNAP_TOLERANCE = {"bonds": 0.05, "angles": 3.0, "dihedrals": 3.0, "planes": 0.05}  # Angstrom, degrees, plane RMS
FINAL_TOLERANCE = {"bonds": 0.01, "angles": 0.5, "dihedrals": 0.5, "planes": 0.01}
K_BOND, K_13, K_REPULSION, K_TARGET = 1.0, 0.3, 10.0, 5.0
K_METAL_BOND = 5.0  # network and restraint strengths (arbitrary units)
REPULSION_MARGIN = 1.1  # non-bonded atoms are pushed apart below 1.1 x their covalent-radii sum (clash: 1.0 x)
DONOR_ANGLE = 109.5  # degrees: an M-donor-C angle squeezed by ETKDG opens to at least this along the ramp
MIN_LIGAND_ANGLE = 70.0  # degrees: an imposed angle's end is kept this far from the apex's other ligands
K_STIFF = 1000.0  # the restraint replacing a rigid snap inside a ring
SNAP_ROUNDS = 5
TERMINAL_SNAP = 30.0  # degrees: how far an angle with a single-atom end may be from its value before the snap  # a plane projection can disturb an angle that shares its atoms, and vice versa


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


def values(coords: np.ndarray, targets: dict) -> dict[str, float]:
    """Every target's current value (plane: its RMS), keyed ``"<kind>:<atoms>"``, for the run record."""
    return {f"{kind}:{t if kind == 'planes' else t['atoms']}": round(measure(coords, kind, t), 3)
            for kind, t in _all_targets(targets)}


def target_atoms(targets: dict) -> list[int]:
    return sorted({a for kind, t in _all_targets(targets) for a in (t if kind == "planes" else t["atoms"])})


def _within(coords: np.ndarray, targets: dict, tolerance: dict, mol: Chem.Mol | None = None) -> list[str]:
    """Descriptions of the targets further than ``tolerance`` from their value (none: all reached). With
    ``mol``, an angle with a single-atom end (I-Sm-I) may be TERMINAL_SNAP off: the snap moves that atom
    alone, and the connectivity is checked after it."""
    def allowed(kind, t):
        if mol is not None and kind == "angles" and any(mol.GetAtomWithIdx(a).GetDegree() == 1 for a in t["atoms"][::2]):
            return max(tolerance[kind], TERMINAL_SNAP)
        return tolerance[kind]

    return [f"{kind[:-1]} {t if kind == 'planes' else t['atoms']}: off by {deviation(coords, kind, t):.3f}"
            for kind, t in _all_targets(targets) if deviation(coords, kind, t) > allowed(kind, t)]


# --------------------------------------------------------------------------- the elastic network
class Network:
    """Bonds and 1-3 distances held at the embedded lengths, non-bonded pairs kept apart.

    Bonds to a metal are the exception: ETKDG knows no metal radii (Sm-O 1.8, Sm-I 1.9 Angstrom where
    2.5 and 3.1 are right), so they go from their embedded length to the covalent-radii sum along the
    ramp (:meth:`set_fraction`); the 1-3 distances that include one follow, an M-donor-C angle opening
    to at least DONOR_ANGLE. They are stiffer (K_METAL_BOND) than the other bonds, so that the repulsion
    of crowded ligands does not stretch them.
    The pre-optimisation then refines them."""

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
        target_bonds = set(free)
        around = {}  # (ligand, ligand) of an apex -> the apex: no spring, but kept apart (below)
        for t in targets.get("angles") or []:  # the ends move around the apex, against its other ligands
            ends, apex = {t["atoms"][0], t["atoms"][2]}, t["atoms"][1]
            nbrs = {n.GetIdx() for n in mol.GetAtomWithIdx(apex).GetNeighbors()}
            around |= {tuple(sorted((e, o))): apex for e in ends for o in nbrs if o != e}
        free |= set(around)
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
        metal = {a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() not in ORGANIC_ELEMENTS}
        index = {tuple(p): n for n, p in enumerate(self.springs.tolist())}
        self.metal_bonds = np.array([index[b] for b in bonds if set(b) & metal], dtype=int)
        self.ideal = radii[self.springs[self.metal_bonds, 0]] + radii[self.springs[self.metal_bonds, 1]]
        self.k[self.metal_bonds] = K_METAL_BOND
        self.embedded = self.d0[self.metal_bonds].copy()
        # 1-3 distances that include a bond to a metal (donor-M-donor, M-donor-C) follow those bonds; an
        # M-donor-C angle opens from the embedded one (ETKDG squeezes it along with the bond) to at least
        # DONOR_ANGLE, a donor-M-donor angle keeps its embedded value:
        # (1-3 spring, its two bond springs, start angle, end angle)
        self.through_metal = []
        metal_bond_rows = set(self.metal_bonds.tolist())
        for a, b in pairs13:
            for m in {nb.GetIdx() for nb in mol.GetAtomWithIdx(a).GetNeighbors()} \
                    & {nb.GetIdx() for nb in mol.GetAtomWithIdx(b).GetNeighbors()}:
                ba, bb = index.get(tuple(sorted((a, m)))), index.get(tuple(sorted((b, m))))
                if ba is not None and bb is not None and {ba, bb} & metal_bond_rows:
                    start_angle = cons.angle(coords, a, m, b)
                    end_angle = start_angle if m in metal else max(start_angle, DONOR_ANGLE)
                    self.through_metal.append((index[(a, b)], ba, bb, start_angle, end_angle))
                    if m not in metal:  # the donor's own angle (M-N-C): as stiff as the metal bond it opens with
                        self.k[index[(a, b)]] = K_METAL_BOND
        # every pair without a spring is kept apart, the freed ones included (an angle's end must not slide
        # onto the apex's other ligands); only a bond the targets set is left to its target
        near = set(map(tuple, self.springs.tolist())) | target_bonds
        iu = [(i, j) for i in range(n) for j in range(i + 1, n) if (i, j) not in near]
        self.nonbonded = np.array(iu, dtype=int).reshape(-1, 2)
        start = np.linalg.norm(coords[self.nonbonded[:, 0]] - coords[self.nonbonded[:, 1]], axis=1)
        # kept a margin beyond the covalent-radii sum (the clash criterion), or where they are when closer
        self.rmin = np.minimum(REPULSION_MARGIN * (radii[self.nonbonded[:, 0]] + radii[self.nonbonded[:, 1]]), start)
        # an imposed angle's end and the apex's other ligands: at least MIN_LIGAND_ANGLE apart as seen
        # from the apex, at the bond lengths the ramp is heading for (a cis angle at a metal is rarely
        # below 60-70 degrees)
        for row, (a, b) in enumerate(self.nonbonded.tolist()):
            apex = around.get((a, b))
            if apex is not None:
                ra, rb = radii[a] + radii[apex], radii[b] + radii[apex]
                self.rmin[row] = np.sqrt(ra ** 2 + rb ** 2 - 2 * ra * rb * np.cos(np.radians(MIN_LIGAND_ANGLE)))

    def start_from(self, x: np.ndarray) -> None:
        """The ramp of the metal bonds starts from their lengths in ``x`` (after :func:`place_axial`)."""
        d = x[self.springs[self.metal_bonds, 0]] - x[self.springs[self.metal_bonds, 1]]
        self.embedded = np.linalg.norm(d, axis=1)
        self.d0[self.metal_bonds] = self.embedded

    def set_fraction(self, frac: float) -> None:
        """Metal bonds ``frac`` of the way to the covalent-radii sum, the 1-3 distances through them along."""
        self.d0[self.metal_bonds] = self.embedded + frac * (self.ideal - self.embedded)
        for n, ba, bb, start_angle, end_angle in self.through_metal:
            r1, r2 = self.d0[ba], self.d0[bb]
            cos = np.cos(np.radians(start_angle + frac * (end_angle - start_angle)))
            self.d0[n] = np.sqrt(max(r1 ** 2 + r2 ** 2 - 2 * r1 * r2 * cos, 1e-6))

    def ideal_length(self, a: int, b: int) -> float | None:
        """The covalent-radii sum of a bond to a metal, None for any other pair."""
        hit = np.where((self.springs[self.metal_bonds] == sorted((a, b))).all(axis=1))[0]
        return float(self.ideal[hit[0]]) if len(hit) else None

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


def snap(mol: Chem.Mol, x: np.ndarray, targets: dict, network: Network | None = None) -> np.ndarray:
    """Every target to its exact value by rigid moves (a few rounds: a plane projection can disturb an angle
    sharing its atoms); a target inside a ring by a stiff restrained minimisation on ``network`` (built on
    ``x`` when not given) instead."""
    ring_snaps = []
    for _ in range(SNAP_ROUNDS):
        ring_snaps = []
        for kind, t in _all_targets(targets):
            snapped = _snap_one(mol, x, kind, t)
            if snapped is None:
                ring_snaps.append((kind, t))
            else:
                x = snapped
        if not _within(x, targets, FINAL_TOLERANCE):
            break
    if ring_snaps:
        network = network or Network(mol, x, targets)
        terms = {"bonds": _bond_term, "angles": _angle_term, "dihedrals": _dihedral_term}

        def stiff(y):
            e, grad = network.energy(y)
            for kind, t in ring_snaps:
                e += terms[kind](y, t["atoms"], t["value"], grad, k=K_STIFF)
            return e, grad
        x = _minimize(x, stiff)
    return x


def off_targets(x: np.ndarray, targets: dict, final: bool = False) -> list[str]:
    """The targets further than the snap (or, ``final``, the final) tolerance from their values."""
    return _within(x, targets, FINAL_TOLERANCE if final else SNAP_TOLERANCE)


# --------------------------------------------------------------------------- holding them in the pre-optimisation
LINEAR = 179.999  # an angle target from here up is held as a line


def holds(x: np.ndarray, targets: dict) -> dict:
    """What the pre-optimisation must keep of an imposed geometry ``x``, and nothing more, so that the bond
    lengths (Sm-I, Ni-N, ...) relax::

        {"planes": [{"atoms": [...], "normal": [nx, ny, nz]}],   each atom moves only within its plane
         "lines": [{"atoms": [i, j, k], "direction": [...]}],     a linear angle: the three only along it
         "bonds" / "angles" / "dihedrals": [{"atoms": [...], "value": v}]}   the other imposed values

    UMA keeps them exactly (ASE FixedPlane / FixedLine / FixInternals); xtb as stiff restraints
    (:func:`xtb_restraints`), followed by a :func:`snap`."""
    out: dict[str, Any] = {"planes": [], "lines": [], "bonds": [], "angles": [], "dihedrals": []}
    for group in targets.get("planes") or []:
        out["planes"].append({"atoms": list(group), "normal": _plane(x, group)[1].round(8).tolist()})
    for kind in ("bonds", "angles", "dihedrals"):
        for t in targets.get(kind) or []:
            if kind == "angles" and t["value"] >= LINEAR:
                i, _, k = t["atoms"]
                d = x[k] - x[i]
                out["lines"].append({"atoms": list(t["atoms"]), "direction": (d / np.linalg.norm(d)).round(8).tolist()})
            else:
                out[kind].append({"atoms": list(t["atoms"]), "value": float(t["value"])})
    return out


def _plane_dihedrals(x: np.ndarray, group: list[int]) -> list[tuple[list[int], float]]:
    """Dihedrals (0 or 180 degrees) that keep a planar group planar: one per atom beyond a reference
    triangle, each through the ordering whose two bond angles are furthest from 0 and 180 (a dihedral
    through a near-linear triple, e.g. trans ligands across a metal, is undefined)."""
    def sin(i, j, k):
        return np.sin(np.radians(cons.angle(x, i, j, k)))

    out = []
    for a in group:
        best = None
        for p in group:
            for q in group:
                for r in group:
                    if len({p, q, r, a}) < 4:
                        continue
                    score = min(sin(p, q, r), sin(q, r, a))
                    if best is None or score > best[0]:
                        best = (score, [p, q, r, a])
        if best is not None and best[1] not in [d for d, _ in out]:
            value = cons.dihedral(x, *best[1])
            out.append((best[1], 0.0 if abs(value) < 90 else 180.0))
    return out


def xtb_restraints(x: np.ndarray, hold: dict) -> list[str]:
    """xtb ``$constrain`` lines (1-based) for :func:`holds`: distances, angles (a line as an angle of 180,
    which xtb's angle restraint holds), dihedrals, and a plane as dihedrals of 0 or 180 degrees."""
    def ids(atoms):
        return ",".join(str(i + 1) for i in atoms)

    lines = [f"distance: {ids(t['atoms'])},{t['value']:.6f}" for t in hold["bonds"]]
    lines += [f"angle: {ids(t['atoms'])},{t['value']:.6f}" for t in hold["angles"]]
    lines += [f"angle: {ids(t['atoms'])},180.0" for t in hold["lines"]]
    lines += [f"dihedral: {ids(t['atoms'])},{t['value']:.6f}" for t in hold["dihedrals"]]
    for plane in hold["planes"]:
        lines += [f"dihedral: {ids(atoms)},{value:.1f}" for atoms, value in _plane_dihedrals(x, plane["atoms"])]
    return lines


AXIS_CANDIDATES = 2000  # directions sampled (a Fibonacci sphere) for the axis of a linear angle's ends


def _sphere(n: int) -> np.ndarray:
    k = np.arange(n) + 0.5
    phi, theta = np.arccos(1 - 2 * k / n), np.pi * (1 + 5 ** 0.5) * k
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def place_axial(mol: Chem.Mol, x: np.ndarray, targets: dict, network: Network | None = None) -> tuple[np.ndarray, list[dict]]:
    """For a linear angle whose two ends are terminal atoms (I-Sm-I): put the ends on the line through the
    apex with the most room, one at each end, at their current distances. The line is the one whose two
    end positions are furthest from every other atom. A ramp from an embedding with both ends on the
    same side of a wrapping ligand (O,N,N,O around Sm) would have to pull one end through the donors;
    moving two single atoms changes nothing else, and the ramp's repulsion and checks follow."""
    x = x.copy()
    placed = []
    for t in targets.get("angles") or []:
        i, j, k = t["atoms"]
        if t["value"] < LINEAR or mol.GetAtomWithIdx(i).GetDegree() != 1 or mol.GetAtomWithIdx(k).GetDegree() != 1:
            continue
        ri, rk = np.linalg.norm(x[i] - x[j]), np.linalg.norm(x[k] - x[j])
        if network is not None:  # at the bond length the ramp is heading for, not ETKDG's
            ri, rk = network.ideal_length(i, j) or ri, network.ideal_length(k, j) or rk
        rest = np.array([a for a in range(len(x)) if a not in (i, j, k)], dtype=int)
        axes = _sphere(AXIS_CANDIDATES)
        ends_i, ends_k = x[j] + ri * axes, x[j] - rk * axes
        room = np.minimum(np.linalg.norm(ends_i[:, None, :] - x[rest][None], axis=2).min(axis=1),
                          np.linalg.norm(ends_k[:, None, :] - x[rest][None], axis=2).min(axis=1))
        axis = axes[int(np.argmax(room))]
        # the axis and its opposite give the same line: take the assignment that moves the ends least
        sign = 1.0 if np.dot(x[i] - x[j], axis) - np.dot(x[k] - x[j], axis) >= 0 else -1.0
        before = {i: x[i].copy(), k: x[k].copy()}
        x[i], x[k] = x[j] + sign * ri * axis, x[j] - sign * rk * axis
        placed.append({"atoms": [i, j, k], "room": round(float(room.max()), 3),
                       "moved": {str(a): round(float(np.linalg.norm(x[a] - before[a])), 3) for a in (i, k)}})
    return x, placed


# --------------------------------------------------------------------------- the procedure
def impose(mol: Chem.Mol, coords: np.ndarray, targets: dict, held: list[int], steps: int = 10,
           check=None) -> tuple[np.ndarray, dict]:
    """Return (coords, info) with every target met within FINAL_TOLERANCE.

    ``check(coords) -> list[str]`` is run after every ramp step and at the end (connectivity); any
    problem, or a target out of reach of the snap, raises :class:`ImposeError`.
    """
    x = np.asarray(coords, dtype=float).copy()
    info: dict[str, Any] = {"steps": steps, "start": values(x, targets)}
    if not any(True for _ in _all_targets(targets)):
        return x, info | {"final": {}}
    network = Network(mol, x, targets)  # on the embedding: a clash the placement makes is pushed apart
    x, placed = place_axial(mol, x, targets, network)
    if placed:
        info["placed_axial"] = placed
    network.start_from(x)
    begin = {id(t): measure(x, kind, t) for kind, t in _all_targets(targets) if kind != "planes"}
    for s in range(1, steps + 1):
        frac = s / steps
        network.set_fraction(frac)
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
    info["after_ramp"] = values(x, targets)
    far = _within(x, targets, SNAP_TOLERANCE, mol)
    if far:
        raise ImposeError(f"after the {steps}-step ramp some targets are beyond the snap tolerance: {far}")

    x = snap(mol, x, targets, network)

    x = _minimize(x, network.energy, fixed=sorted(set(held) | set(target_atoms(targets))))  # settle the rest around them
    off = _within(x, targets, FINAL_TOLERANCE)
    if off:
        raise ImposeError(f"targets not met after the snap: {off}")
    problems = check(x) if check else []
    if problems:
        raise ImposeError(f"imposed geometry has a connectivity problem: {problems}")
    info["final"] = values(x, targets)
    return x, info
