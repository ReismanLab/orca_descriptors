"""Which conformers of an already-computed library are described (the ``prune`` spec block).

This is the **laptop-side** prune, and it is not stage 5. Stage 5
(``desc_utils/submit/tasks.py::energy_window``, ``docs/stages/5_pruning.md``) decides which
conformers get an ORCA or g-xTB *job*, before anything is computed, and is hashed into the
library's identity. This module decides which of the conformers a finished library already holds
are *described*, and it is a property of the descriptor run, not of the library: the same library
can be extracted twice under two windows, and both are legitimate.

Two criteria, applied in this order:

``energy_window_kcal_mol``
    per molecule, against that molecule's own lowest conformer, on ``energy_column``.
``rmsd_threshold_angstrom``
    greedy leader clustering **in ascending energy order**, so the lowest-energy conformer of a
    group of duplicates is the one kept - it is visited first and nothing can displace it. The
    distance is the symmetric heavy-atom RMSD, minimised over the automorphisms of the heavy-atom
    graph, so a ring flip that relabels equivalent atoms does not count as a different structure.

The RMSD definition, the pair pre-filters and the energy gate are ported from
``ConfAnalysis-v1/output/dft_cost/LNiH2_nih2conv/dedupe_counts.py``, the script behind that
project's own threshold study, so a count produced here can be compared with the ladder measured
there. The port reads one geometry per ``conf_N`` folder instead of frames of a multi-frame xyz,
and takes the energies from the numbering table rather than from the comment lines, which is what
lets several libraries be pooled into one table and pruned together.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HARTREE_TO_KCAL = 627.509474

#: Covalent radii (Angstrom) used to perceive the heavy-atom graph, as in the source script.
RCOV = {"H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Si": 1.11, "P": 1.07,
        "S": 1.05, "Cl": 1.02, "Br": 1.20, "I": 1.39, "Ni": 1.24}
MAX_AUTOMORPHISMS = 1000

#: Default energy gate: pairs further apart than this in the pruning energy are never compared.
#: Measured on this project's own data - the largest energy gap of a pair below 0.02 A is
#: 0.04 kcal/mol - so it is safe at the tight thresholds and is refused above
#: :data:`ENERGY_GATE_VALID_BELOW` rather than being extrapolated silently.
ENERGY_GATE_KCAL = 0.2
ENERGY_GATE_VALID_BELOW = 0.02

KEPT = "kept"
ABOVE_WINDOW = "above_energy_window"
DUPLICATE = "duplicate"
NO_ENERGY = "no_energy"


# --------------------------------------------------------------------------- geometry
def read_xyz(path) -> tuple[np.ndarray, np.ndarray]:
    """``(symbols, coordinates)`` of the first frame of an xyz file."""
    lines = open(path, encoding="utf-8").read().splitlines()
    n = int(lines[0])
    rows = [ln.split() for ln in lines[2:2 + n]]
    return (np.array([r[0] for r in rows]),
            np.array([[float(x) for x in r[1:4]] for r in rows]))


def automorphisms(sym, X, max_n: int = MAX_AUTOMORPHISMS):
    """Heavy-atom indices and the automorphisms of the heavy-atom graph.

    The graph is perceived from the geometry (a bond is a distance below 1.2x the sum of covalent
    radii), never from a SMILES: the point is to recognise that two geometries differ only by a
    relabelling of equivalent atoms, and bond orders play no part in that. Returned as an
    ``(m x n_heavy)`` array of permutations with the identity first.
    """
    from rdkit import Chem
    sym = np.asarray(sym)
    heavy = np.flatnonzero(sym != "H")
    D = np.linalg.norm(X[heavy][:, None] - X[heavy][None], axis=-1)
    r = np.array([RCOV.get(s, 1.5) for s in sym[heavy]])
    m = Chem.RWMol()
    for s in sym[heavy]:
        a = Chem.Atom(str(s))
        a.SetNoImplicit(True)
        m.AddAtom(a)
    for a, b in zip(*np.triu_indices(len(heavy), 1)):
        if D[a, b] < 1.2 * (r[a] + r[b]):
            m.AddBond(int(a), int(b), Chem.BondType.SINGLE)
    m = m.GetMol()
    m.UpdatePropertyCache(strict=False)
    P = np.array(m.GetSubstructMatches(m, uniquify=False, useChirality=False, maxMatches=max_n),
                 dtype=np.int32)
    ident = np.arange(len(heavy))
    if len(P) == 0:
        P = ident[None]
    P = np.vstack([ident[None], P[~(P == ident).all(axis=1)]])          # identity first
    return heavy, P[:max_n]


def candidate_pairs(X, E_kcal, heavy, tmax, gate_kcal, chunk: int = 400_000):
    """Index pairs ``i < j`` that can have a symmetric heavy-atom RMSD below ``tmax``.

    Three filters in this order. The first is physical (the energy gate above). The other two are
    rigorous and invariant under rotation and under any permutation of the atoms: if the optimally
    superposed heavy-atom deviations are ``d_i`` with rms ``r`` over ``n`` heavy atoms, then the
    sorted distances to the heavy-atom centroid differ by at most ``sqrt(n) * r`` in L2 (sorting is
    non-expansive), and the sorted vector of all pairwise heavy-atom distances by at most
    ``sqrt(2n - 2) * sqrt(n) * r``. So a pair failing either test cannot be a duplicate, and only
    the survivors get an exact RMSD.
    """
    H = X[:, heavy]
    H = H - H.mean(axis=1, keepdims=True)
    n = len(heavy)
    rad = np.sort(np.linalg.norm(H, axis=2), axis=1)
    iu = np.triu_indices(n, 1)
    dmat = np.empty((len(H), len(iu[0])))
    for s in range(0, len(H), 500):                       # chunked: the full n x n x n x 3 is large
        D = np.linalg.norm(H[s:s + 500, :, None] - H[s:s + 500, None], axis=3)
        dmat[s:s + 500] = np.sort(D[:, iu[0], iu[1]], axis=1)
    b1, b2 = np.sqrt(n) * tmax, np.sqrt(2 * n - 2) * np.sqrt(n) * tmax
    s1, s2 = (rad ** 2).sum(axis=1), (dmat ** 2).sum(axis=1)

    o = np.argsort(E_kcal, kind="stable")
    Es = np.asarray(E_kcal)[o]
    if gate_kcal is None:
        I = np.repeat(o, len(o))
        J = np.tile(o, len(o))
        keep = I < J
        I, J = I[keep], J[keep]
    else:
        hi = np.searchsorted(Es, Es + gate_kcal, side="right")
        counts = hi - np.arange(len(Es)) - 1
        i0 = np.repeat(np.arange(len(Es)), counts)
        j0 = (np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
              + np.repeat(np.arange(len(Es)), counts) + 1)
        I, J = o[i0], o[j0]

    out_i, out_j = [], []
    for s in range(0, len(I), chunk):
        i, j = I[s:s + chunk], J[s:s + chunk]
        keep = s1[i] + s1[j] - 2 * np.einsum("ka,ka->k", rad[i], rad[j]) <= b1 ** 2
        i, j = i[keep], j[keep]
        if len(i):
            keep = s2[i] + s2[j] - 2 * np.einsum("ka,ka->k", dmat[i], dmat[j]) <= b2 ** 2
            i, j = i[keep], j[keep]
        out_i.append(i)
        out_j.append(j)
    if not out_i:                                         # fewer than two conformers: no pair at all
        return np.zeros(0, int), np.zeros(0, int)
    I, J = np.concatenate(out_i), np.concatenate(out_j)
    swap = I > J
    I[swap], J[swap] = J[swap], I[swap]
    return I, J


def pair_rmsd(X, heavy, perms, I, J, tmax, chunk: int = 200_000):
    """Symmetric heavy-atom RMSD of the given pairs, minimised over ``perms`` (Kabsch per pair).

    The identity permutation is tried first for every pair; the others are applied only to pairs
    still above ``tmax``, which is where a symmetry-related match could still bring them below it.
    """
    H = X[:, heavy]
    H = H - H.mean(axis=1, keepdims=True)
    n = len(heavy)
    sq = (H ** 2).sum(axis=(1, 2))

    def msd(a, b, p):
        G = np.einsum("kia,kib->kab", H[a], H[b][:, p])
        sv = np.linalg.svd(G, compute_uv=False)
        det = np.sign(np.linalg.det(G))
        return (sq[a] + sq[b] - 2 * (sv[:, 0] + sv[:, 1] + det * sv[:, 2])) / n

    out = np.empty(len(I))
    for s in range(0, len(I), chunk):
        a, b = I[s:s + chunk], J[s:s + chunk]
        best = msd(a, b, perms[0])
        todo = np.flatnonzero(best > tmax ** 2)
        for p in perms[1:]:
            if len(todo) == 0:
                break
            m = msd(a[todo], b[todo], p)
            np.minimum(best[todo], m, out=m)
            best[todo] = m
            todo = todo[m > tmax ** 2]
        out[s:s + chunk] = np.sqrt(np.clip(best, 0, None))
    return out


def leaders(n: int, E, I, J, R, threshold: float):
    """Greedy leader clustering in ascending energy order.

    Returns ``kept`` (bool array) and ``leader`` (for a dropped conformer, the index of the kept one
    it duplicates). Because the visit order is ascending energy, a molecule's lowest-energy
    conformer is seen first and is always kept, whatever the threshold.
    """
    order = np.argsort(np.asarray(E), kind="stable")
    close = R < threshold
    adj: dict[int, list[int]] = {}
    for i, j in zip(I[close], J[close]):
        adj.setdefault(int(i), []).append(int(j))
        adj.setdefault(int(j), []).append(int(i))
    kept = np.zeros(n, bool)
    leader = np.full(n, -1, int)
    for i in order:
        hit = next((j for j in adj.get(int(i), ()) if kept[j]), None)
        if hit is None:
            kept[i] = True
        else:
            leader[i] = hit
    return kept, leader


# --------------------------------------------------------------------------- the two criteria
def _molecule_record(paths, energies_h, geometry_file, threshold, gate_kcal, max_automorphisms):
    """Per-conformer ``(kept, leader_path, rmsd_to_leader)`` for one molecule's conformers."""
    sym0, X = None, []
    for p in paths:
        s, xyz = read_xyz(f"{p}/{geometry_file}")
        if sym0 is None:
            sym0 = s
        elif not np.array_equal(sym0, s):
            raise ValueError(
                f"the conformers of this molecule do not share one atom order: {p} differs from "
                f"{paths[0]}. An RMSD between them would compare different atoms. Extract these "
                f"libraries separately, or fix the ensemble that was built with a different atom "
                f"order.")
        X.append(xyz)
    X = np.array(X)
    E_kcal = (np.asarray(energies_h, dtype=float) - np.nanmin(energies_h)) * HARTREE_TO_KCAL
    heavy, perms = automorphisms(sym0, X[0], max_automorphisms)
    I, J = candidate_pairs(X, E_kcal, heavy, threshold, gate_kcal)
    R = pair_rmsd(X, heavy, perms, I, J, threshold) if len(I) else np.zeros(0)
    kept, leader = leaders(len(X), E_kcal, I, J, R, threshold)
    rmsd = np.full(len(X), np.nan)
    if len(I):
        pair_r = {}
        for i, j, r in zip(I, J, R):
            pair_r[(int(i), int(j))] = r
        for i in range(len(X)):
            if leader[i] >= 0:
                a, b = sorted((i, int(leader[i])))
                rmsd[i] = pair_r.get((a, b), np.nan)
    return kept, leader, rmsd, len(perms), len(I)


def prune(df: pd.DataFrame, cfg: dict, molecule_column: str, geometry_file: str,
          progress: bool = True, jobs: int | None = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the ``prune`` block to a numbering table.

    Returns ``(survivors, record)``: the rows to describe, and one row per conformer saying what
    happened to it. ``record`` is written beside the other tables, because a count in a metadata
    block that cannot be checked afterwards is not provenance.
    """
    energy_column = cfg["energy_column"]
    window = cfg.get("energy_window_kcal_mol")
    threshold = cfg.get("rmsd_threshold_angstrom")
    gate = cfg.get("energy_gate_kcal_mol", ENERGY_GATE_KCAL)
    max_auto = int(cfg.get("max_automorphisms", MAX_AUTOMORPHISMS))

    if energy_column not in df.columns:
        raise KeyError(f"prune.energy_column {energy_column!r} is not a column of the numbering "
                       f"table; it has {list(df.columns)[:8]}...")
    out = df.reset_index(drop=True).copy()
    e = pd.to_numeric(out[energy_column], errors="coerce")
    emin = e.groupby(out[molecule_column]).transform("min")
    out["prune_delta_e_kcal_mol"] = (e - emin) * HARTREE_TO_KCAL
    out["prune_status"] = KEPT
    out["prune_leader"] = ""
    out["prune_rmsd_to_leader"] = np.nan
    out["prune_n_automorphisms"] = np.nan

    out.loc[e.isna(), "prune_status"] = NO_ENERGY
    if window is not None:
        above = (out["prune_delta_e_kcal_mol"] > float(window)) & e.notna()
        out.loc[above, "prune_status"] = ABOVE_WINDOW

    if threshold is not None:
        alive = out[out["prune_status"] == KEPT]
        from desc_utils.parallel import parallel_map
        tasks = [((list(g["path"]), list(g[energy_column]), list(g.index)),
                  float(threshold), gate, max_auto, geometry_file)
                 for _, g in alive.groupby(molecule_column, sort=True)]
        results = parallel_map(_prune_worker, tasks, jobs, progress=progress,
                               desc=f"rmsd dedupe[{threshold} A]")
        for idx, kept, leader, rmsd, n_perms in results:
            paths = [out.at[r, "path"] for r in idx]
            for k, row in enumerate(idx):
                out.at[row, "prune_n_automorphisms"] = n_perms
                if not kept[k]:
                    out.at[row, "prune_status"] = DUPLICATE
                    out.at[row, "prune_leader"] = paths[int(leader[k])]
                    out.at[row, "prune_rmsd_to_leader"] = rmsd[k]

    record = out[["path", molecule_column, energy_column, "prune_delta_e_kcal_mol",
                  "prune_status", "prune_leader", "prune_rmsd_to_leader",
                  "prune_n_automorphisms"]].copy()
    survivors = out[out["prune_status"] == KEPT].drop(
        columns=["prune_status", "prune_leader", "prune_rmsd_to_leader",
                 "prune_delta_e_kcal_mol", "prune_n_automorphisms"]).reset_index(drop=True)
    return survivors, record


def _prune_worker(arg):
    (paths, energies, idx), threshold, gate, max_auto, geometry_file = arg
    kept, leader, rmsd, n_perms, _ = _molecule_record(
        paths, energies, geometry_file, threshold, gate, max_auto)
    return idx, kept, leader, rmsd, n_perms


def summary(record: pd.DataFrame, max_automorphisms: int = MAX_AUTOMORPHISMS) -> dict:
    """What the prune did, for the run's own report and for the library metadata.

    ``molecules_at_automorphism_cap`` is reported because a molecule whose heavy-atom graph has more
    than ``max_automorphisms`` symmetries is compared against an arbitrary subset of them: its count
    can then differ by a conformer or two between RDKit versions. Measured on this set, two ligands
    of 115 are at the cap. Hiding that would make an unreproducible count look exact.
    """
    counts = record["prune_status"].value_counts().to_dict()
    capped = []
    if "prune_n_automorphisms" in record:
        at_cap = record[record["prune_n_automorphisms"] >= max_automorphisms]
        capped = sorted(set(at_cap.iloc[:, 1].astype(str)))
    return {
        "molecules_at_automorphism_cap": capped,
        "n_conformers": int(len(record)),
        "n_kept": int(counts.get(KEPT, 0)),
        "n_above_energy_window": int(counts.get(ABOVE_WINDOW, 0)),
        "n_duplicates": int(counts.get(DUPLICATE, 0)),
        "n_without_energy": int(counts.get(NO_ENERGY, 0)),
    }
