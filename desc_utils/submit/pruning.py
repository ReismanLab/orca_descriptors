"""Stage 5, ``pruning.method``: which of the conformers left by the topology check and the energy window
get a job. One method per library, applied by ``prepare`` to the conformers still ``prepared``, in the
order of the pooled ensemble. See ``docs/stages/5_pruning.md``.

``rmsd``    greedy leader clustering in ascending energy order on the symmetric heavy-atom RMSD; the same
            code as the descriptor-side prune (``desc_utils.prune.dedupe``), so the counts agree.
``tfd``     the same clustering on RDKit's torsion fingerprint deviation (Schulz-Gasch et al. 2012),
            computed on the stage-1 graph (``structure.mol``): alignment-free, blind to bond angles and to
            atoms without a torsion (a hydride's orientation).
``cregen``  CREST's CREGEN duplicate and rotamer criteria (energy ``ethr``, all-atom RMSD ``rthr``,
            rotational constants ``bthr`` with the anisotropy-dependent threshold), ported from
            ``src/cregen.f90`` and ``src/rotcompare.f90`` of CREST 3.0.2. The binary is not called: its
            standalone ``--cregen`` applies an energy window of 0 in 3.0.2 whatever ``--ewin`` or the TOML
            ``[cregen]`` block say, and keeps only the lowest conformer.
``marc``    navicat-marc (Laplaza et al., JPCL 2024): clusters the ensemble on RMSD, energy and dihedral
            dissimilarities and keeps one conformer per cluster. Optional dependency (``navicat_marc``).

Each method returns, per conformer, ``(status, info)``: ``prepared`` or a skipped status, and what is
recorded in metadata.json beside it (the conformer it duplicates, the distance to it, the MARC cluster).
"""

from __future__ import annotations

import time
import warnings

import numpy as np

from bceg.xyz import HARTREE_TO_KCAL, Conformer

METHODS = ("rmsd", "tfd", "cregen", "marc")
DUPLICATE = "skipped_duplicate"
NOT_REPRESENTATIVE = "skipped_not_representative"

#: per-method defaults; the chosen method's block is filled with these and hashed into params.json
DEFAULTS: dict[str, dict] = {
    "rmsd": {"threshold_angstrom": 0.01, "energy_gate_kcal_mol": 0.2, "max_automorphisms": 10000},
    "tfd": {"threshold": None, "use_weights": True, "energy_gate_kcal_mol": 0.2},   # no default: see the doc
    "cregen": {"ethr_kcal_mol": 0.05, "rthr_angstrom": 0.125, "bthr": 0.01, "bthrmax": 0.025,
               "bthrshift": 0.5, "heavy_atoms": False, "mirror": True, "drop_rotamers": True},
    "marc": {"metric": "avg", "cluster": "kmeans", "n_clusters": None, "pick": "lowest",
             "include_h": False, "scale_factor": 1.10},
}
MARC_METRICS = ("rmsd", "erel", "da", "ewrmsd", "ewda", "mix", "avg")
MARC_CLUSTERS = ("kmeans", "agglomerative", "affprop")
MARC_PICKS = ("lowest", "central")
#: the rmsd energy gate was measured safe below this threshold (desc_utils.prune.ENERGY_GATE_VALID_BELOW)
RMSD_GATE_VALID_BELOW = 0.02


# --------------------------------------------------------------------------- validation
def _number(v, where, positive=True, allow_none=False):
    if v is None and allow_none:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (positive and v <= 0):
        raise ValueError(f"{where} must be a {'positive ' if positive else ''}number, got {v!r}")
    return float(v)


def _flag(v, where):
    if not isinstance(v, bool):
        raise ValueError(f"{where} must be true or false, got {v!r}")
    return v


def check_params(method: str, block, where: str) -> dict:
    """The ``pruning.<method>`` block with its defaults filled in; raises ValueError on a bad value."""
    block = {**DEFAULTS[method], **(block or {})}
    unknown = set(block) - set(DEFAULTS[method])
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}; allowed: {sorted(DEFAULTS[method])}")
    if method == "rmsd":
        block["threshold_angstrom"] = _number(block["threshold_angstrom"], f"{where}.threshold_angstrom")
        block["energy_gate_kcal_mol"] = _number(block["energy_gate_kcal_mol"], f"{where}.energy_gate_kcal_mol",
                                                allow_none=True)
        m = block["max_automorphisms"]
        if isinstance(m, bool) or not isinstance(m, int) or m < 1:
            raise ValueError(f"{where}.max_automorphisms must be a positive integer, got {m!r}")
    elif method == "tfd":
        if block["threshold"] is None:
            raise ValueError(f"{where}.threshold has no default: on the nih2 set no TFD threshold reproduces the "
                             f"0.01 A RMSD duplicates (docs/stages/5_pruning.md); set one explicitly")
        block["threshold"] = _number(block["threshold"], f"{where}.threshold")
        if block["threshold"] >= 1:
            raise ValueError(f"{where}.threshold is a TFD (0 to 1), got {block['threshold']}")
        _flag(block["use_weights"], f"{where}.use_weights")
        block["energy_gate_kcal_mol"] = _number(block["energy_gate_kcal_mol"], f"{where}.energy_gate_kcal_mol",
                                                allow_none=True)
    elif method == "cregen":
        for k in ("ethr_kcal_mol", "rthr_angstrom", "bthr", "bthrmax"):
            block[k] = _number(block[k], f"{where}.{k}")
        block["bthrshift"] = _number(block["bthrshift"], f"{where}.bthrshift", positive=False)
        for k in ("heavy_atoms", "mirror", "drop_rotamers"):
            _flag(block[k], f"{where}.{k}")
    elif method == "marc":
        for k, allowed in (("metric", MARC_METRICS), ("cluster", MARC_CLUSTERS), ("pick", MARC_PICKS)):
            if block[k] not in allowed:
                raise ValueError(f"{where}.{k} must be one of {list(allowed)}, got {block[k]!r}")
        n = block["n_clusters"]
        if n is not None and (isinstance(n, bool) or not isinstance(n, int) or n < 1):
            raise ValueError(f"{where}.n_clusters must be a positive integer or null, got {n!r}")
        if n is not None and block["cluster"] == "affprop":
            raise ValueError(f"{where}: affprop chooses its own number of clusters; leave n_clusters null")
        _flag(block["include_h"], f"{where}.include_h")
        block["scale_factor"] = _number(block["scale_factor"], f"{where}.scale_factor")
    return block


def check_block(block: dict, where: str) -> dict:
    """Validate ``pruning.method`` and the method blocks in place. The method key and its block are removed
    when the method is null, so a library pruned before ``method`` existed keeps its params.json hash; the
    block of a method that is not the chosen one is refused rather than ignored."""
    method = block.get("method")
    given = [m for m in METHODS if m in block and block[m] is not None]
    if method is None:
        if given:
            raise ValueError(f"{where}: {given} given without pruning.method; set method: {given[0]}")
        for k in ("method", *METHODS):
            block.pop(k, None)
        return block
    if method not in METHODS:
        raise ValueError(f"{where}.method must be null or one of {list(METHODS)}, got {method!r}")
    others = [m for m in given if m != method]
    if others:
        raise ValueError(f"{where}: method is {method} but {others} is also given; remove it")
    for m in METHODS:
        if m != method:
            block.pop(m, None)
    raw = block.get(method) or {}
    params = check_params(method, raw, f"{where}.{method}")
    if method == "rmsd" and params["threshold_angstrom"] > RMSD_GATE_VALID_BELOW \
            and params["energy_gate_kcal_mol"] is not None and "energy_gate_kcal_mol" not in raw:
        raise ValueError(
            f"{where}.rmsd.threshold_angstrom {params['threshold_angstrom']} is above {RMSD_GATE_VALID_BELOW} A, "
            f"where the default energy gate ({DEFAULTS['rmsd']['energy_gate_kcal_mol']} kcal/mol) was never "
            f"measured; set rmsd.energy_gate_kcal_mol explicitly, or null to compare every pair")
    block[method] = params
    return block


# --------------------------------------------------------------------------- application
def apply(method: str, params: dict, elements: list[str], conformers: list[Conformer], mol=None,
          names: list[str] | None = None) -> tuple[list[tuple[str, dict]], dict]:
    """``(status, info)`` per conformer of ``conformers`` (the ones still ``prepared``), and a summary for the
    prepare log and ``orca_stage``. ``names`` (``conf_N``) label the leaders in ``info``."""
    names = names or [str(i) for i in range(len(conformers))]
    if not conformers:
        return [], {"method": method, "n_in": 0, "n_kept": 0, "seconds": 0.0}
    t0 = time.monotonic()
    out, extra = {"rmsd": _rmsd, "tfd": _tfd, "cregen": _cregen, "marc": _marc}[method](
        params, elements, conformers, mol, names)
    summary = {"method": method, "n_in": len(conformers), "n_kept": sum(s == "prepared" for s, _ in out),
               "seconds": round(time.monotonic() - t0, 2), **extra}
    return out, summary


def _energies(conformers, method, need: bool) -> np.ndarray:
    E = np.array([np.nan if c.energy is None else c.energy for c in conformers], dtype=float)
    if need and np.isnan(E).any():
        k = int(np.flatnonzero(np.isnan(E))[0])
        raise RuntimeError(f"pruning.method {method} needs an energy on every conformer; conformer {k + 1} of the "
                           f"ones left by the window has none on its comment line")
    return E


def _leader_records(kept, leader, dist, names, dist_key):
    out = []
    for i in range(len(kept)):
        if kept[i]:
            out.append(("prepared", {}))
        else:
            info = {"duplicate_of": names[int(leader[i])]}
            if dist is not None and np.isfinite(dist[i]):
                info[dist_key] = round(float(dist[i]), 5)
            out.append((DUPLICATE, info))
    return out


def _rmsd(params, elements, conformers, mol, names):
    from desc_utils import prune as pr
    X = np.array([c.coords for c in conformers])
    E = _energies(conformers, "rmsd", need=False)
    kept, leader, rmsd, n_perms, n_pairs = pr.dedupe(
        np.array(elements), X, list(E), params["threshold_angstrom"], params["energy_gate_kcal_mol"],
        params["max_automorphisms"])
    return (_leader_records(kept, leader, rmsd, names, "rmsd_to_leader_angstrom"),
            {"n_automorphisms": int(n_perms), "at_automorphism_cap": bool(n_perms >= params["max_automorphisms"]),
             "n_pairs_compared": int(n_pairs)})


def _gated_pairs(E_kcal, gate):
    n = len(E_kcal)
    I, J = np.triu_indices(n, 1)
    if gate is not None:
        keep = np.abs(E_kcal[I] - E_kcal[J]) <= gate
        I, J = I[keep], J[keep]
    return I, J


def tfd_mol(mol):
    """A copy of the stage-1 graph TFD can read: ring info perceived without sanitising (a Ni complex
    with dative bonds does not sanitise), no conformers."""
    from rdkit import Chem
    m = Chem.Mol(mol)
    m.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(m)
    m.RemoveAllConformers()
    return m


def _tfd(params, elements, conformers, mol, names):
    from desc_utils import prune as pr
    from rdkit import Chem
    from rdkit.Chem import TorsionFingerprints as TF
    from rdkit.Geometry import Point3D
    if mol is None:
        raise RuntimeError("pruning.method tfd needs the ensemble's structure (the molecular graph of stage 1)")
    m = tfd_mol(mol)
    if m.GetNumAtoms() != len(elements):
        raise RuntimeError(f"pruning.method tfd: the graph has {m.GetNumAtoms()} atoms, the ensemble {len(elements)}")
    for c in conformers:
        conf = Chem.Conformer(m.GetNumAtoms())
        for i, v in enumerate(c.coords):
            conf.SetAtomPosition(i, Point3D(*map(float, v)))
        m.AddConformer(conf, assignId=True)
    tl, tlr = TF.CalculateTorsionLists(m)
    if not tl and not tlr:
        raise RuntimeError("pruning.method tfd: the graph has no torsion RDKit can use; choose another method")
    weights = TF.CalculateTorsionWeights(m) if params["use_weights"] else None
    angles = [TF.CalculateTorsionAngles(m, tl, tlr, confId=k) for k in range(len(conformers))]
    E = _energies(conformers, "tfd", need=False)
    E_kcal = np.zeros(len(E)) if np.isnan(E).all() else (E - np.nanmin(E)) * HARTREE_TO_KCAL
    I, J = _gated_pairs(E_kcal, params["energy_gate_kcal_mol"])
    T = np.array([TF.CalculateTFD(angles[i], angles[j], weights) for i, j in zip(I, J)])
    kept, leader = pr.leaders(len(conformers), E_kcal, I, J, T, params["threshold"])
    pair = {(int(i), int(j)): t for i, j, t in zip(I, J, T)}
    dist = np.array([pair.get(tuple(sorted((i, int(leader[i])))), np.nan) if leader[i] >= 0 else np.nan
                     for i in range(len(conformers))])
    return (_leader_records(kept, leader, dist, names, "tfd_to_leader"),
            {"n_torsions": len(tl), "n_ring_torsion_groups": len(tlr), "n_pairs_compared": int(len(I))})


# --------------------------------------------------------------------------- CREGEN port
def _masses(elements):
    from rdkit import Chem
    pt = Chem.GetPeriodicTable()
    return np.array([pt.GetAtomicWeight(e) for e in elements]), np.array([pt.GetAtomicNumber(e) for e in elements])


def rotational_constants(X, masses):
    """Rotational constants (arbitrary units, B ~ 1/I), largest first, of each geometry in ``X``. CREGEN
    only compares ratios of them, so the unit drops out."""
    w = masses / masses.sum()
    out = np.empty((len(X), 3))
    for k, x in enumerate(X):
        c = x - (w[:, None] * x).sum(axis=0)
        inertia = np.eye(3) * (masses * (c ** 2).sum(axis=1)).sum() - np.einsum("i,ia,ib->ab", masses, c, c)
        moments = np.clip(np.linalg.eigvalsh(inertia), 1e-8, None)
        out[k] = np.sort(1.0 / moments)[::-1]
    return out


def rotaniso(rot):
    av = rot.mean(axis=-1)
    return np.sqrt(((rot - av[..., None]) ** 2).sum(axis=-1)) / av / (3.0 * np.sqrt(2.0 / 3.0))


def bthrerf(bthr, aniso, bthrmax, bthrshift):
    """CREST's anisotropy-dependent rotational-constant threshold (rotcompare.f90, bthrerf)."""
    from scipy.special import erf
    c = (bthrmax * 100.0 - bthr * 100.0) / 2.0
    a = -erf(-2.5) * c + bthr * 100.0
    b = 4.0 / 0.8
    d = bthrshift / 0.15
    return (erf(aniso * b - d) * c + a) / 100.0


def equalrotaniso(rot_i, rot_j, bthr, bthrmax, bthrshift) -> bool:
    av = (rotaniso(rot_i) + rotaniso(rot_j)) / 2.0
    thr = bthrerf(bthr, av, bthrmax, bthrshift)
    return bool(np.all(np.abs(rot_i / rot_j - 1.0) <= thr))


def _kabsch_rmsd(a, b):
    """Plain RMSD after optimal superposition (index-matched atoms, centred inputs)."""
    G = a.T @ b
    sv = np.linalg.svd(G, compute_uv=False)
    d = np.sign(np.linalg.det(G))
    msd = ((a ** 2).sum() + (b ** 2).sum() - 2 * (sv[0] + sv[1] + d * sv[2])) / len(a)
    return float(np.sqrt(max(msd, 0.0)))


def _cregen(params, elements, conformers, mol, names):
    E = _energies(conformers, "cregen", need=True)
    order = np.argsort(E, kind="stable")                 # CREGEN sorts the ensemble by energy first
    X = np.array([conformers[k].coords for k in order])
    er = E[order]
    masses, Z = _masses(elements)
    rot = rotational_constants(X, masses)
    mask = np.array([e != "H" for e in elements]) if params["heavy_atoms"] else np.ones(len(elements), bool)
    C = X[:, mask] - X[:, mask].mean(axis=1, keepdims=True)
    ethr, rthr, bthr = params["ethr_kcal_mol"], params["rthr_angstrom"], params["bthr"]
    bmax, bshift = params["bthrmax"], params["bthrshift"]
    n = len(X)

    # pass 1, duplicates: j < i (lower energy) with de < ethr and RMSD < rthr, or RMSD < 2 rthr with equal
    # rotational constants at bthr/2. As in CREST, the last such j is recorded and the chain is followed back.
    double = np.full(n, -1)
    rmsd_to = np.full(n, np.nan)
    for i in range(n):
        for j in range(i):
            if (er[i] - er[j]) * HARTREE_TO_KCAL >= ethr:
                continue
            dr = _kabsch_rmsd(C[i], C[j])
            if params["mirror"]:
                mj = C[j].copy()
                mj[:, 0] *= -1
                dr = min(dr, _kabsch_rmsd(C[i], mj))
            if dr < rthr or (dr < 2 * rthr and equalrotaniso(rot[i], rot[j], 0.5 * bthr, bmax, bshift)):
                double[i], rmsd_to[i] = j, dr
    root = double.copy()
    for i in range(n):
        while root[i] >= 0 and double[root[i]] >= 0:
            root[i] = double[root[i]]

    # pass 2, rotamers among the survivors: same energy within ethr, same rotational constants at bthr and the
    # same Coulomb-like invariant sum Z_i Z_j / r_ij^2 to 1e-3 (a permutation of equivalent atoms)
    rotamer_of = np.full(n, -1)
    if params["drop_rotamers"]:
        alive = [i for i in range(n) if double[i] < 0]
        iu = np.triu_indices(len(elements), 1)
        zz = Z[iu[0]] * Z[iu[1]]
        enuc = {i: float((zz / (((X[i][iu[0]] - X[i][iu[1]]) ** 2).sum(axis=1) + 1e-12)).sum()) for i in alive}
        for a, i in enumerate(alive):
            for j in alive[:a]:
                if rotamer_of[j] >= 0:
                    continue
                if abs(er[i] - er[j]) * HARTREE_TO_KCAL < ethr \
                        and equalrotaniso(rot[i], rot[j], bthr, bmax, bshift) \
                        and 2 * abs(enuc[i] - enuc[j]) / (enuc[i] + enuc[j]) < 1e-3:
                    rotamer_of[i] = j
                    break

    out: list = [None] * n
    for s in range(n):                                    # back to the input order
        k = int(order[s])
        if root[s] >= 0:
            out[k] = (DUPLICATE, {"duplicate_of": names[int(order[root[s]])], "cregen": "duplicate",
                                  "rmsd_to_leader_angstrom": round(float(rmsd_to[s]), 5)})
        elif rotamer_of[s] >= 0:
            out[k] = (DUPLICATE, {"duplicate_of": names[int(order[rotamer_of[s]])], "cregen": "rotamer"})
        else:
            out[k] = ("prepared", {})
    return out, {"n_duplicates": int((root >= 0).sum()), "n_rotamers": int((rotamer_of >= 0).sum())}


# --------------------------------------------------------------------------- MARC
def _marc(params, elements, conformers, mol, names):
    try:
        from navicat_marc import marc as marc_mod
        from navicat_marc.molecule import Molecule
    except ImportError as exc:
        raise RuntimeError("pruning.method marc needs navicat_marc (pip install navicat_marc==0.2.8) in the "
                           "environment the prepare task runs in") from exc
    needs_e = params["pick"] == "lowest" or params["metric"] in ("erel", "ewrmsd", "ewda", "mix", "avg")
    E = _energies(conformers, "marc", need=needs_e)
    if len(conformers) < 3:                               # nothing to cluster
        return [("prepared", {}) for _ in conformers], {"n_clusters": len(conformers)}
    mols = []
    for c, e in zip(conformers, E):
        lines = [str(len(elements)), "" if np.isnan(e) else f"{e:.10f}"]
        lines += [f"{el} {x:.8f} {y:.8f} {z:.8f}" for el, (x, y, z) in zip(elements, c.coords)]
        mols.append(Molecule(lines=lines, scale_factor=params["scale_factor"], noh=not params["include_h"]))
    mols = np.array(mols, dtype=object)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        try:
            indices, clusters = marc_mod.run_marc_from_args(
                basename="ensemble", molecules=mols, c=params["cluster"], m=params["metric"],
                n_clusters=params["n_clusters"], ewin=None, mine=params["pick"] == "lowest", verb=0,
                is_write=False)
        except TypeError as exc:
            if "NoneType" in str(exc):                  # navicat_marc 0.2.8: n < n_clusters with n_clusters None
                raise RuntimeError(
                    "navicat_marc dropped identical cluster representatives and then failed comparing with "
                    "n_clusters: null (a bug of navicat_marc 0.2.8); set pruning.marc.n_clusters, or remove exact "
                    "duplicates first") from exc
            raise
        except ValueError as exc:
            if "NaN" in str(exc):                       # a dissimilarity with zero spread (navicat_marc da.py)
                raise RuntimeError(
                    f"navicat_marc built a NaN dissimilarity with metric {params['metric']}: one of its components "
                    "has no spread over these conformers (e.g. every heavy-atom dihedral the same); use metric: rmsd "
                    "or erel") from exc
            raise
    status: list = [(NOT_REPRESENTATIVE, {}) for _ in conformers]
    for k, (rep, members) in enumerate(zip(indices, clusters), start=1):
        for i in members:
            status[int(i)] = (NOT_REPRESENTATIVE, {"marc_cluster": k, "marc_representative": names[int(rep)]})
        status[int(rep)] = ("prepared", {"marc_cluster": k})
    return status, {"n_clusters": len(indices)}
