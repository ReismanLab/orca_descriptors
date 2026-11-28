"""NBO descriptors keyed by what the orbital IS, not by where NBO numbered it.

The rank naming (``nbo_hm2_C2_C1_energy``) counts NBO numbers from the highest-numbered NBO with
occupancy >= 0.1. NBO numbers are not in energy order (Lewis CR/LP/BD first, then BD*/RY), that
reference NBO is often an antibond (an aromatic pi* at 0.3-0.4 e in every BnCl conformer), and the
numbering changes between NBO versions. On BnCl the C1-Cl sigma* sits anywhere from ``hm24`` to
``lp4``, so a rank-named column holds it in 47 of 279 conformers. Here every NBO gets a key from

- its **type** (CR, LP, LV, BD, BD*, 3C...; Rydberg orbitals are dropped),
- its **symmetry class**:
    two-centre: ``pi`` when every centre's hybrid has under ``PI_MAX_S_PERCENT`` s character,
    else ``sigma``;
    lone pair / lone vacancy: ``d`` when the hybrid is mostly d (``D_MIN_PERCENT``), refined to
    the dominant ligand-frame d orbital (``dz2``, ``dxz``...) when the spec defines a frame; else
    ``p`` when p% - s% > ``LP_P_MINUS_S`` (the lone-pair rule of SIMG, Boiko et al., Nat. Mach.
    Intell. 2025, 10.1038/s42256-025-01031-9); else ``sp``;
- its **atoms**, as spec labels written in spec order, so ``C1_Cl`` and ``Cl_C1`` are one key.

Three more rules:

- **Bond or lone pair.** NBO's Lewis search turns a bond polarised beyond ~95 % into a lone pair
  plus a lone vacancy, so a metal-ligand bond can be BD in one complex and LP + LV in another.
  A BD (BD*) with at least ``DONOR_POLARIZATION`` % on one labelled atom is keyed as an LP (LV)
  on that atom, and says so in ``from_bond``.
- **pi bonds per atom.** NBO picks one Kekule structure per molecule, so a ring pi bond can sit
  on A1-A2 in one molecule and A2-A3 in the next. Every pi NBO is also keyed per labelled atom
  with ``any`` partner (``BD_pi_A1_any``); spec labels are positions, so this is the same key
  whether the ring atom is C or N.
- **Groups.** Several NBOs can share a key (the two p lone pairs of Cl, the cores of Cl, a ring
  with two pi bonds on one atom's key). Each key reports ``n`` and the mean, min and max of
  energy and occupancy, as kraken (Gensch et al., JACS 2022, 10.1021/jacs.1c09718) and the
  Sigman bisphosphine work (Dotson et al., JACS 2023, 10.1021/jacs.2c08513) do for equivalent
  orbitals; the hybrid fields are means. A key with n = 1 has mean = min = max.

Second-order (E2) interactions are summed per (donor key, acceptor key) pair. Open-shell blocks
key the alpha and beta sets separately (``_alpha`` / ``_beta``): their Lewis structures can
differ.

Fields are physically meaningful only: energy (Eh), occupancy, polarisation (% of the NBO on each
labelled atom), the s/p/d character of each labelled atom's hybrid, and for a metal with a spec
frame the d character in that frame. Raw coefficients on px/py/pz or on basis functions depend on
the arbitrary input axes or the basis set and are not reported.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Mapping

import numpy as np

PI_MAX_S_PERCENT = 2.0
LP_P_MINUS_S = 80.0
D_MIN_PERCENT = 50.0
D_DOMINANT_FRACTION = 0.6
DONOR_POLARIZATION = 90.0
DROPPED_TYPES = {"RY", "RY*"}
ONE_CENTRE = {"CR", "LP", "LV", "LP*"}
D_LABELS = ("dz2", "dxz", "dyz", "dx2y2", "dxy")


# --------------------------------------------------------------------------- #
# ligand frame and d character
# --------------------------------------------------------------------------- #
def frame_axes(centre: np.ndarray, x_atom: np.ndarray, y_atom: np.ndarray) -> np.ndarray:
    """Rows x, y, z of the frame: x from the centre towards ``x_atom``, y towards ``y_atom``
    within the plane of the three atoms (Gram-Schmidt), z = x cross y. The axes follow the
    labels, so for ``frame: [Ni, N1, N2]`` x always points at N1 and y at N2's side."""
    x = np.asarray(x_atom, float) - np.asarray(centre, float)
    x /= np.linalg.norm(x)
    y = np.asarray(y_atom, float) - np.asarray(centre, float)
    y = y - np.dot(y, x) * x
    n = np.linalg.norm(y)
    if n < 1e-6:
        raise ValueError("the three frame atoms are collinear")
    y /= n
    return np.array([x, y, np.cross(x, y)])


def _d_tensors() -> dict:
    """Real d functions as orthonormal symmetric traceless 3x3 tensors (Frobenius product)."""
    s2, s6 = math.sqrt(2.0), math.sqrt(6.0)
    t = {k: np.zeros((3, 3)) for k in D_LABELS}
    t["dxy"][0, 1] = t["dxy"][1, 0] = 1 / s2
    t["dxz"][0, 2] = t["dxz"][2, 0] = 1 / s2
    t["dyz"][1, 2] = t["dyz"][2, 1] = 1 / s2
    t["dx2y2"][0, 0], t["dx2y2"][1, 1] = 1 / s2, -1 / s2
    t["dz2"][0, 0] = t["dz2"][1, 1] = -1 / s6
    t["dz2"][2, 2] = 2 / s6
    return t


_D = _d_tensors()


def d_character(nao_coefficients: list[float], nao_angular: list[str],
                axes: np.ndarray) -> dict | None:
    """Fraction of a hybrid's d part in each d orbital of the frame ``axes``.

    ``nao_coefficients`` is the hybrid over the atom's NAOs in NAO order and ``nao_angular``
    those NAOs' labels (``dxy``, ``dz2``...). Each d shell (3d, 4d...) is rotated on its own and
    the squares summed, so the answer does not depend on the input orientation."""
    if not nao_coefficients or len(nao_coefficients) != len(nao_angular):
        return None
    shells: dict[int, dict] = defaultdict(dict)
    seen: dict[str, int] = defaultdict(int)
    for c, lab in zip(nao_coefficients, nao_angular):
        lab = str(lab).replace("-", "").replace("^", "").lower()
        if lab in _D:
            shells[seen[lab]][lab] = c
            seen[lab] += 1
    w = dict.fromkeys(D_LABELS, 0.0)
    for comp in shells.values():
        T = sum(comp.get(k, 0.0) * _D[k] for k in D_LABELS)
        Tr = axes @ T @ axes.T
        for k in D_LABELS:
            w[k] += float(np.sum(Tr * _D[k])) ** 2
    tot = sum(w.values())
    return {k: v / tot for k, v in w.items()} if tot > 0 else None


# --------------------------------------------------------------------------- #
# keys
# --------------------------------------------------------------------------- #
def _token(nbo_type: str) -> str:
    return str(nbo_type).replace("*", "star")


def symmetry(nbo_type: str, hybrids: list[dict], d_frame: dict | None = None) -> str | None:
    """``sigma``/``pi`` for a multi-centre NBO; ``sp``/``p``/``d``/``dz2``... for a lone pair or
    lone vacancy; None for a core (its key needs none) or when the hybrids are unknown."""
    if nbo_type == "CR":
        return None
    if not hybrids or any(h.get("s_percent") is None for h in hybrids):
        return None
    if len(hybrids) >= 2:
        return "pi" if all(h["s_percent"] < PI_MAX_S_PERCENT for h in hybrids) else "sigma"
    h = hybrids[0]
    if (h.get("d_percent") or 0.0) >= D_MIN_PERCENT:
        if d_frame:
            best = max(d_frame, key=d_frame.get)
            if d_frame[best] >= D_DOMINANT_FRACTION:
                return best
        return "d"
    return "p" if (h.get("p_percent") or 0.0) - h["s_percent"] > LP_P_MINUS_S else "sp"


def _rows(table) -> list[dict]:
    if isinstance(table, dict) and "columns" in table:
        return [dict(zip(table["columns"], r)) for r in table["rows"]]
    return list(table or [])


def has_hybrids(nbo_results: Mapping) -> bool:
    """Whether the one-centre hybrids were parsed (condensed files made before they were are
    missing them, and their lone pairs cannot be classed)."""
    return any(c.get("one_centre") for b in _rows(nbo_results.get("bond_orbital_details"))
               for c in b.get("contributions") or [])


def keyed_nbos(nbo_results: Mapping, atom_map: Mapping[str, int], frame: list[str] | None = None,
               coords: np.ndarray | None = None) -> list[dict]:
    """Every NBO on labelled atoms with its keys: ``{"spin", "nbo_num", "type", "keys": [...],
    "primary": key or None, "energy", "occupancy", "pol": {label: %}, "hyb": {label: {s,p,d}},
    "dframe": {label: {dz2...}}, "from_bond"}``. ``atom_map`` is label -> 1-based atom number."""
    order = list(atom_map)
    label_of = {int(v): k for k, v in atom_map.items() if v is not None}
    details = {(b.get("spin"), b.get("bond_num")): b
               for b in _rows(nbo_results.get("bond_orbital_details"))}
    angular = defaultdict(list)
    for r in _rows(nbo_results.get("natural_atomic_orbitals")):
        angular[int(r["atom_num"])].append(r.get("angular"))
    axes = metal = None
    if frame and coords is not None and all(f in atom_map and atom_map[f] is not None for f in frame):
        c = [coords[int(atom_map[f]) - 1] for f in frame]
        axes, metal = frame_axes(*c), frame[0]

    out = []
    for r in _rows(nbo_results.get("natural_bond_orbitals")):
        t, num, spin = r.get("nbo_type"), r.get("nbo_num"), r.get("spin")
        if t in DROPPED_TYPES or num is None:
            continue
        atoms = [int(a) for a in r.get("atoms") or []]
        contribs = (details.get((spin, num)) or {}).get("contributions") or []
        hyb_by_atom = {int(c["atom_num"]): c for c in contribs if c.get("atom_num") is not None}
        hybrids = [hyb_by_atom.get(a, {}) for a in atoms]
        pol = {label_of[a]: hyb_by_atom.get(a, {}).get("percentage") for a in atoms if a in label_of}
        from_bond = False
        # A bond polarised onto one labelled atom is that atom's lone pair (vacancy).
        if t in ("BD", "BD*") and len(atoms) == 2 and all(h.get("percentage") is not None for h in hybrids):
            top = max(range(2), key=lambda i: hybrids[i]["percentage"])
            if hybrids[top]["percentage"] >= DONOR_POLARIZATION and atoms[top] in label_of:
                t, atoms, hybrids, from_bond = ("LP" if t == "BD" else "LV"), [atoms[top]], [hybrids[top]], True
        dframe = {}
        if axes is not None:
            for a, h in zip(atoms, hybrids):
                if label_of.get(a) == metal and h.get("nao_coefficients"):
                    dc = d_character(h["nao_coefficients"], angular.get(a, []), axes)
                    if dc:
                        dframe[metal] = dc
        sym = symmetry(t, hybrids if not from_bond else [{**hybrids[0], "percentage": 100.0}],
                       dframe.get(metal) if len(atoms) == 1 else None)
        if t not in ("CR",) and sym is None:
            continue
        labelled = [a for a in atoms if a in label_of]
        head = _token(t) + (f"_{sym}" if sym else "")
        keys, primary = [], None
        if labelled and len(labelled) == len(atoms):
            names = sorted((label_of[a] for a in atoms), key=order.index)
            primary = f"{head}_{'_'.join(names)}"
            keys.append(primary)
        if sym == "pi" and len(atoms) == 2:
            for a in labelled:
                k = f"{head}_{label_of[a]}_any"
                keys.append(k)
                primary = primary or k
        if not keys:
            continue
        out.append({"spin": spin, "nbo_num": num, "type": t, "keys": keys, "primary": primary,
                    "n_centres": len(r.get("atoms") or []),
                    "energy": r.get("energy"), "occupancy": r.get("occupancy"), "pol": pol,
                    "hyb": {label_of[a]: h for a, h in zip(atoms, hybrids) if a in label_of},
                    "dframe": dframe, "from_bond": from_bond})
    return out


def _mean(xs: Iterable) -> float | None:
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def chemical_descriptors(nbo_results: Mapping, atom_map: Mapping[str, int],
                         frame: list[str] | None = None, coords: np.ndarray | None = None) -> dict:
    """Flat ``nbo_<key>[_<spin>]_<field>`` descriptors of one conformer (see the module doc)."""
    nbos = keyed_nbos(nbo_results, atom_map, frame, coords)
    groups: dict[str, list[dict]] = defaultdict(list)
    for nb in nbos:
        for k in nb["keys"]:
            groups[(k, nb["spin"])].append(nb)
    flat: dict = {}
    for (key, spin), members in groups.items():
        base = f"nbo_{key}" + (f"_{spin}" if spin else "")
        flat[f"{base}_n"] = len(members)
        for field in ("energy", "occupancy"):
            vals = [m[field] for m in members if m[field] is not None]
            flat[f"{base}_{field}_mean"] = _mean(vals)
            flat[f"{base}_{field}_min"] = min(vals) if vals else None
            flat[f"{base}_{field}_max"] = max(vals) if vals else None
        labels = key.rsplit("_any", 1)[0].split("_")
        for lab in [x for x in labels if any(x in m["hyb"] for m in members)]:
            # Polarisation only means something for an NBO drawn on two or more centres.
            if any(m["n_centres"] > 1 for m in members):
                flat[f"{base}_pol_{lab}"] = _mean(m["pol"].get(lab) for m in members
                                                  if m["n_centres"] > 1)
            for comp, field in (("s", "s_percent"), ("p", "p_percent"), ("d", "d_percent")):
                flat[f"{base}_{comp}_{lab}"] = _mean(m["hyb"].get(lab, {}).get(field) for m in members)
            if any(lab in m["dframe"] for m in members):
                for d in D_LABELS:
                    flat[f"{base}_{d}_{lab}"] = _mean(m["dframe"].get(lab, {}).get(d) for m in members
                                                      if lab in m["dframe"])
        if any(m["from_bond"] for m in members):
            flat[f"{base}_from_bond"] = sum(1 for m in members if m["from_bond"])

    # E2: summed per (donor key, acceptor key), on primary keys only.
    primary = {(nb["spin"], nb["nbo_num"]): nb["primary"] for nb in nbos if nb["primary"]}
    e2: dict = defaultdict(float)
    for r in _rows(nbo_results.get("second_order_perturbation")):
        spin = r.get("spin")
        d = primary.get((spin, r.get("donor_nbo_num")))
        a = primary.get((spin, r.get("acceptor_nbo_num")))
        if d and a and r.get("e2_energy") is not None:
            e2[(d, a, spin)] += float(r["e2_energy"])
    for (d, a, spin), v in e2.items():
        flat[f"nbo_e2_{d}_to_{a}" + (f"_{spin}" if spin else "")] = v
    return flat


def fill_absent_e2(df):
    """Set an ``nbo_e2_<donor>_to_<acceptor>`` cell to 0 where it is missing but the conformer has
    both orbitals (``nbo_<donor>_n`` and ``nbo_<acceptor>_n`` filled): NBO only prints pairs above
    its threshold (0.50 kcal/mol closed shell, 0.25 per spin open shell), so the pair is ~0. The
    cell stays NaN when either orbital is absent or the conformer's NBO extraction failed."""
    for col in [c for c in df.columns if c.startswith("nbo_e2_")]:
        body, spin = col[len("nbo_e2_"):], ""
        for s in ("_alpha", "_beta"):
            if body.endswith(s):
                body, spin = body[: -len(s)], s
        if "_to_" not in body:
            continue
        d, a = body.split("_to_", 1)
        nd, na = f"nbo_{d}{spin}_n", f"nbo_{a}{spin}_n"
        if nd in df.columns and na in df.columns:
            df.loc[df[col].isna() & df[nd].notna() & df[na].notna(), col] = 0.0
    return df


def nlmo_descriptors(nlmo_rows, nbo_results: Mapping, atom_map: Mapping[str, int],
                     frame: list[str] | None = None, coords: np.ndarray | None = None) -> dict:
    """NLMO columns of one conformer, under the NBO keys of :func:`keyed_nbos`.

    Each NLMO is its parent NBO plus tails on other atoms. ``nlmo_rows`` come from
    ``NBOProps.get_NLMO`` and ``nbo_results`` is the NBO analysis of the same run, which supplies
    the keys (NLMO i has NBO i as parent). Per key (mean over the NBOs sharing it):

    - ``nbo_<key>[_<spin>]_nlmo_parent``: % of the NLMO that is the parent NBO (100 = fully
      localized; a lone pair donating into a sigma* drops below it),
    - ``nbo_<key>[_<spin>]_nlmo_on_<label>``: % of the NLMO on each labelled atom outside the key,
      0 when NBO printed no contribution there (it prints down to ~0.01 %).
    """
    nbos = keyed_nbos(nbo_results, atom_map, frame, coords)
    by_num = {(nb["spin"], nb["nbo_num"]): nb for nb in nbos}
    groups: dict = defaultdict(list)
    for r in nlmo_rows:
        nb = by_num.get((r.get("spin"), r["nlmo_num"]))
        if nb is None:
            continue
        expected = ({"LP": "BD", "LV": "BD*"}[nb["type"]] if nb["from_bond"] else nb["type"])
        if r["parent_type"] != expected:
            continue        # numbering mismatch: never attach an NLMO to the wrong NBO
        for k in nb["keys"]:
            groups[(k, nb["spin"])].append(r)
    labels = list(atom_map)
    flat: dict = {}
    for (key, spin), rows in groups.items():
        base = f"nbo_{key}" + (f"_{spin}" if spin else "")
        own = set(key.rsplit("_any", 1)[0].split("_")) & set(labels)
        flat[f"{base}_nlmo_parent"] = _mean(r["parent_percent"] for r in rows)
        for lab in labels:
            if lab in own or atom_map[lab] is None:
                continue
            flat[f"{base}_nlmo_on_{lab}"] = _mean(r["atoms"].get(int(atom_map[lab]), 0.0) for r in rows)
    return flat
