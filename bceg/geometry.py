"""Start geometries for single-start searchers (CREST, GOAT, molassembler).

Each replicate gets its own ETKDG embedding (seed = replicate seed), optionally
pre-optimised with xtb (``start_preopt: gfn2 | gfn1 | gfn0 | gfnff``) or with UMA
(``start_preopt: uma``, parameters in ``start_preopt_params``), then checked:

- every bond to a metal (e.g. Ni-N, Ni-F, Ni-H) must be <= ``max_metal_ligand_distance`` and
  >= COLLAPSE_FACTOR x the covalent-radii sum;
- every other bond of the graph must stay < 1.35 x the covalent-radii sum;
- no two non-bonded atoms may be closer than the covalent-radii sum (e.g. collapsed H-H).

An embedding with two atoms on top of each other (closer than COLLAPSE_FACTOR x their covalent-radii sum,
e.g. an iodide trapped on Sm, where distance geometry has no gradient to pull it off) is rejected before the
pre-optimisation, which cannot undo it. A rejected geometry is re-embedded with the next seed (``seed + attempt * 10007``),
so the result is deterministic per replicate.

With ``start_constraints`` (``pre_refinement: constrained``) the searcher's ``constraints`` are imposed
on each embedding before the pre-optimisation (bceg.impose: a gradual ramp, then an exact snap). The
pre-optimisation keeps what was imposed (planes, linear angles, values: bceg.impose.holds) and relaxes
everything else, bond lengths included; the values are snapped exact once more after it. During the imposition the connectivity and the stereocentres
must not change from the embedding's; after the pre-optimisation they must pass outright. Otherwise the
seed is rejected.

The full pre-optimisation (``start_preopt_level: normal``) takes rigid chelates
(e.g. bpy-NiF2) to the same minimum from every seed. This is accepted: the N-Ni-N bite
angle varies between ligands but should be consistent for one ligand, and replicate
variability comes from the searches. The info returned records the RMSD to the raw
embedding and the geometry at each metal (angles, ``tau4``) for monitoring.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from rdkit.Geometry import Point3D

from bceg import constraints as cons
from bceg.chem import ORGANIC_ELEMENTS, embed, stereo_matches
from bceg.external import resolve_threads
from bceg.impose import ImposeError, holds, impose, off_targets, snap, target_atoms, values, xtb_restraints
from bceg.structure import Structure
from bceg.xtb import xtb_identity, xtb_optimize
from bceg.xyz import Conformer

SEED_STRIDE = 10007
WALL_MARGIN = 0.3  # Angstrom below max_metal_ligand_distance where an imposed atom's bond to a metal meets its wall
BOND_STRETCH_FACTOR = 1.35
CLASH_FACTOR = 1.0
#: closer than this x the covalent-radii sum, two atoms are on top of each other: such an embedding is rejected
#: before the pre-optimisation, and a bond to a metal this short is a problem (real bonds: Ni-H 0.94x,
#: Sm-O 0.91x; ETKDG squeezes Sm-I to ~0.65x, which the pre-optimisation fixes)
COLLAPSE_FACTOR = 0.5


def pair_geometry(mol: Chem.Mol, coords: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """(symbols, covalent radii, distance matrix, graph adjacency as bool) of ``coords`` against ``mol``."""
    pt = Chem.GetPeriodicTable()
    coords = np.asarray(coords, dtype=float)
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    radii = np.array([pt.GetRcovalent(a.GetAtomicNum()) for a in mol.GetAtoms()])
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    bonded = Chem.GetAdjacencyMatrix(mol).astype(bool)
    return symbols, radii, dist, bonded


def _coordination_issues(mol: Chem.Mol, coords: np.ndarray, max_metal_ligand_distance: float) -> list[tuple]:
    """(i, j, distance, +1 if too long / -1 if too short, message) of every problem pair."""
    symbols, radii, dist, bonded = pair_geometry(mol, coords)

    issues = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        d, label = dist[i, j], f"{symbols[i]}{i}-{symbols[j]}{j}"
        if symbols[i] not in ORGANIC_ELEMENTS or symbols[j] not in ORGANIC_ELEMENTS:
            if d > max_metal_ligand_distance:
                issues.append((i, j, d, 1, f"{label} {d:.2f} A > {max_metal_ligand_distance} A (decoordinated)"))
            elif d < COLLAPSE_FACTOR * (radii[i] + radii[j]):
                issues.append((i, j, d, -1, f"{label} {d:.2f} A (metal-ligand bond collapsed)"))
        elif d > BOND_STRETCH_FACTOR * (radii[i] + radii[j]):
            issues.append((i, j, d, 1, f"{label} {d:.2f} A (bond broken)"))
    iu = np.triu_indices(len(symbols), k=1)
    for i, j in zip(*iu):
        if not bonded[i, j] and dist[i, j] < CLASH_FACTOR * (radii[i] + radii[j]):
            issues.append((i, j, dist[i, j], -1, f"{symbols[i]}{i}...{symbols[j]}{j} {dist[i, j]:.2f} A (non-bonded clash)"))
    return issues


def collapsed_pairs(mol: Chem.Mol, coords: np.ndarray) -> list[str]:
    """Pairs (bonded or not) closer than COLLAPSE_FACTOR x their covalent-radii sum: atoms on top of each other."""
    symbols, radii, dist, _ = pair_geometry(mol, coords)
    i, j = np.nonzero(np.triu(dist < COLLAPSE_FACTOR * (radii[:, None] + radii[None, :]), k=1))
    return [f"{symbols[a]}{a}...{symbols[b]}{b} {dist[a, b]:.2f} A (collapsed embedding)" for a, b in zip(i, j)]


def coordination_problems(mol: Chem.Mol, coords: np.ndarray, max_metal_ligand_distance: float) -> list[str]:
    return [issue[-1] for issue in _coordination_issues(mol, coords, max_metal_ligand_distance)]


def connectivity_problems(mol: Chem.Mol, coords: np.ndarray, max_metal_ligand_distance: float) -> list[str]:
    """:func:`coordination_problems` plus the tetrahedral stereocentres: a geometry whose 3D configuration
    disagrees with the graph's (an inversion pushed through by imposing constraints) is a problem too."""
    problems = coordination_problems(mol, coords, max_metal_ligand_distance)
    if not stereo_matches(_with_coords(mol, coords)):
        problems.append("a stereocentre's configuration changed")
    return problems


#: Angstrom a pair must have moved the wrong way (closer for a clash, apart for a bond) since the reference to
#: count as changed by imposing constraints; ETKDG's squeezed metal complexes sit right at the clash limit
#: (M-O-C 1-3 pairs), where a 0.01 A wobble would otherwise reject the seed
CHANGE_MARGIN = 0.1


def new_problems(mol: Chem.Mol, reference: np.ndarray, max_metal_ligand_distance: float):
    """``check(coords)``: the problems of ``coords`` that imposing constraints made, compared with
    ``reference`` pair by pair: a pair past the limit and at least CHANGE_MARGIN further the wrong way than
    in ``reference``, or an inverted stereocentre. Imposing constraints must change nothing; the
    embedding's own problems (ETKDG squeezes or stretches metal-ligand bonds) are the pre-optimisation's
    to fix, as without constraints."""
    ref = np.asarray(reference, dtype=float)

    def check(coords: np.ndarray) -> list[str]:
        out = [msg for i, j, d, sign, msg in _coordination_issues(mol, coords, max_metal_ligand_distance)
               if sign * (d - np.linalg.norm(ref[i] - ref[j])) > CHANGE_MARGIN]
        if not stereo_matches(_with_coords(mol, coords)):
            out.append("a stereocentre's configuration changed")
        return out
    return check


def _with_coords(mol: Chem.Mol, coords: np.ndarray) -> Chem.Mol:
    m = Chem.Mol(mol)
    m.RemoveAllConformers()
    conf = Chem.Conformer(m.GetNumAtoms())
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    m.AddConformer(conf)
    return m


def heavy_atom_rmsd(mol: Chem.Mol, a: np.ndarray, b: np.ndarray) -> float:
    """Symmetry-aware heavy-atom RMSD (Angstrom) between two geometries of ``mol``."""
    ma = Chem.RemoveHs(_with_coords(mol, a), sanitize=False)
    mb = Chem.RemoveHs(_with_coords(mol, b), sanitize=False)
    return float(rdMolAlign.GetBestRMS(ma, mb, maxMatches=10000))


def metal_geometry(mol: Chem.Mol, coords: np.ndarray) -> dict[str, dict]:
    """Ligand-metal-ligand angles and, for 4-coordinate metals, tau4 (0 square planar, 1 tetrahedral)."""
    out = {}
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in ORGANIC_ELEMENTS:
            continue
        m, nbrs = atom.GetIdx(), [n.GetIdx() for n in atom.GetNeighbors()]
        angles = {}
        for k, i in enumerate(nbrs):
            for j in nbrs[k + 1:]:
                u, v = coords[i] - coords[m], coords[j] - coords[m]
                cos = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
                label = f"{mol.GetAtomWithIdx(i).GetSymbol()}{i}-{atom.GetSymbol()}{m}-{mol.GetAtomWithIdx(j).GetSymbol()}{j}"
                angles[label] = round(float(np.degrees(np.arccos(np.clip(cos, -1, 1)))), 1)
        info = {"angles_deg": angles}
        if len(nbrs) == 4:
            a, b = sorted(angles.values())[-2:]
            info["tau4"] = round((360.0 - a - b) / 141.0, 3)
        out[f"{atom.GetSymbol()}{m}"] = info
    return out


def _uma_preoptimizer(params: dict):
    """The UMA refiner used for ``start_preopt: uma`` (built once per call of make_start_geometry)."""
    from bceg.refine.ase_opt import package_version
    from bceg.refine.uma import UMA

    # one geometry: no pool; the start geometry is made at the transformation's spin state (no multiplicity)
    allowed = set(UMA.defaults) - {"lot", "keep_unconverged", "workers", "constraints", "multiplicity"}
    unknown = set(params["start_preopt_params"]) - allowed
    if unknown:
        raise ValueError(f"start_preopt_params: unknown UMA parameter(s) {sorted(unknown)}; allowed: {sorted(allowed)}")
    return UMA(**params["start_preopt_params"]), "fairchem-core", package_version("fairchem-core")


# --------------------------------------------------------------------------- configurations of metal-bound N
_TAG = {"CW": Chem.CHI_TETRAHEDRAL_CW, "CCW": Chem.CHI_TETRAHEDRAL_CCW}


def bound_amine_stereocentres(mol: Chem.Mol) -> list[int]:
    """Indices of the sp3 N bonded to a metal whose three other neighbours are all different.

    A tertiary amine bound to a metal is a stereocentre that cannot invert without letting go of the
    metal, so a conformer search started from one configuration rarely reaches the others. RDKit does
    not perceive it (dative bonds do not count as neighbours for stereo perception), hence this function.
    An N with two identical substituents (NMe2, NBn2) is not a stereocentre and is left out.
    """
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    out = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "N" or atom.GetIsAromatic() or atom.GetDegree() != 4:
            continue
        nbrs = list(atom.GetNeighbors())
        metal = [n for n in nbrs if n.GetSymbol() not in ORGANIC_ELEMENTS]
        others = [n for n in nbrs if n.GetSymbol() in ORGANIC_ELEMENTS]
        if len(metal) == 1 and len(others) == 3 and len({ranks[n.GetIdx()] for n in others}) == 3:
            out.append(atom.GetIdx())
    return out


def chiral_tag_from_coords(mol: Chem.Mol, idx: int, coords: np.ndarray) -> str:
    """'CW' or 'CCW': the RDKit chiral tag of atom ``idx`` (neighbours in its bond order) read from 3D.

    Works for any 4-coordinate atom, including an N whose fourth bond is dative (which RDKit's own
    perception from 3D ignores); for an ordinary carbon stereocentre it agrees with RDKit.
    """
    atom = mol.GetAtomWithIdx(idx)
    nbrs = [b.GetOtherAtomIdx(idx) for b in atom.GetBonds()]
    if len(nbrs) != 4:
        raise ValueError(f"atom {idx} has {len(nbrs)} neighbours, need 4")
    p = [np.asarray(coords[j], dtype=float) for j in nbrs]
    v = float(np.dot(p[1] - p[0], np.cross(p[2] - p[0], p[3] - p[0])))
    return "CW" if v > 0 else "CCW"


def with_bound_n_configuration(mol: Chem.Mol, centres: list[int], combination) -> Chem.Mol:
    """A copy of ``mol`` with the given chiral tags on the metal-bound N, which ETKDG then enforces.

    The tags are set raw: RDKit's stereo assignment would strip them (the N is no stereocentre to
    RDKit), but the embedding honours them.
    """
    out = Chem.Mol(mol)
    for idx, tag in zip(centres, combination):
        out.GetAtomWithIdx(idx).SetChiralTag(_TAG[tag])
    return out


def bound_n_combinations(mol: Chem.Mol, centres: list[int], seed: int, tries: int = 3) -> tuple[list, list]:
    """(feasible, infeasible) configuration combinations of ``centres``, in a fixed order.

    A combination is feasible when ETKDG embeds it with the enforced configuration within ``tries``
    seeds (some are impossible in fused or strained chelate rings). Deterministic for given inputs.
    """
    import itertools

    feasible, infeasible = [], []
    for combo in itertools.product(("CW", "CCW"), repeat=len(centres)):
        tagged = with_bound_n_configuration(mol, centres, combo)
        ok = False
        for t in range(tries):
            try:
                xyz = embed(tagged, seed + SEED_STRIDE * t, ff="none")
            except RuntimeError:
                continue
            if tuple(chiral_tag_from_coords(tagged, i, xyz) for i in centres) == combo:
                ok = True
                break
        (feasible if ok else infeasible).append(combo)
    return feasible, infeasible


def _bound_n_setup(structure: Structure, seed: int, params: dict) -> tuple[Chem.Mol, dict | None]:
    """start_bound_n_stereo: enumerate -> (the mol to embed, the record); otherwise (structure.mol, None).

    Replicate ``i`` (seed = seed_offset + i) starts from the ((i - 1) mod n)-th of the n feasible
    configurations of the metal-bound amine N, so replicates 1..n cover every configuration once.
    """
    if params.get("start_bound_n_stereo", "none") != "enumerate":
        return structure.mol, None
    if params["start_geometry"] != "etkdg":
        raise ValueError("start_bound_n_stereo: enumerate needs start_geometry: etkdg")
    centres = bound_amine_stereocentres(structure.mol)
    if not centres:
        return structure.mol, {"centres": [], "configuration": [], "index": None, "feasible": [], "infeasible": []}
    offset = params.get("seed_offset", 0)
    feasible, infeasible = bound_n_combinations(structure.mol, centres, offset + 1)
    if not feasible:
        raise RuntimeError(f"no configuration of the metal-bound N {centres} could be embedded")
    k = (seed - offset - 1) % len(feasible)
    return (with_bound_n_configuration(structure.mol, centres, feasible[k]),
            {"centres": centres, "configuration": list(feasible[k]), "index": k,
             "feasible": [list(c) for c in feasible], "infeasible": [list(c) for c in infeasible]})


def _bound_n_problems(mol: Chem.Mol, coords: np.ndarray, record: dict | None, max_dist: float) -> list[str]:
    """The metal-bound N whose configuration differs from the one imposed (decoordinated N excepted)."""
    if not record or not record["centres"]:
        return []
    out = []
    for idx, want in zip(record["centres"], record["configuration"]):
        metal = next(n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors() if n.GetSymbol() not in ORGANIC_ELEMENTS)
        if np.linalg.norm(np.asarray(coords[idx]) - np.asarray(coords[metal])) > max_dist:
            continue  # let go of the metal: may invert, start_allow_decoordination decides about the bond
        got = chiral_tag_from_coords(mol, idx, coords)
        if got != want:
            out.append(f"the configuration of metal-bound N{idx + 1} changed ({want} -> {got})")
    return out


def make_start_geometry(structure: Structure, seed: int, workdir: Path, params: dict) -> tuple[np.ndarray, dict]:
    """Return (coords, info) following the start_* parameters of a searcher."""
    rejected = []
    mol, bound_n = _bound_n_setup(structure, seed, params)
    preopt = params["start_preopt"]
    spec = params.get("constraints") if params.get("start_constraints") else None
    targets = cons.targets(spec, structure.mol) if spec else None

    # which program ran is not in the params hash (xtb_executable is a resource key), so record it here
    uma = None
    if preopt == "uma":
        uma, program, version = _uma_preoptimizer(params)
    elif preopt != "none":
        program, version = xtb_identity(params["xtb_executable"])
    else:
        program, version = None, None
    for attempt in range(params["start_max_attempts"]):
        s = seed + attempt * SEED_STRIDE
        if params["start_geometry"] == "input":
            if structure.coords is None:
                raise ValueError("start_geometry='input' but the transformation provided no coordinates")
            coords, source = structure.coords, "input"
        elif params["start_geometry"] == "etkdg":
            try:
                coords, source = embed(mol, s, ff=params["start_ff"]), f"etkdg(seed={s}, ff={params['start_ff']})"
            except RuntimeError as exc:
                rejected.append({"seed": s, "problems": [str(exc)]})
                continue
            # embed() falls back to an unenforced embedding, which RDKit's stereo check does not cover for N
            wrong = collapsed_pairs(mol, coords) + _bound_n_problems(mol, coords, bound_n, float("inf"))
            if wrong:
                rejected.append({"seed": s, "stage": "embedded", "problems": wrong})
                continue
        else:
            raise ValueError(f"Unknown start_geometry {params['start_geometry']!r}")

        energy, embedded, imposed, hold = None, coords, None, None
        if spec:
            try:
                check = new_problems(structure.mol, coords, params["max_metal_ligand_distance"])
                coords, imposed = impose(structure.mol, coords, targets, target_atoms(targets),
                                         steps=params["start_impose_steps"], check=check)
            except ImposeError as exc:
                rejected.append({"seed": s, "stage": "imposed", "problems": [str(exc)]})
                if params["start_geometry"] == "input":
                    break
                continue
            hold = holds(coords, targets)  # the pre-optimisation keeps these and relaxes the rest, bonds included
            # the imposed atoms' own bonds to a metal (Sm-I of I-Sm-I) must not break: a flat-bottom wall (UMA)
            held = set(target_atoms(targets))
            hold["walls"] = [{"atoms": [b.GetBeginAtomIdx(), b.GetEndAtomIdx()],
                              "max": params["max_metal_ligand_distance"] - WALL_MARGIN}
                             for b in structure.mol.GetBonds()
                             if {b.GetBeginAtomIdx(), b.GetEndAtomIdx()} <= held
                             and not {b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol()} <= ORGANIC_ELEMENTS]
        if uma is not None:
            result = uma.run(structure, [Conformer(coords)], workdir / f"preopt_{attempt}", **({"holds": hold} if hold else {}))
            if not result.conformers:
                raise RuntimeError("UMA pre-optimisation returned no geometry")
            coords, energy = np.asarray(result.conformers[0].coords), result.conformers[0].energy
        elif preopt != "none":
            coords, energy = xtb_optimize(
                structure.elements, coords, structure.charge, structure.uhf, workdir / f"preopt_{attempt}",
                method=preopt, level=params["start_preopt_level"],
                executable=params["xtb_executable"], threads=resolve_threads(params.get("threads", params.get("nprocs"))),
                restraints=xtb_restraints(coords, hold) if hold else None)
        if spec:
            # UMA keeps the holds exactly, xtb's restraints nearly: the exact values once more, a small move
            imposed["after_preopt"] = values(coords, targets)
            far = off_targets(coords, targets)
            if far:
                rejected.append({"seed": s, "stage": "preopt", "problems": [f"the pre-optimisation left {far}"]})
                if params["start_geometry"] == "input":
                    break
                continue
            coords = snap(structure.mol, coords, targets)
        # start_allow_decoordination: a stretched metal-ligand bond is recorded, unless it is one of the
        # imposed atoms' (I-Sm of I-Sm-I), which must hold
        decoordinated = []
        if spec or params.get("start_allow_decoordination"):
            strict = set(target_atoms(targets)) if spec else set()
            problems = []
            for i, j, d, sign, msg in _coordination_issues(structure.mol, coords, params["max_metal_ligand_distance"]):
                if params.get("start_allow_decoordination") and "decoordinated" in msg and not {i, j} <= strict:
                    decoordinated.append(msg)
                else:
                    problems.append(msg)
        else:
            problems = coordination_problems(structure.mol, coords, params["max_metal_ligand_distance"])
        if spec:
            if not stereo_matches(_with_coords(structure.mol, coords)):
                problems.append("a stereocentre's configuration changed")
            problems += off_targets(coords, targets, final=True)
        problems += _bound_n_problems(mol, coords, bound_n, params["max_metal_ligand_distance"])
        if not problems:
            extra = {"imposed": imposed | {"preopt_holds": hold}} if spec else {}
            if bound_n is not None:
                extra["bound_n_stereo"] = bound_n
            if params.get("start_allow_decoordination"):
                extra["decoordinated"] = decoordinated
            return coords, extra | {
                "source": source, "seed": s, "attempts": attempt + 1, "preopt": preopt,
                "preopt_level": params["start_preopt_params"] if preopt == "uma" else params["start_preopt_level"],
                "preopt_energy": energy, "rejected": rejected,
                "preopt_program": program, "preopt_version": version,
                "rmsd_preopt_vs_embedded": round(heavy_atom_rmsd(structure.mol, embedded, coords), 3),
                "metal_geometry": metal_geometry(structure.mol, coords),
            }
        rejected.append({"seed": s, "problems": problems} | ({"stage": "preopt"} if spec else {}))
        if params["start_geometry"] == "input":
            break
    raise RuntimeError(f"No acceptable start geometry after {len(rejected)} attempt(s): {rejected}")
