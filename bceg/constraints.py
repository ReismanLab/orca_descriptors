"""Geometric constraints on the conformer search, written in terms of the molecule, not of atom indices.

A searcher that supports them (CREST, GOAT) takes a ``constraints`` parameter::

    constraints:
      force_constant: 0.5              # optional: harmonic restraints of this strength (CREST only)
      freeze:                          # atoms kept at their start-geometry positions
        - {smarts: "[Ni]", neighbors: 1}     # Ni and every atom bound to it
      bonds:                           # distances, by default at their start-geometry value
        - "[Ni]~[#7]"                  # a bare string is a SMARTS
      angles:    [{smarts: "[#7]~[Ni]~[#7]"}]
      dihedrals: [{smarts: "[#7:1]~[Ni:2]~[#7:3]~[#6:4]", value: 0.0, force_constant: 1.0}]

Each entry selects atoms with ``smarts`` or ``element`` (a symbol or a list of symbols). In a SMARTS
with atom maps only the mapped atoms count, in map order; otherwise every atom of the pattern does.
``freeze`` takes the atoms of every match, plus those within ``neighbors`` bonds of them (dative bonds
included); ``bonds`` / ``angles`` / ``dihedrals`` make one constraint per match, which must select
2 / 3 / 4 atoms, held at ``value`` (Angstrom / degrees) or, by default, at the start geometry's value;
one whose atoms are all frozen is dropped (it is fixed already). An entry that matches nothing fails
the run unless it says ``optional: true``.

The strength: without a force constant every constraint is exact (CREST ``freeze``, ORCA's
``{C i C}``) or left at the program's default strength (CREST's harmonic bond, angle and dihedral
potentials, 0.01 Eh/Bohr^2 or Eh/rad^2). ``force_constant`` (top level, or per entry, which wins) sets
the strength in Eh/Bohr^2 (distances) or Eh/rad^2 (angles, dihedrals); a frozen group with a force
constant becomes a harmonic restraint of every distance within the group (xtb's ``$constrain atoms:``),
which keeps its shape but lets it move as a whole. GOAT's constraints are always exact: a
``force_constant`` there is refused.

The selection is resolved per molecule when the search runs (the values need the start geometry):
:func:`resolve` returns plain atom indices (0-based, the order of every xyz file), which the searchers
write into their inputs and which are recorded in ``run_<i>.json`` under ``details.constraints``.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
from rdkit import Chem

KINDS = {"freeze": None, "bonds": 2, "angles": 3, "dihedrals": 4}  # kind -> atoms per constraint
TOP_KEYS = {*KINDS, "force_constant"}
ENTRY_KEYS = {"smarts", "element", "neighbors", "value", "force_constant", "optional"}
ATOMIC_NUMBERS = {Chem.GetPeriodicTable().GetElementSymbol(z): z for z in range(1, 119)}


# --------------------------------------------------------------------------- validation
def _force_constant(value: Any, where: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{where}.force_constant must be a positive number, got {value!r}")
    return float(value)


def _pattern(entry: dict, where: str) -> Chem.Mol:
    if ("smarts" in entry) == ("element" in entry):
        raise ValueError(f"{where}: give exactly one of `smarts` and `element`, got {entry!r}")
    if "element" in entry:
        symbols = entry["element"] if isinstance(entry["element"], list) else [entry["element"]]
        unknown = [s for s in symbols if s not in ATOMIC_NUMBERS]
        if unknown or not symbols:
            raise ValueError(f"{where}.element: unknown element(s) {unknown or symbols!r}")
        smarts = "[" + ",".join(f"#{ATOMIC_NUMBERS[s]}" for s in symbols) + "]"
    else:
        smarts = entry["smarts"]
    pattern = Chem.MolFromSmarts(smarts) if isinstance(smarts, str) else None
    if pattern is None:
        raise ValueError(f"{where}.smarts: not a valid SMARTS: {smarts!r}")
    return pattern


def _selected_positions(pattern: Chem.Mol) -> list[int]:
    """Positions in a match that count: the mapped atoms in map order, or all of them."""
    mapped = sorted((a.GetAtomMapNum(), a.GetIdx()) for a in pattern.GetAtoms() if a.GetAtomMapNum())
    return [i for _, i in mapped] if mapped else list(range(pattern.GetNumAtoms()))


def _entry(entry: Any, kind: str, where: str) -> dict:
    if isinstance(entry, str):
        entry = {"smarts": entry}
    if not isinstance(entry, dict):
        raise ValueError(f"{where}: expected a SMARTS string or a mapping, got {entry!r}")
    unknown = set(entry) - ENTRY_KEYS
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}; allowed: {sorted(ENTRY_KEYS)}")
    pattern = _pattern(entry, where)
    n = KINDS[kind]
    if kind == "freeze":
        if "value" in entry:
            raise ValueError(f"{where}: `value` is for bonds, angles and dihedrals, not freeze")
        neighbors = entry.get("neighbors", 0)
        if isinstance(neighbors, bool) or not isinstance(neighbors, int) or neighbors < 0:
            raise ValueError(f"{where}.neighbors must be a non-negative integer, got {neighbors!r}")
    else:
        if "neighbors" in entry:
            raise ValueError(f"{where}: `neighbors` is for freeze only")
        selected = len(_selected_positions(pattern))
        if selected != n:
            raise ValueError(f"{where}: a {kind[:-1]} constraint needs a SMARTS selecting {n} atoms, this one "
                             f"selects {selected} (use atom maps, e.g. [Ni:1]~[#7:2], to pick them out of a larger pattern)")
        value = entry.get("value")
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError(f"{where}.value must be a number, got {value!r}")
    if not isinstance(entry.get("optional", False), bool):
        raise ValueError(f"{where}.optional must be true or false, got {entry['optional']!r}")
    _force_constant(entry.get("force_constant"), where)
    return entry


def validate(spec: Any, where: str = "constraints") -> dict:
    """Check a ``constraints`` mapping (raises ValueError); returns it unchanged."""
    if spec is None:
        return {}
    if not isinstance(spec, dict):
        raise ValueError(f"{where}: expected a mapping, got {spec!r}")
    unknown = set(spec) - TOP_KEYS
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}; allowed: {sorted(TOP_KEYS)}")
    _force_constant(spec.get("force_constant"), where)
    for kind in KINDS:
        entries = spec.get(kind) or []
        if not isinstance(entries, list):
            raise ValueError(f"{where}.{kind}: expected a list, got {entries!r}")
        for i, entry in enumerate(entries):
            _entry(entry, kind, f"{where}.{kind}[{i}]")
    return spec


def has_force_constant(spec: dict) -> bool:
    return spec.get("force_constant") is not None or any(
        isinstance(e, dict) and e.get("force_constant") is not None for k in KINDS for e in spec.get(k) or [])


# --------------------------------------------------------------------------- geometry
def distance(coords: np.ndarray, i: int, j: int) -> float:
    return float(np.linalg.norm(coords[i] - coords[j]))


def angle(coords: np.ndarray, i: int, j: int, k: int) -> float:
    u, v = coords[i] - coords[j], coords[k] - coords[j]
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1))))


def dihedral(coords: np.ndarray, i: int, j: int, k: int, l: int) -> float:
    b0, b1, b2 = coords[i] - coords[j], coords[k] - coords[j], coords[l] - coords[k]
    b1 = b1 / np.linalg.norm(b1)
    v, w = b0 - np.dot(b0, b1) * b1, b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


MEASURE = {2: distance, 3: angle, 4: dihedral}


# --------------------------------------------------------------------------- resolution
def _matches(mol: Chem.Mol, entry: dict, where: str) -> list[tuple[int, ...]]:
    pattern = _pattern(entry, where)
    positions = _selected_positions(pattern)
    out = []
    for match in mol.GetSubstructMatches(pattern, uniquify=False, maxMatches=100000):
        atoms = tuple(match[p] for p in positions)
        atoms = min(atoms, atoms[::-1])  # i-j and j-i (i-j-k and k-j-i, ...) are one constraint
        if len(set(atoms)) == len(atoms) and atoms not in out:
            out.append(atoms)
    if not out and not entry.get("optional", False):
        raise ValueError(f"{where}: {entry.get('smarts') or entry.get('element')!r} matches no atom of the "
                         f"structure (add `optional: true` if it may not apply to every molecule){_any_bond_hint(mol, pattern)}")
    return out


def _any_bond_hint(mol: Chem.Mol, pattern: Chem.Mol) -> str:
    """The pattern with every bond as ``~`` when that matches: metal bonds are usually dative, which
    ``-`` (single) does not match."""
    if not pattern.GetNumBonds():
        return ""
    relaxed = Chem.RWMol(pattern)
    any_bond = Chem.MolFromSmarts("*~*").GetBondWithIdx(0)
    for bond in pattern.GetBonds():
        relaxed.ReplaceBond(bond.GetIdx(), any_bond, preserveProps=False)
    if not mol.HasSubstructMatch(relaxed):
        return ""
    return (f"; it matches with any bond type, {Chem.MolToSmarts(relaxed)!r}: `~` is any bond, while `-` is single "
            f"only and bonds to a metal are usually dative (`<-` from the metal's side)")


def _within(mol: Chem.Mol, atoms: set[int], shells: int) -> set[int]:
    for _ in range(shells):
        atoms = atoms | {n.GetIdx() for a in atoms for n in mol.GetAtomWithIdx(a).GetNeighbors()}
    return atoms


def resolve(spec: dict, mol: Chem.Mol, coords: np.ndarray) -> dict:
    """The constraints of ``spec`` for one molecule and its start geometry, with plain 0-based indices::

        {"freeze": [i, ...],                                   exact
         "bonds": [{"atoms": [i, j], "value": d, "force_constant": k | None}, ...],   also angles, dihedrals
         "labels": {...}}                                       e.g. "Ni4", for the run record

    A frozen group with a force constant is returned as its pairwise distances under ``bonds``.
    """
    spec = validate(spec)
    default_k = spec.get("force_constant")
    coords = np.asarray(coords, dtype=float)
    frozen: set[int] = set()
    restrained: list[tuple[set[int], float]] = []
    out: dict[str, Any] = {"freeze": [], "bonds": [], "angles": [], "dihedrals": []}
    for kind, n in KINDS.items():
        for i, entry in enumerate(spec.get(kind) or []):
            entry = {"smarts": entry} if isinstance(entry, str) else entry
            where = f"constraints.{kind}[{i}]"
            k = entry.get("force_constant", default_k)
            matches = _matches(mol, entry, where)
            if kind == "freeze":
                atoms = _within(mol, {a for m in matches for a in m}, entry.get("neighbors", 0))
                if k is None:
                    frozen |= atoms
                else:
                    restrained.append((atoms, float(k)))
                continue
            for atoms in matches:
                value = entry.get("value")
                out[kind].append({"atoms": list(atoms), "force_constant": None if k is None else float(k),
                                  "value": round(float(value) if value is not None else MEASURE[n](coords, *atoms), 4)})
    seen = {tuple(b["atoms"]) for b in out["bonds"]}
    for atoms, k in restrained:
        for pair in combinations(sorted(atoms), 2):
            if pair not in seen:
                seen.add(pair)
                out["bonds"].append({"atoms": list(pair), "force_constant": k, "value": round(distance(coords, *pair), 4)})
    out["freeze"] = sorted(frozen)
    for kind in ("bonds", "angles", "dihedrals"):  # already fixed by the frozen atoms; ORCA loosens redundant sets
        out[kind] = [c for c in out[kind] if not set(c["atoms"]) <= frozen]
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    used = set(out["freeze"]) | {a for kind in ("bonds", "angles", "dihedrals") for c in out[kind] for a in c["atoms"]}
    out["labels"] = {str(a): f"{symbols[a]}{a}" for a in sorted(used)}
    return out


def is_empty(resolved: dict) -> bool:
    return not any(resolved.get(kind) for kind in KINDS)
