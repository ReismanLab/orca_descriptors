"""Load and validate a descriptor specification file.

A spec is a YAML (or JSON) document with these top-level sections:

``name``          dataset name, used in output file names
``data``          where conformer folders live and how they are laid out
``atoms``         ordered map ``label -> expected element`` (``null`` = unconstrained)
``numbering``     how the ``label -> atom index`` table is produced (backend + options)
``descriptors``   ``orca`` and ``descripytor`` families, every term written with labels
``aggregate``     Boltzmann aggregation over conformers of one molecule
``filter``        keyword arguments for ``data_extraction.filter_numeric_descriptors``
``output``        output file names

Everything downstream of the numbering table works with the contract
``{label: 1-based atom index}`` per conformer; :func:`atom_map` and
:func:`resolve_terms` are the two helpers that apply it.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# --------------------------------------------------------------------------- constants

#: DescriPyTor "Schema B" keys that take atom terms, with their arity.
DESCRIPYTOR_TERM_ARITY: dict[str, int] = {
    "charges": 1,
    "charge_diff": 2,
    "sterimol": 2,
    "bond_length": 2,
    "stretching_vibration": 2,
    "bending_vibration": 2,
    "dipole": 3,
    "bond_angle": 3,
}
#: DescriPyTor scalar options accepted next to the term keys.
DESCRIPYTOR_SCALAR_KEYS: dict[str, Any] = {
    "stretch_threshold": 1400,
    "upper_stretch_threshold": 3500,
    "bend_threshold": 1300,
    "radii": "CPK",
    "drop_atoms": [],
    "input": {"kind": "json"},  # json = <conf>/<data.condensed_file>; feather = legacy orca2gaussian path
}

#: ORCA-side (morfeus / OrcaReader) families and the arity of each term list.
ORCA_XYZ_TERM_ARITY: dict[str, int | None] = {
    "atoms": 1,
    "bonds": 2,
    "angles": 3,
    "dihedrals": 4,
    "quadrants": None,  # >= 4 labels
}
ORCA_FAMILIES = ("xyz", "charges", "nbo", "redox")
NBO_SPIN_CHOICES = ("alpha", "beta", "both")  # od.NBOProps.SPIN_CHOICES

NUMBERING_BACKENDS = ("smarts", "manual", "first_n", "molalign")
AGGREGATE_STATS = ("min", "max", "lowE", "boltz")

DEFAULTS: dict[str, Any] = {
    "data": {
        "layout": "molecule/conformer",
        "conformer_glob": "conf_*",
        "geometry_file": "optimization.xyz",
        "molecule_id_column": "inchi",
        # desc_utils.condense output read instead of the raw ORCA files when present; null = raw files only
        "condensed_file": "orca_condensed.json",
    },
    "numbering": {
        "index_base": 1,
        "consistency": "per_molecule",
        "orient": [],
        "annotators": [],
        "smiles": {"mode": "whole_molecule", "strip": []},
        "energy": {"basename": "singlepoint", "column": "SCF_energy"},
        "smarts": {"match_selection": "first"},
    },
    "orca_xyz": {
        "vbur_radii": [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        "use_cached_json": True,
        "quadrants": [],
        "angles": [],
        "dihedrals": [],
    },
    "orca_charges": {"basename": "singlepoint"},
    # `spin`: which NBO spin set to read from open-shell blocks (alpha | beta | both); see od.NBOProps.
    "orca_nbo": {"basename": "singlepoint", "homo_lumo_rank_range": [1, 2, 3, 4, 5], "drop_analysis_info": True,
                 "nbo_block_index": 0, "spin": "alpha"},
    "orca_redox": {"prefixes": ["cm5"], "nbo_block_map": {"singlepoint": 1, "oxidation": 2, "reduction": 3}},
    "aggregate": {
        "group_by": "inchi",
        "energy_column": "SCF_energy",
        "temperature_K": 298.15,
        "kB_hartree_per_K": 3.166811563e-6,
        "stats": list(AGGREGATE_STATS),
        "carry_first": ["smiles", "ligand_class"],
        "exclude_columns": [],
    },
    # The noise filter (correlation / variance pruning) is opt-in: the library should stay complete and
    # reproducible; enable it in the spec or with `aggregate --filter` when preparing a modelling set.
    "filter": {"enabled": False},
    "output": {
        "numbering_csv": "atom_numbering_results.csv",
        "per_conformer_csv": "combined_descriptors.csv",
        "aggregated_csv": "{date}_{name}_descriptors.csv",
    },
}


class SpecError(ValueError):
    """Raised with every problem found in a spec, not just the first one."""

    def __init__(self, problems: Sequence[str]):
        self.problems = list(problems)
        super().__init__("Invalid spec:\n  - " + "\n  - ".join(self.problems))


@dataclass(frozen=True)
class Spec:
    name: str
    data: dict
    atoms: dict[str, str | None]
    numbering: dict
    orca: dict | None
    descripytor: dict | None
    aggregate: dict
    filter: dict
    output: dict
    path: Path | None = None
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def labels(self) -> list[str]:
        return list(self.atoms.keys())

    @property
    def root(self) -> Path:
        return Path(self.data["root"])


# --------------------------------------------------------------------------- helpers

def _merge_defaults(cfg: Mapping | None, defaults: Mapping) -> dict:
    out = dict(defaults)
    out.update(cfg or {})
    return out


def _is_label_list(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value)


def term_label(term: Sequence[str], sep: str = "_") -> str:
    """Join labels into the column token used by the ORCA-side readers (``C1_N1``)."""
    return sep.join(term)


def atom_map(row: Mapping[str, Any], labels: Iterable[str]) -> dict[str, int]:
    """``{label: int(row[label])}``, the contract every extractor consumes."""
    return {label: int(row[label]) for label in labels}


def resolve_terms(terms: Sequence[Sequence[str] | str], row: Mapping[str, Any]) -> list[tuple[int, ...]]:
    """Turn label terms into index tuples for one conformer row.

    A bare string is treated as a 1-tuple, so ``charges: [H1, H2]`` and
    ``bonds: [[C1, N1]]`` both work.
    """
    out: list[tuple[int, ...]] = []
    for term in terms:
        if isinstance(term, str):
            term = (term,)
        out.append(tuple(int(row[label]) for label in term))
    return out


# --------------------------------------------------------------------------- validation

def _check_terms(section: str, key: str, terms: Any, arity: int | None, labels: set[str], problems: list[str]) -> list[list[str]]:
    """Normalise a term list to ``list[list[str]]`` and record problems."""
    if terms is None:
        return []
    if not isinstance(terms, (list, tuple)):
        problems.append(f"{section}.{key}: expected a list, got {type(terms).__name__}")
        return []
    norm: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for i, term in enumerate(terms):
        if isinstance(term, str):
            term = [term]
        if not _is_label_list(term):
            problems.append(f"{section}.{key}[{i}]: expected atom labels, got {term!r}")
            continue
        term = list(term)
        if arity is not None and len(term) != arity:
            problems.append(f"{section}.{key}[{i}]: expected {arity} labels, got {len(term)} ({term})")
        if arity is None and len(term) < 4:
            problems.append(f"{section}.{key}[{i}]: quadrants need at least 4 labels, got {len(term)}")
        unknown = [l for l in term if l not in labels]
        if unknown:
            problems.append(f"{section}.{key}[{i}]: unknown atom label(s) {unknown}; declare them under `atoms`")
        if tuple(term) in seen:
            problems.append(f"{section}.{key}[{i}]: duplicate term {term}")
        seen.add(tuple(term))
        norm.append(term)
    return norm


def _validate_smarts_stages(numbering: dict, labels: list[str], problems: list[str]) -> None:
    smarts_cfg = _merge_defaults(numbering.get("smarts"), DEFAULTS["numbering"]["smarts"])
    numbering["smarts"] = smarts_cfg
    if smarts_cfg.get("match_selection") not in ("first", "lowest_indices"):
        problems.append("numbering.smarts.match_selection must be 'first' or 'lowest_indices'")
    stages = smarts_cfg.get("stages")
    if not stages:
        problems.append("numbering.smarts.stages must list at least one stage")
        return
    try:
        from rdkit import Chem  # noqa: WPS433 (optional at validation time)
    except ImportError:  # pragma: no cover
        Chem = None
    assigned: set[str] = set()
    for i, stage in enumerate(stages):
        where = f"numbering.smarts.stages[{i}]"
        if not isinstance(stage, dict):
            problems.append(f"{where}: expected a mapping")
            continue
        stage.setdefault("name", f"stage_{i}")
        stage.setdefault("required", True)
        patterns = stage.get("smarts")
        if isinstance(patterns, str):
            patterns = [patterns]
        if not patterns or not all(isinstance(p, str) for p in patterns):
            problems.append(f"{where}: `smarts` must be a quoted SMARTS string or a list of them")
            continue
        stage["smarts"] = list(patterns)
        roles = stage.get("roles")
        if not isinstance(roles, dict) or not roles:
            problems.append(f"{where}: `roles` must map atom labels to match positions")
            continue
        for label, pos in roles.items():
            if label not in labels:
                problems.append(f"{where}.roles: unknown label {label!r}")
            if not isinstance(pos, int) or pos < 0:
                problems.append(f"{where}.roles.{label}: position must be a non-negative integer")
        positions = [p for p in roles.values() if isinstance(p, int)]
        if len(set(positions)) != len(positions):
            problems.append(f"{where}.roles: two labels share one match position")
        if Chem is not None:
            for p in patterns:
                patt = Chem.MolFromSmarts(p)
                if patt is None:
                    problems.append(f"{where}: SMARTS does not parse: {p!r} (is it quoted? '#' starts a YAML comment)")
                    continue
                n = patt.GetNumAtoms()
                bad = [f"{l}:{pos}" for l, pos in roles.items() if isinstance(pos, int) and pos >= n]
                if bad:
                    problems.append(f"{where}: positions {bad} exceed the {n} atoms of SMARTS {p!r}")
        assigned.update(roles.keys())
    missing = [l for l in labels if l not in assigned]
    if missing:
        problems.append(f"numbering.smarts: labels {missing} are never assigned by any stage")


def _validate_numbering(numbering: dict, labels: list[str], problems: list[str]) -> dict:
    numbering = _merge_defaults(numbering, DEFAULTS["numbering"])
    backend = numbering.get("backend")
    if backend not in NUMBERING_BACKENDS:
        problems.append(f"numbering.backend must be one of {NUMBERING_BACKENDS}, got {backend!r}")
    if numbering.get("index_base") not in (0, 1):
        problems.append("numbering.index_base must be 0 or 1")
    if numbering.get("consistency") not in ("per_molecule", "none"):
        problems.append("numbering.consistency must be 'per_molecule' or 'none'")
    if backend == "smarts":
        _validate_smarts_stages(numbering, labels, problems)
    elif backend == "manual":
        manual = numbering.get("manual") or {}
        if not manual.get("csv"):
            problems.append("numbering.manual.csv is required for the manual backend")
        manual.setdefault("key_column", "path")
        manual.setdefault("index_base", 1)
        numbering["manual"] = manual
    elif backend == "molalign":
        molalign = numbering.get("molalign") or {}
        if not molalign.get("reference"):
            problems.append("numbering.molalign.reference (path to the reference structure) is required")
        numbering["molalign"] = molalign
    for i, rule in enumerate(numbering.get("orient") or []):
        where = f"numbering.orient[{i}]"
        if not isinstance(rule, dict) or rule.get("rule") != "closest":
            problems.append(f"{where}: only rule 'closest' is supported")
            continue
        move, to = rule.get("move") or [], rule.get("to") or []
        if len(move) != len(to) or not move:
            problems.append(f"{where}: `move` and `to` must be non-empty lists of equal length")
        for l in list(move) + list(to):
            if l not in labels:
                problems.append(f"{where}: unknown label {l!r}")
    smiles = _merge_defaults(numbering.get("smiles"), DEFAULTS["numbering"]["smiles"])
    if smiles.get("mode") not in ("whole_molecule", "strip_atoms"):
        problems.append("numbering.smiles.mode must be 'whole_molecule' or 'strip_atoms'")
    for l in smiles.get("strip") or []:
        if l not in labels:
            problems.append(f"numbering.smiles.strip: unknown label {l!r}")
    numbering["smiles"] = smiles
    numbering["energy"] = _merge_defaults(numbering.get("energy"), DEFAULTS["numbering"]["energy"])
    return numbering


def _validate_orca(orca: dict | None, labels: set[str], problems: list[str]) -> dict | None:
    if orca is None:
        return None
    out: dict = {}
    for fam in orca:
        if fam not in ORCA_FAMILIES:
            problems.append(f"descriptors.orca: unknown family {fam!r}; expected {ORCA_FAMILIES}")
    xyz = orca.get("xyz")
    if xyz is not None:
        xyz = _merge_defaults(xyz, DEFAULTS["orca_xyz"])
        for key, arity in ORCA_XYZ_TERM_ARITY.items():
            xyz[key] = _check_terms("descriptors.orca.xyz", key, xyz.get(key, []), arity, labels, problems)
        if not xyz["atoms"]:
            problems.append("descriptors.orca.xyz.atoms must list at least one label")
        xyz["atoms"] = [t[0] for t in xyz["atoms"]]
        out["xyz"] = xyz
    for fam, defaults in (("charges", DEFAULTS["orca_charges"]), ("nbo", DEFAULTS["orca_nbo"]), ("redox", DEFAULTS["orca_redox"])):
        cfg = orca.get(fam)
        if cfg is None:
            continue
        cfg = _merge_defaults(cfg, defaults)
        atoms = _check_terms(f"descriptors.orca.{fam}", "atoms", cfg.get("atoms"), 1, labels, problems)
        if not atoms:
            problems.append(f"descriptors.orca.{fam}.atoms must list at least one label")
        cfg["atoms"] = [t[0] for t in atoms]
        if fam == "nbo" and cfg.get("spin") not in NBO_SPIN_CHOICES:
            problems.append(f"descriptors.orca.nbo.spin must be one of {list(NBO_SPIN_CHOICES)}, got {cfg.get('spin')!r}")
        out[fam] = cfg
    return out


def _validate_descripytor(cfg: dict | None, labels: set[str], problems: list[str]) -> dict | None:
    if cfg is None:
        return None
    out = dict(DESCRIPYTOR_SCALAR_KEYS)
    out["drop_atoms"] = list(DESCRIPYTOR_SCALAR_KEYS["drop_atoms"])
    out["input"] = dict(DESCRIPYTOR_SCALAR_KEYS["input"])
    for key, value in cfg.items():
        if key in DESCRIPYTOR_TERM_ARITY:
            out[key] = _check_terms("descriptors.descripytor", key, value, DESCRIPYTOR_TERM_ARITY[key], labels, problems)
        elif key in DESCRIPYTOR_SCALAR_KEYS:
            out[key] = value
        elif key == "connectivity":
            out[key] = value
        else:
            problems.append(
                f"descriptors.descripytor: unknown key {key!r}; expected one of "
                f"{sorted(DESCRIPYTOR_TERM_ARITY) + sorted(DESCRIPYTOR_SCALAR_KEYS)}"
            )
    for key in DESCRIPYTOR_TERM_ARITY:
        out.setdefault(key, [])
    for l in out.get("drop_atoms") or []:
        if l not in labels:
            problems.append(f"descriptors.descripytor.drop_atoms: unknown label {l!r}")
    if not isinstance(out["input"], dict) or out["input"].get("kind") not in ("feather", "json"):
        problems.append("descriptors.descripytor.input.kind must be 'feather' or 'json'")
    if not any(out[k] for k in DESCRIPYTOR_TERM_ARITY):
        problems.append("descriptors.descripytor: no descriptor terms given")
    return out


def validate_spec(raw: Mapping[str, Any], path: Path | None = None) -> Spec:
    """Validate a parsed spec document and return a :class:`Spec` with defaults filled in."""
    problems: list[str] = []
    if not isinstance(raw, Mapping):
        raise SpecError(["spec document must be a mapping"])

    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        problems.append("`name` is required")
        name = "unnamed"
    if not re.fullmatch(r"[A-Za-z0-9_.\-]+", name):
        problems.append("`name` may only contain letters, digits, '_', '-' and '.' (it is used in file names)")

    data = _merge_defaults(raw.get("data"), DEFAULTS["data"])
    if not data.get("root"):
        problems.append("data.root is required")
    if data.get("layout") not in ("molecule/conformer", "flat"):
        problems.append("data.layout must be 'molecule/conformer' or 'flat'")

    atoms_raw = raw.get("atoms")
    atoms: dict[str, str | None] = {}
    if not isinstance(atoms_raw, Mapping) or not atoms_raw:
        problems.append("`atoms` must be a non-empty mapping label -> element (or null)")
    else:
        for label, element in atoms_raw.items():
            if not isinstance(label, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", label):
                problems.append(f"atoms: label {label!r} must be alphanumeric and start with a letter")
            if element is not None and not (isinstance(element, str) and re.fullmatch(r"[A-Z][a-z]?", element)):
                problems.append(f"atoms.{label}: element must be a symbol like 'Ni' or null, got {element!r}")
            atoms[str(label)] = element
    labels = list(atoms)
    label_set = set(labels)

    numbering = _validate_numbering(raw.get("numbering") or {}, labels, problems)

    descriptors = raw.get("descriptors") or {}
    for key in descriptors:
        if key not in ("orca", "descripytor"):
            problems.append(f"descriptors: unknown engine {key!r}; expected 'orca' and/or 'descripytor'")
    orca = _validate_orca(descriptors.get("orca"), label_set, problems)
    descripytor = _validate_descripytor(descriptors.get("descripytor"), label_set, problems)
    if orca is None and descripytor is None:
        problems.append("descriptors: at least one of `orca` or `descripytor` is required")

    aggregate = _merge_defaults(raw.get("aggregate"), DEFAULTS["aggregate"])
    bad_stats = [s for s in aggregate.get("stats", []) if s not in AGGREGATE_STATS]
    if bad_stats:
        problems.append(f"aggregate.stats: unsupported {bad_stats}; choose from {AGGREGATE_STATS}")
    if not (isinstance(aggregate.get("temperature_K"), (int, float)) and aggregate["temperature_K"] > 0):
        problems.append("aggregate.temperature_K must be a positive number")

    filt = _merge_defaults(raw.get("filter"), DEFAULTS["filter"])
    output = _merge_defaults(raw.get("output"), DEFAULTS["output"])

    if problems:
        raise SpecError(problems)
    return Spec(
        name=name, data=data, atoms=atoms, numbering=numbering, orca=orca,
        descripytor=descripytor, aggregate=aggregate, filter=filt, output=output,
        path=path, raw=dict(raw),
    )


def load_spec(path: str | Path) -> Spec:
    """Read a YAML or JSON spec file and validate it."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        if path.suffix.lower() == ".json":
            raw = json.loads(text)
        else:
            import yaml

            raw = yaml.safe_load(text)
    except Exception as exc:  # yaml.YAMLError / json.JSONDecodeError
        hint = ""
        if "[" in text and "smarts" in text and "'[" not in text and '"[' not in text:
            hint = " (hint: SMARTS strings must be quoted; '#' starts a YAML comment and '[' starts a list)"
        raise SpecError([f"{path}: could not parse file{hint}: {str(exc).splitlines()[0]}"]) from exc
    return validate_spec(raw, path=path)
