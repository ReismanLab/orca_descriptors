"""YAML configuration.

Relative paths in the YAML are resolved against the config file's folder.
``bceg submit`` applies command-line selections (SMILES file, transformations,
methods, multiplicity) with :func:`select` and stores the fully resolved result
as a snapshot next to the job scripts, which the cluster jobs then read.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from bceg.layout import check_name
from bceg.refine import REFINERS, Refiner
from bceg.search import SEARCHERS, Searcher
from bceg.transforms import IDENTITY, Transformation

TOP_LEVEL_KEYS = {"inputs", "results_dir", "replicates", "transformations", "methods", "post_optimization", "slurm"}
TRANSFORMATION_KEYS = {"script", "params", "charge", "multiplicity"}
SLURM_KEYS = {"mode", "setup", "tasks_per_job", "array_throttle", "command", "cpu", "gpu"}


@dataclass
class Method:
    name: str
    searcher: Searcher
    slurm: dict[str, Any] = field(default_factory=dict)  # per-method overrides of the slurm resource profile


def _transformation_spec(spec: Any) -> dict:
    """``null`` | ``"path.py"`` | ``{script, params, charge, multiplicity}`` -> dict."""
    return {"script": spec} if spec is None or isinstance(spec, str) else dict(spec)


def read_raw(path: str | Path) -> dict:
    """Read a config YAML and make every path in it absolute."""
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text()) or {}
    unknown = set(raw) - TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(f"{path}: unknown top-level keys {sorted(unknown)}")
    if "results_dir" not in raw:
        raise ValueError(f"{path}: 'results_dir' is required (e.g. ../DATA)")

    def rel(p):
        return None if p is None else str((path.parent / p).resolve())

    raw["inputs"] = rel(raw.get("inputs"))
    raw["results_dir"] = rel(raw["results_dir"])
    transformations = {}
    for name, spec in (raw.get("transformations") or {IDENTITY: None}).items():
        spec = _transformation_spec(spec)
        spec["script"] = rel(spec.get("script"))
        transformations[name] = spec
    raw["transformations"] = transformations
    return raw


def select(raw: dict, smiles: str | Path | None = None, transformations: list[str] | None = None,
           methods: list[str] | None = None, multiplicity: int | None = None) -> dict:
    """Apply command-line selections to a raw config (returns a copy).

    A ``multiplicity`` different from the one configured for a transformation
    renames its folder to ``<name>_m<multiplicity>`` so results of different
    spin states never share a folder.
    """
    raw = copy.deepcopy(raw)
    if smiles is not None:
        raw["inputs"] = str(Path(smiles).resolve())
    if transformations:
        missing = set(transformations) - set(raw["transformations"])
        if missing:
            raise ValueError(f"Unknown transformation(s) {sorted(missing)}; configured: {sorted(raw['transformations'])}")
        raw["transformations"] = {t: raw["transformations"][t] for t in transformations}
    if methods:
        missing = set(methods) - set(raw.get("methods") or {})
        if missing:
            raise ValueError(f"Unknown method(s) {sorted(missing)}; configured: {sorted(raw.get('methods') or {})}")
        raw["methods"] = {m: raw["methods"][m] for m in methods}
    if multiplicity is not None:
        renamed = {}
        for name, spec in raw["transformations"].items():
            spec = dict(spec)
            if multiplicity != (spec.get("multiplicity") or 1):
                name = f"{name}_m{multiplicity}"
            spec["multiplicity"] = multiplicity
            renamed[name] = spec
        raw["transformations"] = renamed
    return raw


def write_raw(raw: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False))


@dataclass
class Config:
    path: Path
    inputs: Path | None
    results_dir: Path
    replicates: int
    transformations: dict[str, Transformation]
    methods: dict[str, Method]
    refiner: Refiner | None
    refiner_slurm: dict[str, Any]
    slurm: dict[str, Any]

    @property
    def chained(self) -> bool:
        """True (default): each job runs search then post-optimisation; False: separate job arrays."""
        mode = self.slurm.get("mode", "chained")
        if mode not in ("chained", "split"):
            raise ValueError(f"slurm.mode must be 'chained' or 'split', got {mode!r}")
        return mode == "chained"

    @classmethod
    def load(cls, path: str | Path) -> Config:
        return cls.from_raw(read_raw(path), Path(path).resolve())

    @classmethod
    def from_raw(cls, raw: dict, path: Path) -> Config:
        transformations = {}
        for name, spec in raw["transformations"].items():
            spec = _transformation_spec(spec)
            unknown = set(spec) - TRANSFORMATION_KEYS
            if unknown:
                raise ValueError(f"transformation {name}: unknown keys {sorted(unknown)}")
            transformations[check_name(name, "transformation")] = Transformation(
                name, spec.get("script"), spec.get("params") or {}, spec.get("charge"), spec.get("multiplicity"))

        methods = {}
        for name, spec in (raw.get("methods") or {}).items():
            spec = dict(spec)
            searcher_name = spec.pop("searcher")
            if searcher_name not in SEARCHERS:
                raise ValueError(f"method {name}: unknown searcher {searcher_name!r}; available: {sorted(SEARCHERS)}")
            searcher = SEARCHERS[searcher_name](**(spec.pop("params", None) or {}))
            methods[check_name(name, "method")] = Method(name, searcher, spec.pop("slurm", None) or {})
            if spec:
                raise ValueError(f"method {name}: unknown keys {sorted(spec)}")
        if not methods:
            raise ValueError(f"{path}: no methods defined")

        post = raw.get("post_optimization", {"refiner": "uma"})
        refiner, refiner_slurm = None, {}
        if post:
            post = dict(post)
            refiner = REFINERS[post.pop("refiner", "uma")](**(post.pop("params", None) or {}))
            refiner_slurm = post.pop("slurm", None) or {}
            if post:
                raise ValueError(f"post_optimization: unknown keys {sorted(post)}")

        slurm = raw.get("slurm") or {}
        unknown = set(slurm) - SLURM_KEYS
        if unknown:
            raise ValueError(f"slurm: unknown keys {sorted(unknown)}; allowed: {sorted(SLURM_KEYS)}")

        return cls(
            path=path,
            inputs=None if raw.get("inputs") is None else Path(raw["inputs"]),
            results_dir=Path(raw["results_dir"]),
            replicates=int(raw.get("replicates", 3)),
            transformations=transformations,
            methods=methods,
            refiner=refiner,
            refiner_slurm=refiner_slurm,
            slurm=slurm,
        )
