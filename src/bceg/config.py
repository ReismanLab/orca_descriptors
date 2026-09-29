"""YAML configuration. Relative paths are resolved against the config file's folder."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from bceg.layout import check_name
from bceg.refine import REFINERS, Refiner
from bceg.search import SEARCHERS, Searcher
from bceg.transforms import IDENTITY, Transformation


@dataclass
class Method:
    name: str
    searcher: Searcher
    slurm: dict[str, Any] = field(default_factory=dict)  # per-method overrides of the slurm resource profile


@dataclass
class Config:
    path: Path
    inputs: Path
    results_dir: Path
    replicates: int
    transformations: dict[str, Transformation]
    methods: dict[str, Method]
    refiner: Refiner | None
    refiner_slurm: dict[str, Any]
    slurm: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> Config:
        path = Path(path).resolve()
        raw = yaml.safe_load(path.read_text()) or {}
        known = {"inputs", "results_dir", "replicates", "transformations", "methods", "post_optimization", "slurm"}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"{path}: unknown top-level keys {sorted(unknown)}")
        here = path.parent

        def rel(p: str | None) -> Path | None:
            return None if p is None else (here / p).resolve()

        transformations = {}
        for name, script in (raw.get("transformations") or {IDENTITY: None}).items():
            transformations[check_name(name, "transformation")] = Transformation(name, rel(script))

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

        return cls(
            path=path,
            inputs=rel(raw["inputs"]),
            results_dir=rel(raw.get("results_dir", "results")),
            replicates=int(raw.get("replicates", 3)),
            transformations=transformations,
            methods=methods,
            refiner=refiner,
            refiner_slurm=refiner_slurm,
            slurm=raw.get("slurm") or {},
        )
