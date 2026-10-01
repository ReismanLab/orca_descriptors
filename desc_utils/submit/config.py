"""The library config: one YAML per ORCA library.

Relative paths are resolved against the config file. ``submit`` freezes the resolved config (plus the
template it rendered from and the git commit of this clone) under ``<LIB>/_jobs/<timestamp>/``; every
cluster task reads that snapshot, so editing the YAML while jobs are queued changes nothing.

The bceg settings (transformations, methods, post-optimisation, bceg's SLURM block) live in the same
file, as the ``source.config`` mapping; ``source.config`` may also be the path of a separate bceg YAML.
Inline, the input list is ``source.smiles``, ``replicates`` defaults to 1, bceg's ``slurm.setup`` defaults
to the library's, and the ensembles go to ``<library_dir>/ensembles`` unless ``results_dir`` says otherwise.

Two kinds of settings, as in bceg's ``params.json``:

- **compared**: what defines the results. ``source.kind``, ``transformation``, ``method``, ``lot``, the bceg
  settings of that transformation, method and post-optimisation (and ``replicates``), the ``dft`` block and
  the template text. The first ``submit`` writes them to ``<LIB>/params.json``; a later submit with
  different values is refused with a diff. Different settings mean a different library directory.
- **recorded** (``orca`` on/off, ``run_bceg``, paths, both ``slurm`` blocks, ``condense``): kept for
  reference, never compared.
"""

from __future__ import annotations

import copy
import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from bceg.config import TOP_LEVEL_KEYS as BCEG_TOP_LEVEL_KEYS
from bceg.config import _script_path, _transformation_spec
from bceg.config import read_raw as bceg_read_raw
from bceg.provenance import make_params
from bceg.transforms import IDENTITY

TOP_KEYS = {"library_dir", "source", "orca", "dft", "slurm", "condense", "_pipeline"}
SOURCE_KINDS = {"bceg", "xyz"}
SOURCE_KEYS = {"kind", "config", "results_dir", "transformation", "method", "lot", "smiles", "run_bceg",
               "file", "charge", "multiplicity", "name"}
DFT_KEYS = {"ewin_kcal_mol", "calc_param_template", "level_of_theory"}
SLURM_KEYS = {"setup", "prepare", "orca", "array_throttle", "orca_cmd"}
PROFILE_KEYS = {"partition", "account", "time", "mem", "mem_per_cpu", "ntasks", "cpus_per_task", "nodes",
                "qos", "constraint", "gres", "extra"}

DEFAULT_TEMPLATE = Path(__file__).with_name("calcParam.inp.tmpl")
DEFAULTS: dict[str, Any] = {
    "orca": "on",
    "condense": True,
    "dft": {"ewin_kcal_mol": 5.0},
    "slurm": {
        "setup": [],
        "prepare": {"time": "00:30:00", "mem": "2G", "cpus_per_task": 1},
        "orca": {"time": "160:00:00", "mem_per_cpu": "4G", "ntasks": 16, "nodes": 1},
        "array_throttle": 50,
    },
}
REPO_ROOT = Path(__file__).resolve().parents[2]


def _check_keys(section: dict, allowed: set, where: str) -> None:
    unknown = set(section) - allowed
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}; allowed: {sorted(allowed)}")


def inline_bceg_raw(block: dict, base_dir: Path, library_dir: Path, smiles: str, default_setup: list) -> dict:
    """Normalise a bceg config written inline in the library config into the dict ``bceg.config.Config.from_raw``
    expects (what ``bceg.config.read_raw`` does for a file): absolute paths, resolved transformation scripts.
    Defaults: ``inputs`` = the library's SMILES list, ``replicates`` = 1, ``results_dir`` =
    ``<library_dir>/ensembles``, ``slurm.setup`` = the library's."""
    raw = copy.deepcopy(block or {})
    _check_keys(raw, BCEG_TOP_LEVEL_KEYS, "source.config (inline bceg config)")
    raw["inputs"] = smiles
    raw["results_dir"] = str((base_dir / raw["results_dir"]).resolve()) if raw.get("results_dir") \
        else str(Path(library_dir) / "ensembles")
    raw.setdefault("replicates", 1)
    transformations = {}
    for name, spec in (raw.get("transformations") or {IDENTITY: None}).items():
        spec = _transformation_spec(spec)
        spec["script"] = _script_path(spec.get("script"), base_dir)
        transformations[name] = spec
    raw["transformations"] = transformations
    slurm = dict(raw.get("slurm") or {})
    slurm.setdefault("setup", list(default_setup))
    raw["slurm"] = slurm
    return raw


PLACEHOLDER = "CHANGE_ME"


def find_placeholders(obj, prefix: str = "") -> list[str]:
    """Dotted paths of every value still equal to the CHANGE_ME placeholder of the example configs."""
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            hits += find_placeholders(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits += find_placeholders(v, f"{prefix}[{i}]")
    elif isinstance(obj, str) and PLACEHOLDER in obj.upper():
        hits.append(prefix)
    return hits


def read_raw(path: str | Path) -> dict:
    """Read a library config and make every path in it absolute; fill the defaults."""
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text()) or {}
    _check_keys(raw, TOP_KEYS, str(path))
    placeholders = find_placeholders(raw)
    if placeholders:
        raise ValueError(f"{path}: replace the {PLACEHOLDER} placeholder(s) at: " + ", ".join(placeholders)
                         + " (or delete the key to use the cluster default)")
    for key in ("library_dir", "source", "dft"):
        if key not in raw:
            raise ValueError(f"{path}: '{key}' is required")

    def rel(p):
        return None if p is None else str((path.parent / p).resolve())

    raw["library_dir"] = rel(raw["library_dir"])

    source = dict(raw["source"])
    _check_keys(source, SOURCE_KEYS, f"{path}: source")
    kind = source.get("kind", "bceg")
    if kind not in SOURCE_KINDS:
        raise ValueError(f"{path}: source.kind must be one of {sorted(SOURCE_KINDS)}, got {kind!r}")
    source["kind"] = kind
    if kind == "bceg":
        for key in ("config", "transformation", "method", "smiles"):
            if key not in source:
                raise ValueError(f"{path}: source.{key} is required for kind: bceg")
        source["smiles"] = rel(source["smiles"])
        setup = list(((raw.get("slurm") or {}).get("setup")) or [])
        if isinstance(source["config"], dict):
            # the bceg settings written in this file
            source["config"] = inline_bceg_raw(source["config"], path.parent, Path(raw["library_dir"]),
                                               source["smiles"], setup)
            source["results_dir"] = rel(source["results_dir"]) if source.get("results_dir") else source["config"]["results_dir"]
            source["config"]["results_dir"] = source["results_dir"]
        else:
            # a separate bceg YAML; its own results_dir unless the library says otherwise
            source["config"] = rel(source["config"])
            source["results_dir"] = rel(source["results_dir"]) if source.get("results_dir") \
                else bceg_read_raw(source["config"])["results_dir"]
        source.setdefault("run_bceg", True)
        source.setdefault("lot", None)
    else:
        for key in ("file", "smiles", "charge", "multiplicity"):
            if key not in source:
                raise ValueError(f"{path}: source.{key} is required for kind: xyz")
        source["file"] = rel(source["file"])
        source["run_bceg"] = False
    raw["source"] = source

    dft = {**DEFAULTS["dft"], **raw["dft"]}
    _check_keys(dft, DFT_KEYS, f"{path}: dft")
    if "level_of_theory" not in dft:
        raise ValueError(f"{path}: dft.level_of_theory is required")
    dft["calc_param_template"] = rel(dft.get("calc_param_template")) or str(DEFAULT_TEMPLATE)
    raw["dft"] = dft

    slurm = copy.deepcopy(DEFAULTS["slurm"])
    for key, value in (raw.get("slurm") or {}).items():
        if key in ("prepare", "orca"):
            slurm[key] = {**slurm[key], **(value or {})}
        else:
            slurm[key] = value
    _check_keys(slurm, SLURM_KEYS, f"{path}: slurm")
    for prof in ("prepare", "orca"):
        _check_keys(slurm[prof], PROFILE_KEYS, f"{path}: slurm.{prof}")
    raw["slurm"] = slurm

    orca = raw.get("orca", DEFAULTS["orca"])
    if isinstance(orca, bool):
        orca = "on" if orca else "off"
    if orca not in ("on", "off"):
        raise ValueError(f"{path}: orca must be on or off, got {orca!r}")
    raw["orca"] = orca
    raw["condense"] = bool(raw.get("condense", DEFAULTS["condense"]))
    return raw


def write_raw(raw: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False))


def git_info(repo_root: Path = REPO_ROOT) -> dict:
    """Commit and cleanliness of the clone the pipeline runs from."""
    info: dict[str, Any] = {"repo": str(repo_root)}
    if shutil.which("git") is None:
        return info | {"git_commit": None, "git_dirty": None}
    try:
        commit = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip() != ""
        return info | {"git_commit": commit, "git_dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return info | {"git_commit": None, "git_dirty": None}


@dataclass
class Config:
    path: Path
    raw: dict

    @classmethod
    def load(cls, path: str | Path) -> Config:
        return cls(Path(path).resolve(), read_raw(path))

    # ------------------------------------------------------------------ sections
    @property
    def library_dir(self) -> Path:
        return Path(self.raw["library_dir"])

    @property
    def source(self) -> dict:
        return self.raw["source"]

    @property
    def dft(self) -> dict:
        return self.raw["dft"]

    @property
    def slurm(self) -> dict:
        return self.raw["slurm"]

    @property
    def orca_on(self) -> bool:
        return self.raw["orca"] == "on"

    @property
    def condense(self) -> bool:
        return self.raw["condense"]

    @property
    def template_path(self) -> Path:
        return Path(self.dft["calc_param_template"])

    @property
    def pipeline(self) -> dict:
        """Clone and commit recorded at freeze time (empty before freezing)."""
        return self.raw.get("_pipeline") or {}

    @property
    def jobdir(self) -> Path:
        """The ``_jobs/<timestamp>`` folder a frozen config lives in."""
        return self.path.parent

    # ------------------------------------------------------------------ provenance
    def template_sha256(self) -> str:
        return hashlib.sha256(self.template_path.read_bytes()).hexdigest()

    def bceg_raw(self) -> dict | None:
        """bceg's resolved config dict (inline or read from the file), None for an xyz source."""
        if self.source["kind"] != "bceg":
            return None
        cfg = self.source["config"]
        if isinstance(cfg, dict):
            return copy.deepcopy(cfg)
        raw = bceg_read_raw(cfg)
        raw["results_dir"] = self.source["results_dir"]
        return raw

    def compared(self) -> dict:
        s = self.source
        source = {k: s[k] for k in ("kind", "transformation", "method", "lot", "file", "charge", "multiplicity") if k in s}
        braw = self.bceg_raw()
        if braw is not None:
            # only the bceg settings this library's ensembles depend on
            source["bceg"] = {
                "replicates": braw.get("replicates"),
                "transformation": braw["transformations"].get(s["transformation"]),
                "method": (braw.get("methods") or {}).get(s["method"]),
                "post_optimization": braw.get("post_optimization", {"refiner": "uma"}),
            }
        dft = dict(self.dft)
        dft["calc_param_template"] = {"path": str(self.template_path), "sha256": self.template_sha256()}
        return {"source": source, "dft": dft}

    def recorded(self) -> dict:
        s = self.source
        paths = {k: s.get(k) for k in ("results_dir", "smiles") if k in s}
        if isinstance(s.get("config"), str):
            paths["config"] = s["config"]
        braw = self.bceg_raw()
        return {"orca": self.raw["orca"], "run_bceg": s.get("run_bceg", False), "source_paths": paths,
                "bceg_slurm": (braw or {}).get("slurm"), "slurm": self.slurm, "condense": self.condense}

    def params(self) -> dict:
        """bceg-style params document: ``sha256`` + ``compared`` + ``recorded``."""
        return make_params(self.compared(), self.recorded())

    def library_params_path(self) -> Path:
        return self.library_dir / "params.json"

    # ------------------------------------------------------------------ freezing
    def freeze(self, jobdir: Path) -> Config:
        """Write the resolved config, the template and the git commit into ``jobdir``; return the frozen copy."""
        jobdir.mkdir(parents=True, exist_ok=True)
        raw = copy.deepcopy(self.raw)
        template_copy = jobdir / "calcParam.inp.tmpl"
        shutil.copyfile(self.template_path, template_copy)
        raw["dft"]["calc_param_template"] = str(template_copy)
        raw["_pipeline"] = git_info() | {"frozen_from": str(self.path), "template_sha256": self.template_sha256()}
        frozen = jobdir / "config.yaml"
        write_raw(raw, frozen)
        return Config(frozen, raw)
