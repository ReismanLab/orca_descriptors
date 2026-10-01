"""The library config: one YAML per ORCA library, one block per stage of the pipeline.

::

    library_dir, smiles, ensembles_dir, run_bceg          where things are, whether bceg is called
    transformation       1. SMILES -> structure computed (name = depth-1 folder of the ensembles)
    pre_refinement       2. pre-optimisation of the start geometry handed to the search (xtb | uma | none)
    conformer_search     3. the searcher and its parameters (name = parameter set = method folder)
    post_optimization    4. refinement of every conformer (uma | null)
    pruning              5. which conformers get an ORCA job (energy window on the stage-4 energies; null = all)
    orca, dft            6. the ORCA compound job (level_of_theory, template)
    slurm, condense      resources: setup lines, bceg's profiles (slurm.ensembles), prepare and orca profiles

Stages 1 to 4 are run by bceg: ``Config.bceg_raw()`` turns the blocks into bceg's own config format
(one transformation, one method, ``replicates: 1``). Relative paths are resolved against the config
file. ``submit`` freezes the resolved config (plus the template it rendered from and the git commit of
this clone) under ``<LIB>/_jobs/<timestamp>/``; every cluster task reads that snapshot.

Two kinds of settings, as in bceg's ``params.json``:

- **compared**: what defines the results: the six stage blocks (with the searcher's and refiner's
  defaults filled in, and the template's text as a sha256). The first ``submit`` writes them to
  ``<LIB>/params.json``; a later submit with different values is refused with a diff. Different
  settings mean a different library directory.
- **recorded** (``orca`` on/off, ``run_bceg``, the paths, ``slurm``, ``condense``): kept for reference,
  never compared.
"""

from __future__ import annotations

import copy
import hashlib
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from bceg.config import SLURM_KEYS as BCEG_SLURM_KEYS
from bceg.config import Config as BcegConfig
from bceg.config import _script_path
from bceg.provenance import make_params
from bceg.refine import REFINERS
from bceg.search import SEARCHERS
from bceg.transforms import IDENTITY
from bceg.xtb import METHOD_FLAGS as XTB_METHODS
from bceg.xtb import OPT_LEVELS as XTB_LEVELS

STAGES = ("transformation", "pre_refinement", "conformer_search", "post_optimization", "pruning", "dft")
TOP_KEYS = {"library_dir", "smiles", "ensembles_dir", "run_bceg", *STAGES, "orca", "slurm", "condense", "_pipeline"}
TRANSFORMATION_KEYS = {"name", "script", "params", "charge", "multiplicity"}
PRE_REFINEMENT_KEYS = {"method", "params", "start_geometry", "start_ff", "max_metal_ligand_distance", "max_attempts",
                       "xtb_executable"}
PRE_REFINEMENT_METHODS = ("xtb", "uma", "none")
XTB_PARAMS = {"gfn": "gfn2", "level": "normal"}
#: pre_refinement key -> bceg start_* parameter (bceg.search.base.START_DEFAULTS)
EMBEDDING_KEYS = {"start_geometry": "start_geometry", "start_ff": "start_ff",
                  "max_metal_ligand_distance": "max_metal_ligand_distance", "max_attempts": "start_max_attempts",
                  "xtb_executable": "xtb_executable"}
SEARCH_KEYS = {"name", "searcher", "params", "slurm"}
POST_OPT_KEYS = {"refiner", "params", "slurm"}
PRUNING_KEYS = {"energy_window_kcal_mol"}
DFT_KEYS = {"calc_param_template", "level_of_theory"}
SLURM_KEYS = {"setup", "ensembles", "prepare", "orca", "array_throttle", "orca_cmd"}
PROFILE_KEYS = {"partition", "account", "time", "mem", "mem_per_cpu", "ntasks", "cpus_per_task", "nodes",
                "qos", "constraint", "gres", "extra"}

DEFAULT_TEMPLATE = Path(__file__).with_name("calcParam.inp.tmpl")
DEFAULTS: dict[str, Any] = {
    "orca": "on",
    "condense": True,
    "run_bceg": True,
    "pruning": {"energy_window_kcal_mol": None},
    "slurm": {
        "setup": [],
        "ensembles": {},
        "prepare": {"time": "00:30:00", "mem": "2G", "cpus_per_task": 1},
        "orca": {"time": "160:00:00", "mem_per_cpu": "4G", "ntasks": 16, "nodes": 1},
        "array_throttle": 50,
    },
}
REPO_ROOT = Path(__file__).resolve().parents[2]
PLACEHOLDER = "CHANGE_ME"


def _check_keys(section: dict, allowed: set, where: str) -> None:
    if not isinstance(section, dict):
        raise ValueError(f"{where}: expected a mapping, got {section!r}")
    unknown = set(section) - allowed
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}; allowed: {sorted(allowed)}")


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


def searcher_has_start_step(searcher: str) -> bool:
    """CREST, GOAT and molassembler start from one pre-optimised geometry; RDKit ETKDG and LoQI do not."""
    return "start_preopt" in SEARCHERS[searcher].defaults


# --------------------------------------------------------------------------- the stage blocks
def _transformation(block, where: str, base_dir: Path) -> dict:
    """``none`` | ``{name, script, params, charge, multiplicity}`` -> normalised dict with an absolute script."""
    if block is None or block == IDENTITY:
        block = {"name": IDENTITY}
    if isinstance(block, str):
        raise ValueError(f"{where}: write `transformation: none` or a mapping with name and script, got {block!r}")
    _check_keys(block, TRANSFORMATION_KEYS, where)
    out = {"name": block.get("name"), "script": block.get("script"), "params": dict(block.get("params") or {}),
           "charge": block.get("charge"), "multiplicity": block.get("multiplicity")}
    if out["name"] is None:
        if out["script"] is not None:
            raise ValueError(f"{where}: `name` is required when a script is given (it names the ensembles folder)")
        out["name"] = IDENTITY
    out["script"] = _script_path(out["script"], base_dir)
    return out


def _pre_refinement(block, where: str, searcher: str) -> dict:
    """Defaults: xtb GFN2 ``normal`` for searchers with a start step, ``none`` for the others."""
    has_start = searcher_has_start_step(searcher)
    if block is None:
        block = {"method": "xtb", "params": dict(XTB_PARAMS)} if has_start else {"method": "none"}
    if isinstance(block, str):
        block = {"method": block}
    _check_keys(block, PRE_REFINEMENT_KEYS, where)
    method = block.get("method", "xtb" if has_start else "none")
    if method not in PRE_REFINEMENT_METHODS:
        raise ValueError(f"{where}.method must be one of {PRE_REFINEMENT_METHODS}, got {method!r}")
    params = dict(block.get("params") or {})
    if method == "xtb":
        unknown = set(params) - set(XTB_PARAMS)
        if unknown:
            raise ValueError(f"{where}.params: unknown keys {sorted(unknown)} for xtb; allowed: {sorted(XTB_PARAMS)}")
        params = {**XTB_PARAMS, **params}
        if params["gfn"] not in XTB_METHODS:
            raise ValueError(f"{where}.params.gfn must be one of {sorted(XTB_METHODS)}, got {params['gfn']!r}")
        if params["level"] not in XTB_LEVELS:
            raise ValueError(f"{where}.params.level must be one of {XTB_LEVELS}, got {params['level']!r}")
    elif method == "none" and params:
        raise ValueError(f"{where}: method none takes no params")
    if method != "none" and not has_start:
        raise ValueError(f"{where}: searcher {searcher!r} has no start geometry to pre-optimise "
                         f"(only {sorted(s for s in SEARCHERS if searcher_has_start_step(s))} have one); "
                         f"set `pre_refinement: none`")
    out = {"method": method, "params": params}
    for key in EMBEDDING_KEYS:
        if key in block and block[key] is not None:
            out[key] = block[key]
    return out


def _conformer_search(block, where: str) -> dict:
    _check_keys(block, SEARCH_KEYS, where)
    for key in ("name", "searcher"):
        if not block.get(key):
            raise ValueError(f"{where}.{key} is required")
    if block["searcher"] not in SEARCHERS:
        raise ValueError(f"{where}.searcher must be one of {sorted(SEARCHERS)}, got {block['searcher']!r}")
    return {"name": str(block["name"]), "searcher": block["searcher"], "params": dict(block.get("params") or {}),
            "slurm": dict(block.get("slurm") or {})}


def _post_optimization(block, where: str, present: bool) -> dict | None:
    if not present:
        block = {"refiner": "uma"}
    if block is None or block in ("none", False):
        return None
    _check_keys(block, POST_OPT_KEYS, where)
    refiner = block.get("refiner", "uma")
    if refiner not in REFINERS:
        raise ValueError(f"{where}.refiner must be one of {sorted(REFINERS)}, got {refiner!r}")
    return {"refiner": refiner, "params": dict(block.get("params") or {}), "slurm": dict(block.get("slurm") or {})}


def _pruning(block, where: str) -> dict:
    block = {**DEFAULTS["pruning"], **(block or {})}
    _check_keys(block, PRUNING_KEYS, where)
    ewin = block["energy_window_kcal_mol"]
    if ewin is not None:
        if isinstance(ewin, bool) or not isinstance(ewin, (int, float)) or ewin <= 0:
            raise ValueError(f"{where}.energy_window_kcal_mol must be a positive number or null, got {ewin!r}")
        block["energy_window_kcal_mol"] = float(ewin)
    return block


# --------------------------------------------------------------------------- bceg's view of the stages
def start_params(pre: dict) -> dict:
    """The bceg ``start_*`` searcher parameters that the pre_refinement block stands for."""
    out: dict[str, Any] = {}
    if pre["method"] == "none":
        out["start_preopt"] = "none"
    elif pre["method"] == "xtb":
        out["start_preopt"] = pre["params"]["gfn"]
        out["start_preopt_level"] = pre["params"]["level"]
    else:
        out["start_preopt"] = "uma"
        out["start_preopt_params"] = dict(pre["params"])
    for key, bceg_key in EMBEDDING_KEYS.items():
        if key in pre:
            out[bceg_key] = pre[key]
    return out


def to_bceg_raw(raw: dict) -> dict:
    """bceg's config dict (what ``bceg.config.read_raw`` produces) for the stages 1 to 4 of a library:
    one transformation, one method, ``replicates: 1``, bceg's ``slurm`` from ``slurm.ensembles`` with
    the library's ``setup`` unless it has its own."""
    t = raw["transformation"]
    spec = {k: t[k] for k in ("script", "params", "charge", "multiplicity") if t.get(k) is not None}
    search = raw["conformer_search"]
    params = dict(search["params"])
    if searcher_has_start_step(search["searcher"]):
        params.update(start_params(raw["pre_refinement"]))
    slurm = {"setup": list(raw["slurm"]["setup"]), **copy.deepcopy(raw["slurm"]["ensembles"])}
    return {
        "inputs": raw["smiles"],
        "results_dir": raw["ensembles_dir"],
        "replicates": 1,
        "transformations": {t["name"]: spec},
        "methods": {search["name"]: {"searcher": search["searcher"], "params": params, "slurm": dict(search["slurm"])}},
        "post_optimization": copy.deepcopy(raw["post_optimization"]),
        "slurm": slurm,
    }


# --------------------------------------------------------------------------- reading
def read_raw(path: str | Path) -> dict:
    """Read a library config: check the keys, make every path absolute, fill the defaults, normalise the
    stage blocks, and build bceg's config once so that a wrong searcher or refiner parameter fails here."""
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text()) or {}
    _check_keys(raw, TOP_KEYS, str(path))
    placeholders = find_placeholders(raw)
    if placeholders:
        raise ValueError(f"{path}: replace the {PLACEHOLDER} placeholder(s) at: " + ", ".join(placeholders)
                         + " (or delete the key to use the cluster default)")
    for key in ("library_dir", "smiles", "conformer_search", "dft"):
        if key not in raw:
            raise ValueError(f"{path}: '{key}' is required")

    def rel(p):
        return None if p is None else str((path.parent / p).resolve())

    out: dict[str, Any] = {"library_dir": rel(raw["library_dir"]), "smiles": rel(raw["smiles"])}
    out["ensembles_dir"] = rel(raw["ensembles_dir"]) if raw.get("ensembles_dir") else str(Path(out["library_dir"]) / "ensembles")
    run_bceg = raw.get("run_bceg", DEFAULTS["run_bceg"])
    if not isinstance(run_bceg, bool):
        raise ValueError(f"{path}: run_bceg must be true or false, got {run_bceg!r}")
    out["run_bceg"] = run_bceg

    out["conformer_search"] = _conformer_search(raw["conformer_search"], f"{path}: conformer_search")
    out["transformation"] = _transformation(raw.get("transformation"), f"{path}: transformation", path.parent)
    out["pre_refinement"] = _pre_refinement(raw.get("pre_refinement"), f"{path}: pre_refinement",
                                            out["conformer_search"]["searcher"])
    out["post_optimization"] = _post_optimization(raw.get("post_optimization"), f"{path}: post_optimization",
                                                  "post_optimization" in raw)
    out["pruning"] = _pruning(raw.get("pruning"), f"{path}: pruning")

    dft = dict(raw["dft"] or {})
    _check_keys(dft, DFT_KEYS, f"{path}: dft")
    if "level_of_theory" not in dft:
        raise ValueError(f"{path}: dft.level_of_theory is required")
    dft["calc_param_template"] = rel(dft.get("calc_param_template")) or str(DEFAULT_TEMPLATE)
    out["dft"] = dft

    slurm = copy.deepcopy(DEFAULTS["slurm"])
    for key, value in (raw.get("slurm") or {}).items():
        if key in ("prepare", "orca"):
            slurm[key] = {**slurm[key], **(value or {})}
        else:
            slurm[key] = value if value is not None else slurm.get(key)
    _check_keys(slurm, SLURM_KEYS, f"{path}: slurm")
    for prof in ("prepare", "orca"):
        _check_keys(slurm[prof], PROFILE_KEYS, f"{path}: slurm.{prof}")
    slurm["setup"] = list(slurm.get("setup") or [])
    slurm["ensembles"] = dict(slurm.get("ensembles") or {})
    _check_keys(slurm["ensembles"], BCEG_SLURM_KEYS, f"{path}: slurm.ensembles")
    out["slurm"] = slurm

    orca = raw.get("orca", DEFAULTS["orca"])
    if isinstance(orca, bool):
        orca = "on" if orca else "off"
    if orca not in ("on", "off"):
        raise ValueError(f"{path}: orca must be on or off, got {orca!r}")
    out["orca"] = orca
    out["condense"] = bool(raw.get("condense", DEFAULTS["condense"]))
    if "_pipeline" in raw:
        out["_pipeline"] = raw["_pipeline"]

    # bceg validates the searcher / refiner parameters and the transformation script
    try:
        BcegConfig.from_raw(to_bceg_raw(out), path)
    except (ValueError, TypeError, FileNotFoundError) as exc:
        raise ValueError(f"{path}: {exc}") from exc
    return out


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
    _bcfg: BcegConfig | None = field(default=None, repr=False, compare=False)

    @classmethod
    def load(cls, path: str | Path) -> Config:
        return cls(Path(path).resolve(), read_raw(path))

    # ------------------------------------------------------------------ paths and switches
    @property
    def library_dir(self) -> Path:
        return Path(self.raw["library_dir"])

    @property
    def smiles_path(self) -> Path:
        return Path(self.raw["smiles"])

    @property
    def ensembles_dir(self) -> Path:
        return Path(self.raw["ensembles_dir"])

    @property
    def run_bceg(self) -> bool:
        return bool(self.raw["run_bceg"])

    @property
    def orca_on(self) -> bool:
        return self.raw["orca"] == "on"

    @property
    def condense(self) -> bool:
        return self.raw["condense"]

    @property
    def slurm(self) -> dict:
        return self.raw["slurm"]

    # ------------------------------------------------------------------ the stages
    @property
    def transformation(self) -> dict:
        return self.raw["transformation"]

    @property
    def pre_refinement(self) -> dict:
        return self.raw["pre_refinement"]

    @property
    def search(self) -> dict:
        return self.raw["conformer_search"]

    @property
    def post_optimization(self) -> dict | None:
        return self.raw["post_optimization"]

    @property
    def pruning(self) -> dict:
        return self.raw["pruning"]

    @property
    def ewin(self) -> float | None:
        return self.pruning["energy_window_kcal_mol"]

    @property
    def dft(self) -> dict:
        return self.raw["dft"]

    @property
    def transformation_name(self) -> str:
        return self.transformation["name"]

    @property
    def method_name(self) -> str:
        return self.search["name"]

    @property
    def has_preopt(self) -> bool:
        return self.pre_refinement["method"] != "none"

    @property
    def lot(self) -> str | None:
        """Level label of the post-optimisation (the file refined is ``raw_1_<lot>.xyz``), None without one."""
        refiner = self.bceg_config().refiner
        return refiner.lot if refiner is not None else None

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

    # ------------------------------------------------------------------ bceg
    def bceg_raw(self) -> dict:
        return to_bceg_raw(self.raw)

    def bceg_config(self) -> BcegConfig:
        if self._bcfg is None:
            self._bcfg = BcegConfig.from_raw(self.bceg_raw(), self.path)
        return self._bcfg

    # ------------------------------------------------------------------ provenance
    def template_sha256(self) -> str:
        return hashlib.sha256(self.template_path.read_bytes()).hexdigest()

    def compared(self) -> dict:
        """The six stage blocks as bceg will run them (defaults filled in), plus the template's sha256."""
        bcfg = self.bceg_config()
        transformation = bcfg.transformations[self.transformation_name].identity()
        searcher = bcfg.methods[self.method_name].searcher
        start_keys = set(start_params(self.pre_refinement)) | {"start_preopt", "start_preopt_level", "start_preopt_params"}
        search = {"name": self.method_name, "searcher": searcher.name,
                  "params": {k: v for k, v in searcher.compared_params().items() if k not in start_keys}}
        post = None
        if bcfg.refiner is not None:
            post = {"refiner": bcfg.refiner.name, "lot": bcfg.refiner.lot, "params": bcfg.refiner.compared_params()}
        dft = dict(self.dft)
        dft["calc_param_template"] = {"path": str(self.template_path), "sha256": self.template_sha256()}
        return {"transformation": transformation, "pre_refinement": copy.deepcopy(self.pre_refinement),
                "conformer_search": search, "post_optimization": post, "pruning": dict(self.pruning), "dft": dft,
                "replicates": 1}

    def recorded(self) -> dict:
        return {"orca": self.raw["orca"], "run_bceg": self.run_bceg,
                "paths": {"smiles": str(self.smiles_path), "ensembles_dir": str(self.ensembles_dir)},
                "slurm": self.slurm, "condense": self.condense}

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
