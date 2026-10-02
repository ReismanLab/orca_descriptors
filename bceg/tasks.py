"""Task expansion and execution.

A task is one replicate of one method for one (transformation, molecule):
a ``search`` (writes raw_<i>.xyz), a ``refine`` (writes raw_<i>_<lot>.xyz), or a
``chain`` (search then refine in the same job; the default on the cluster).
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import time
import traceback
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from bceg.config import Config
from bceg.external import available_cpus, cpu_model, gpu_names, workdir
from bceg.inputs import InputMolecule, load_inputs, update_index
from bceg.layout import MethodDir, method_dir
from bceg.provenance import ParamsMismatch, check_or_write, make_params, read_record, write_record
from bceg.search import SEED_RULE
from bceg.structure import Structure
from bceg.xyz import read_xyz, write_xyz

DONE, FAILED, NOT_APPLICABLE, RUNNING = "done", "failed", "not_applicable", "running"
PRE_REFINEMENT, SEARCH = "pre_refinement", "search"   # phases of a search task (run_<i>.json "phase")


@dataclass(frozen=True)
class Task:
    kind: str  # "search" | "refine" | "chain"
    transformation: str
    inchikey: str
    input_smiles: str
    method: str
    replicate: int
    resource: str  # "cpu" | "gpu"

    @property
    def id(self) -> str:
        return f"{self.kind}:{self.transformation}/{self.inchikey}/{self.method}/{self.replicate}"

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, line: str) -> Task:
        return cls(**json.loads(line))


# ----------------------------------------------------------------------------
# hyper-parameter logs
# ----------------------------------------------------------------------------
def search_params(cfg: Config, method: str, transformation: str, structure: Structure) -> dict:
    searcher = cfg.methods[method].searcher
    compared = {
        "searcher": searcher.name,
        "params": searcher.compared_params(),
        "seed_rule": SEED_RULE,
        "transformation": cfg.transformations[transformation].identity(),
        "structure": structure.identity(),
    }
    return make_params(compared, {"params": searcher.recorded_params(), "replicates": cfg.replicates})


def refine_params(cfg: Config, search_sha256: str) -> dict:
    r = cfg.refiner
    compared = {"refiner": r.name, "lot": r.lot, "params": r.compared_params(), "search_params_sha256": search_sha256}
    return make_params(compared, {"params": r.recorded_params()})


def _hint(method: str, what: str = "method") -> str:
    return f"Give the {what} a new name in the config (e.g. '{method}_v2') or restore the logged parameters."


# ----------------------------------------------------------------------------
# expansion
# ----------------------------------------------------------------------------
def _is_complete(record: dict | None, xyz: Path) -> bool:
    if record is None:
        return False
    return record.get("status") == NOT_APPLICABLE or (record.get("status") == DONE and xyz.exists())


def expand(cfg: Config, molecules: list[InputMolecule] | None = None, write_params: bool = True,
           pending_only: bool = True, chain: bool | None = None) -> tuple[list[Task], list[str]]:
    """All tasks implied by the config. Returns (tasks, params-mismatch messages).

    Methods whose existing params log does not match are excluded and reported. With
    ``write_params`` missing logs are created; otherwise the check is read-only.
    ``chain`` (default: ``cfg.chained``) merges search + refine of a replicate into one task.
    """
    chain = cfg.chained if chain is None else chain
    if chain and cfg.refiner is not None:
        split, problems = expand(cfg, molecules, write_params, pending_only, chain=False)
        return _merge_chains(split, cfg), problems
    if molecules is None and cfg.inputs is None:
        raise ValueError("No input SMILES: set 'inputs' in the config or pass --smiles")
    molecules = molecules if molecules is not None else load_inputs(cfg.inputs)
    tasks, problems = [], []
    for tname, transformation in cfg.transformations.items():
        for mol in molecules:
            structure = apply_transform(transformation, mol)
            if structure is None:
                continue
            for mname, method in cfg.methods.items():
                if method.searcher.applicable(structure):
                    continue  # e.g. LoQI for a Ni complex: no folder, no task
                md = method_dir(cfg.results_dir, tname, mol.inchikey, mname)
                sp = search_params(cfg, mname, tname, structure)
                rp = refine_params(cfg, sp["sha256"]) if cfg.refiner else None
                try:
                    check_or_write(md.params(), sp, _hint(mname), write=write_params)
                    if rp:
                        check_or_write(md.params(cfg.refiner.lot), rp, _hint(mname), write=write_params)
                except ParamsMismatch as exc:
                    problems.append(str(exc))
                    continue
                for rep in range(1, cfg.replicates + 1):
                    s = Task("search", tname, mol.inchikey, mol.input_smiles, mname, rep, method.searcher.resource_class())
                    rec = read_record(md.run_record(rep))
                    search_done = _is_complete(rec, md.raw(rep))
                    if not (pending_only and search_done):
                        tasks.append(s)
                    if rp is None or (rec and rec.get("status") == NOT_APPLICABLE):
                        continue
                    lot = cfg.refiner.lot
                    r = Task("refine", tname, mol.inchikey, mol.input_smiles, mname, rep, cfg.refiner.resource_class())
                    if not (pending_only and _is_complete(read_record(md.run_record(rep, lot)), md.refined(rep, lot))):
                        tasks.append(r)
    return tasks, problems


def _merge_chains(split: list[Task], cfg: Config) -> list[Task]:
    """One chain task per replicate with any pending step (GPU if either step needs it)."""
    merged: dict[tuple, Task] = {}
    for t in split:
        key = (t.transformation, t.inchikey, t.method, t.replicate)
        resource = t.resource
        if key in merged:
            resource = "gpu" if "gpu" in (resource, merged[key].resource) else "cpu"
        elif t.kind == "refine":  # search already complete; the chain only needs the refiner
            resource = cfg.refiner.resource_class()
        else:
            resource = "gpu" if "gpu" in (t.resource, cfg.refiner.resource_class()) else "cpu"
        merged[key] = Task("chain", *key[:2], t.input_smiles, *key[2:], resource)
    return list(merged.values())


def apply_transform(transformation, mol: InputMolecule) -> Structure | None:
    structure = transformation.apply(mol.canonical_smiles)
    if structure is None:
        warnings.warn(f"Transformation {transformation.name} skipped {mol.input_smiles}")
    return structure


# ----------------------------------------------------------------------------
# execution
# ----------------------------------------------------------------------------
def _env_record() -> dict:
    """Where the task ran, on what hardware. Written before it starts."""
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "cpus": available_cpus(),
        "cpu_model": cpu_model(),
        "gpus": list(gpu_names()),
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _usage(record: dict, wall: float) -> dict:
    """Wall time and the allocation it occupied: core-hours, and GPU-hours if the node had GPUs.

    ``cpu_hours`` is wall time x the CPUs allocated to the job, not the CPUs the program
    kept busy: it is the cost of the run, not its parallel efficiency (use ``seff`` for that).
    """
    wall = round(wall, 2)  # the hours are derived from it, so the record agrees with itself
    usage = {"wall_time_s": wall, "cpu_hours": round(wall / 3600 * record["cpus"], 5)}
    if record["gpus"]:
        usage["gpu_hours"] = round(wall / 3600 * len(record["gpus"]), 5)
    return usage


def _search_usage(record: dict, t0: float, t_split: float | None, phase: str) -> dict:
    """Usage of a search task: the total, and the same figures per phase under ``phases``.

    ``pre_refinement`` is the time until the start geometry was accepted (embedding, force field, xtb
    or UMA pre-optimisation, every rejected attempt included); ``search`` is the searcher proper.
    ``t_split`` is when the start-geometry callback fired; without one (no start step, or the task
    died before a geometry was accepted) the whole time goes to ``phase``, the phase the task ended in.
    """
    t_end = time.time()
    total = _usage(record, t_end - t0)
    if t_split is None:
        phases = {phase: dict(total)}
    else:
        phases = {PRE_REFINEMENT: _usage(record, t_split - t0), SEARCH: _usage(record, t_end - t_split)}
    return total | {"phases": phases}


def _write_xyz_atomic(path: Path, elements, conformers) -> None:
    tmp = path.with_name(path.name + ".tmp")
    write_xyz(tmp, elements, conformers)
    tmp.replace(path)


FAILED_LOG_MAX_BYTES = 10_000_000


def _keep_failed_workdir(wd: Path, md: MethodDir, replicate: int) -> None:
    """Copy the small files of a failed run (incl. preopt_*/ subfolders) to logs/raw_<i>_failed/."""
    dest = md.logs / f"raw_{replicate}_failed"
    if dest.exists():
        shutil.rmtree(dest)
    for f in wd.rglob("*"):
        if f.is_file() and f.stat().st_size <= FAILED_LOG_MAX_BYTES:
            target = dest / f.relative_to(wd)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)


def _start_geometry_rmsd_to_replicates(structure: Structure, wd: Path, md: MethodDir, replicate: int) -> dict:
    """Heavy-atom RMSD of this replicate's start geometry to those of already finished replicates."""
    from bceg.geometry import heavy_atom_rmsd

    own = wd / "start_geometry.xyz"
    if not own.exists():
        return {}
    _, (mine,) = read_xyz(own)
    out = {}
    for other in sorted(md.logs.glob("raw_*_start_geometry.xyz")) if md.logs.exists() else []:
        j = int(other.name.split("_")[1])
        if j != replicate:
            out[str(j)] = round(heavy_atom_rmsd(structure.mol, mine.coords, read_xyz(other)[1][0].coords), 3)
    return out


def _copy_logs(files: list[Path], md: MethodDir, prefix: str) -> None:
    for f in files:
        if f.exists():
            md.logs.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, md.logs / f"{prefix}_{f.name}")


def run_task(cfg: Config, task: Task) -> str:
    """Run one task. Returns its final status ("done", "skipped", "failed", ...)."""
    transformation = cfg.transformations[task.transformation]
    mol = InputMolecule.from_smiles(task.input_smiles)
    if mol.inchikey != task.inchikey:
        raise RuntimeError(f"{task.id}: InChIKey changed ({mol.inchikey}); RDKit version drift?")
    structure = transformation.apply(mol.canonical_smiles)
    md = method_dir(cfg.results_dir, task.transformation, task.inchikey, task.method)
    sp = search_params(cfg, task.method, task.transformation, structure)
    check_or_write(md.params(), sp, _hint(task.method))
    if task.kind in ("search", "chain"):
        status = _run_search(cfg, task, structure, md, sp)
        if task.kind == "search" or cfg.refiner is None or status not in (DONE, "skipped"):
            return status
    rp = refine_params(cfg, sp["sha256"])
    check_or_write(md.params(cfg.refiner.lot), rp, _hint(task.method))
    return _run_refine(cfg, task, structure, md, sp)


def _base_record(task: Task, structure: Structure, params_sha: str) -> dict:
    return {
        "task": asdict(task),
        "input_smiles": task.input_smiles,
        "structure": structure.identity() | {"metadata": structure.metadata},
        "params_sha256": params_sha,
        **_env_record(),
    }


def _run_search(cfg: Config, task: Task, structure: Structure, md: MethodDir, sp: dict) -> str:
    rec_path, xyz = md.run_record(task.replicate), md.raw(task.replicate)
    if _is_complete(read_record(rec_path), xyz):
        return "skipped"
    searcher = cfg.methods[task.method].searcher
    record = _base_record(task, structure, sp["sha256"]) | {"seed": searcher.seed(task.replicate)}

    reason = searcher.applicable(structure)
    if reason:
        write_record(rec_path, record | {"status": NOT_APPLICABLE, "reason": reason})
        return NOT_APPLICABLE

    # phase: "pre_refinement" while the start geometry is embedded and pre-optimised (CREST, GOAT,
    # molassembler), "search" once the searcher proper runs. Lets a status reader tell the two apart.
    split = searcher.has_start_step()  # a start step: time it apart from the search (run record "phases")
    phase = {"phase": PRE_REFINEMENT if split else SEARCH}
    write_record(rec_path, record | {"status": RUNNING} | phase)
    t0 = time.time()
    t_split: list[float | None] = [None]

    def _start_geometry_done(info: dict) -> None:
        if split:
            t_split[0] = time.time()
        phase["phase"] = SEARCH
        write_record(rec_path, record | {"status": RUNNING, "phase": SEARCH, "start_geometry": info})

    searcher.on_start_geometry = _start_geometry_done
    try:
        with workdir(prefix=f"{task.method}_{task.replicate}_") as wd:
            try:
                result = searcher.run(structure, task.replicate, wd)
                _write_xyz_atomic(xyz, structure.elements, result.conformers)
            except Exception as exc:
                _keep_failed_workdir(wd, md, task.replicate)
                write_record(rec_path, record | _search_usage(record, t0, t_split[0], phase["phase"]) | phase | {
                    "status": FAILED, "error": repr(exc), "traceback": traceback.format_exc()[-5000:],
                    "failed_files": f"logs/raw_{task.replicate}_failed/"})
                return FAILED
            if isinstance(result.details.get("start_geometry"), dict):
                result.details["start_geometry"]["rmsd_to_replicates"] = _start_geometry_rmsd_to_replicates(
                    structure, wd, md, task.replicate)
            _copy_logs(result.log_files, md, f"raw_{task.replicate}")
    finally:
        searcher.on_start_geometry = None
    write_record(rec_path, record | _search_usage(record, t0, t_split[0], phase["phase"]) | phase | {
        "status": DONE, "n_conformers": len(result.conformers),
        "versions": result.versions, "details": result.details,
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    return DONE


def _run_refine(cfg: Config, task: Task, structure: Structure, md: MethodDir, sp: dict) -> str:
    lot = cfg.refiner.lot
    rec_path, xyz = md.run_record(task.replicate, lot), md.refined(task.replicate, lot)
    if _is_complete(read_record(rec_path), xyz):
        return "skipped"
    search_rec = read_record(md.run_record(task.replicate))
    if not search_rec or search_rec.get("status") != DONE or not md.raw(task.replicate).exists():
        return "waiting_for_search"
    if search_rec.get("params_sha256") != sp["sha256"]:
        raise ParamsMismatch(f"{md.raw(task.replicate)} was produced with different search parameters")

    elements, conformers = read_xyz(md.raw(task.replicate))
    if elements != structure.elements:
        raise RuntimeError(f"{md.raw(task.replicate)}: atom order does not match the structure template")
    record = _base_record(task, structure, refine_params(cfg, sp["sha256"])["sha256"]) | {"lot": lot}
    write_record(rec_path, record | {"status": RUNNING})
    t0 = time.time()
    with workdir(prefix=f"{task.method}_{task.replicate}_{lot}_") as wd:
        try:
            result = cfg.refiner.run(structure, conformers, wd)
            _write_xyz_atomic(xyz, structure.elements, result.conformers)
        except Exception as exc:
            write_record(rec_path, record | _usage(record, time.time() - t0) | {
                "status": FAILED, "error": repr(exc), "traceback": traceback.format_exc()[-5000:]})
            return FAILED
    write_record(rec_path, record | _usage(record, time.time() - t0) | {
        "status": DONE, "n_conformers": len(result.conformers),
        "versions": result.versions, "details": result.details,
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    return DONE


def run_local(cfg: Config) -> dict[str, int]:
    """Run every pending task sequentially (searches before refinements)."""
    molecules = load_inputs(cfg.inputs)
    update_index(cfg.results_dir, molecules)
    tasks, problems = expand(cfg, molecules)
    for p in problems:
        warnings.warn(p)
    counts: dict[str, int] = {}
    for task in sorted(tasks, key=lambda t: t.kind == "refine"):
        status = run_task(cfg, task)
        counts[status] = counts.get(status, 0) + 1
        print(f"[{status:>14}] {task.id}", flush=True)
    return counts
