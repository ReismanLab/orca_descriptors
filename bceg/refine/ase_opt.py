"""The ASE optimisation loop shared by the calculator-based refiners (UMA, AIMNet2).

The conformers are independent, so on CPUs they are shared out over a pool of worker processes
(the refiner's ``workers``), each loading the model once and running torch on ``threads`` threads.
One multi-threaded forward pass on a molecule of a few dozen atoms leaves most cores idle; one
single-threaded process per core does not. A single worker (the default on a GPU, or one conformer)
optimises in-process, as before.

The workers are started with the ``spawn`` method: a fork of a process that has initialised torch's
thread pools can deadlock. Each worker therefore rebuilds the refiner from its parameters (with
``threads`` set to its share) and asks it for a calculator; the per-conformer ``opt_<i>.log`` files
are written by the workers.
"""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from bceg.external import allocated_cpus, resolve_threads
from bceg.refine.base import RefineResult
from bceg.structure import Structure
from bceg.xyz import HARTREE_TO_EV, Conformer

START_METHOD = "spawn"  # the test-suite forks so that its stand-in modules reach the workers
MAX_DEFAULT_WORKERS_OUTSIDE_SLURM = 8  # every worker holds a copy of the model: no 64-process pool on a login node
_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def package_version(pkg: str) -> str:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "unknown"


def resolve_workers(workers: int | None, device: str, n_conformers: int) -> int:
    """``None``/0: one worker per CPU of the allocation on ``cpu`` (at most 8 outside SLURM), 1 on a
    GPU; never more workers than conformers."""
    if not workers:
        if device != "cpu":
            workers = 1
        elif "SLURM_JOB_ID" in os.environ:
            workers = allocated_cpus()
        else:
            workers = min(allocated_cpus(), MAX_DEFAULT_WORKERS_OUTSIDE_SLURM)
    return max(1, min(int(workers), n_conformers))


def threads_per_worker(threads: int | None, workers: int) -> int:
    """``None``/0: the allocation's CPUs shared equally between the workers (all of them for one)."""
    if threads:
        return int(threads)
    return resolve_threads(None) if workers == 1 else max(1, allocated_cpus() // workers)


# ---------------------------------------------------------------------------------------- constraints
def ase_constraints(fixed=(), holds: dict | None = None) -> list:
    """ASE constraints for atoms ``fixed`` in place and for bceg.impose.holds: plane atoms move only within
    their plane (FixedPlane), the atoms of a linear angle only along its line (FixedLine), the other imposed
    values are kept by FixInternals. An atom held in two ways that do not combine (two planes; a plane and
    a line across it; a line or plane and an internal coordinate) is fixed outright."""
    from ase.constraints import FixAtoms, FixedLine, FixedPlane, FixInternals

    fixed = set(fixed)
    out = []
    if holds:
        lines = {a: line for line in holds["lines"] for a in line["atoms"]}
        planes: dict[int, list] = {}
        for plane in holds["planes"]:
            for a in plane["atoms"]:
                planes.setdefault(a, []).append(plane)
        internal = {a for kind in ("bonds", "angles", "dihedrals") for t in holds[kind] for a in t["atoms"]}
        for a in set(lines) | set(planes):
            across = a in lines and a in planes and abs(np.dot(lines[a]["direction"], planes[a][0]["normal"])) > 1e-6
            if len(planes.get(a, [])) > 1 or across or a in internal:
                fixed.add(a)
        for line in holds["lines"]:
            free = [a for a in line["atoms"] if a not in fixed]
            if free:
                out.append(FixedLine(free, line["direction"]))
        for plane in holds["planes"]:
            free = [a for a in plane["atoms"] if a not in fixed and a not in lines]  # a line inside the plane wins
            if free:
                out.append(FixedPlane(free, plane["normal"]))
        if internal:
            out.append(FixInternals(bonds=[[t["value"], t["atoms"]] for t in holds["bonds"]],
                                    angles_deg=[[t["value"], t["atoms"]] for t in holds["angles"]],
                                    dihedrals_deg=[[t["value"], t["atoms"]] for t in holds["dihedrals"]]))
    if fixed:
        out.insert(0, FixAtoms(indices=sorted(fixed)))  # get_forces() then zeroes them
    return out


# ---------------------------------------------------------------------------------------- one conformer
def _optimize(calc, structure: Structure, i: int, coords: np.ndarray, workdir: Path, *,
              optimizer: str, fmax: float, max_steps: int, fixed: list[int] = (),
              holds: dict | None = None) -> dict[str, Any]:
    """Optimise one conformer with ``calc``, the atoms ``fixed`` held in place and the ``holds`` of an
    imposed geometry kept (see :func:`ase_constraints`); returns the geometry, the energy in eV, the
    residual force (with the constraints applied) and the number of steps. ``converged`` is decided by
    the caller."""
    from ase import Atoms, optimize

    atoms = Atoms(symbols=structure.elements, positions=coords)
    atoms.info.update(charge=structure.charge, spin=structure.multiplicity)
    if fixed or holds:
        atoms.set_constraint(ase_constraints(fixed, holds))
    atoms.calc = calc
    opt = getattr(optimize, optimizer)(atoms, logfile=str(workdir / f"opt_{i}.log"))
    opt.run(fmax=fmax, steps=max_steps)
    residual = float(np.sqrt((atoms.get_forces() ** 2).sum(axis=1)).max())
    return {"index": i, "coords": atoms.get_positions(), "energy_ev": float(atoms.get_potential_energy()),
            "residual": residual, "steps": opt.nsteps}


# ---------------------------------------------------------------------------------------- the worker pool
_WORKER: dict[str, Any] = {}  # calculator and settings of this worker process, filled by _init_worker


def _init_worker(refiner_cls, params: dict, structure: Structure, workdir: Path, settings: dict, load_lock) -> None:
    """Pin the numerical libraries to the worker's share of the CPUs, then build the refiner and its
    calculator (the model is loaded once per worker; ``params['threads']`` is already the share).

    The workers load their model one at a time: loading the UMA checkpoint briefly takes ~5 GB in the
    process (~2 GB once loaded), so eight simultaneous loads would need ~40 GB at the pool's start."""
    for var in _THREAD_VARS:
        os.environ[var] = str(params["threads"])
    refiner = refiner_cls(**params)
    with load_lock:
        calc = refiner.make_calculator(structure)
    _WORKER.update(calc=calc, structure=structure, workdir=workdir, settings=settings)


def _optimize_in_worker(i: int, coords: np.ndarray) -> dict[str, Any]:
    w = _WORKER
    return _optimize(w["calc"], w["structure"], i, coords, w["workdir"], **w["settings"])


@contextmanager
def _thread_env(threads: int):
    """Single-threaded numerical libraries in the environment the workers are spawned with (read when
    numpy / torch are imported, before the initializer runs); the parent's values are restored after."""
    saved = {var: os.environ.get(var) for var in _THREAD_VARS}
    os.environ.update({var: str(threads) for var in _THREAD_VARS})
    try:
        yield
    finally:
        for var, value in saved.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value


def _optimize_in_pool(refiner, structure: Structure, conformers: list[Conformer], workdir: Path,
                      workers: int, threads: int, settings: dict) -> list[dict[str, Any]]:
    """Futures are submitted one by one and never cancelled: on Python 3.11, ``pool.map`` cancels the
    pending futures when a worker dies, the pool's manager thread then fails on them
    (``InvalidStateError``) before terminating the surviving workers, and the job hangs at exit
    waiting for them. On any failure the workers still alive are killed here, before the shutdown."""
    params = {**refiner.params, "threads": threads}
    context = multiprocessing.get_context(START_METHOD)
    before = set(multiprocessing.active_children())
    with _thread_env(threads):
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_init_worker,
                                   initargs=(type(refiner), params, structure, workdir, settings, context.Lock()))
    try:
        futures = [pool.submit(_optimize_in_worker, i, c.coords) for i, c in enumerate(conformers)]
        out = [f.result() for f in futures]
    except BrokenProcessPool as exc:
        _kill_workers(before)
        raise RuntimeError(
            f"{refiner.name}: a post-optimisation worker process died ({workers} workers x {threads} threads); "
            f"every worker holds its own copy of the model, so this is usually the job's memory limit. Raise "
            f"the step's `mem` or lower `workers` in post_optimization.params.") from exc
    except BaseException:
        _kill_workers(before)
        raise
    finally:
        pool.shutdown(wait=True)
    return out


def _kill_workers(before: set) -> None:
    """SIGKILL the pool's processes still alive (children started since ``before``) and reap them."""
    for p in set(multiprocessing.active_children()) - before:
        p.kill()
        p.join(timeout=30)


# ---------------------------------------------------------------------------------------- the loop
def optimize_conformers(refiner, structure: Structure, conformers: list[Conformer], workdir: Path, *,
                        versions: dict[str, Any], holds: dict | None = None) -> RefineResult:
    """Optimise every conformer with the refiner's ASE calculator; energies come back in Hartree.

    ``refiner.params`` supplies ``optimizer`` (a class of ``ase.optimize``: LBFGS, BFGS, FIRE), ``fmax``,
    ``max_steps``, ``keep_unconverged``, ``device``, ``workers`` and ``threads``. One ``opt_<i>.log`` per
    conformer is written to ``workdir``. A conformer whose residual force exceeds ``fmax`` after
    ``max_steps`` is kept with ``converged=False`` or dropped, after ``keep_unconverged``. The
    ``details`` record ``workers`` and ``threads`` (per worker) as resolved.
    """
    p = refiner.params
    fixed = refiner.fixed_atoms(structure)
    settings = {"optimizer": p["optimizer"], "fmax": p["fmax"], "max_steps": p["max_steps"], "fixed": fixed,
                "holds": holds}
    workers = resolve_workers(p["workers"], p["device"], len(conformers))
    threads = threads_per_worker(p["threads"], workers)
    workdir.mkdir(parents=True, exist_ok=True)   # e.g. preopt_<attempt> of start_preopt: uma
    if workers == 1:
        calc = refiner.make_calculator(structure)
        raw = [_optimize(calc, structure, i, c.coords, workdir, **settings) for i, c in enumerate(conformers)]
    else:
        raw = _optimize_in_pool(refiner, structure, conformers, workdir, workers, threads, settings)

    out, n_unconverged = [], 0
    for r in raw:
        converged = r["residual"] <= p["fmax"]
        n_unconverged += not converged
        if not converged and not p["keep_unconverged"]:
            continue
        out.append(Conformer(r["coords"], r["energy_ev"] / HARTREE_TO_EV,
                             {"converged": converged, "steps": r["steps"], "fmax": round(r["residual"], 5),
                              "source_index": r["index"]}))
    return RefineResult(out, versions=versions,
                        details={"n_input": len(conformers), "n_unconverged": n_unconverged,
                                 "workers": workers, "threads": threads} | ({"fixed_atoms": fixed} if fixed else {}))
