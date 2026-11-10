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

With a ``checkpoint`` every optimised conformer is appended to a JSON-lines file as soon as it comes
back, so a job killed by its time limit (or anything else) loses only the conformers in flight: the
next run reads the file and optimises only the others. Its first line is a key (the caller's hash of
the input ensemble and the parameters); a file with another key is from other inputs and is ignored.

``stall_steps`` stops an optimisation whose lowest energy went down by less than ``stall_energy`` (eV)
over the last ``stall_steps`` steps: one oscillating between two geometries without reaching ``fmax``
would otherwise run to ``max_steps``. The conformer is then unconverged (kept or dropped after
``keep_unconverged``) and flagged ``stalled=True``. On 2359 converged uma-s-1p2 optimisations (SmI2
complexes, fmax 0.01) the lowest energy always went down by more than 7.8 meV in any 100 steps; the one
oscillating went down by less than 1 meV, so ``stall_steps: 100, stall_energy: 0.001`` stopped it at step
437 and no converged one.
"""

from __future__ import annotations

import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

import numpy as np

from bceg.external import allocated_cpus, resolve_threads
from bceg.refine.base import RefineResult
from bceg.refine.checkpoint import CheckpointWriter as _CheckpointWriter  # noqa: F401  (re-exported)
from bceg.refine.checkpoint import load_checkpoint
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
WALL_K = 10.0  # eV/A^2 of the flat-bottom walls that keep an imposed angle's bonds to a metal from breaking


def ase_constraints(fixed=(), holds: dict | None = None) -> list:
    """ASE constraints for atoms ``fixed`` in place and for bceg.impose.holds: plane atoms move only within
    their plane (FixedPlane), the atoms of a linear angle only along its line (FixedLine), the other imposed
    values are kept by FixInternals, and a bond in ``walls`` is pulled back only beyond its ``max`` (Hookean). An atom held in two ways that do not combine (two planes; a plane and
    a line across it; a line or plane and an internal coordinate) is fixed outright."""
    from ase.constraints import FixAtoms, FixedLine, FixedPlane, FixInternals, Hookean

    fixed = set(fixed)
    out = []
    if holds:
        # flat-bottom walls first (Hookean adds its force, the line / plane constraints then project it)
        out += [Hookean(w["atoms"][0], w["atoms"][1], k=WALL_K, rt=w["max"]) for w in holds.get("walls") or []]
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
              holds: dict | None = None, stall_steps: int | None = None,
              stall_energy: float = 0.001) -> dict[str, Any]:
    """Optimise one conformer with ``calc``, the atoms ``fixed`` held in place and the ``holds`` of an
    imposed geometry kept (see :func:`ase_constraints`); returns the geometry, the energy in eV, the
    residual force (with the constraints applied), the number of steps and whether the stall stop ended it
    (module docstring). ``converged`` is decided by the caller."""
    from ase import Atoms, optimize

    atoms = Atoms(symbols=structure.elements, positions=coords)
    atoms.info.update(charge=structure.charge, spin=structure.multiplicity)
    if fixed or holds:
        atoms.set_constraint(ase_constraints(fixed, holds))
    atoms.calc = calc
    opt = getattr(optimize, optimizer)(atoms, logfile=str(workdir / f"opt_{i}.log"))
    stalled = False
    if stall_steps:
        best = []   # the lowest energy so far, after each step
        for converged in opt.irun(fmax=fmax, steps=max_steps):
            e = float(atoms.get_potential_energy())
            best.append(min(e, best[-1]) if best else e)
            if not converged and len(best) > stall_steps and best[-1 - stall_steps] - best[-1] < stall_energy:
                stalled = True
                break
    else:
        opt.run(fmax=fmax, steps=max_steps)
    residual = float(np.sqrt((atoms.get_forces() ** 2).sum(axis=1)).max())
    return {"index": i, "coords": atoms.get_positions(), "energy_ev": float(atoms.get_potential_energy()),
            "residual": residual, "steps": opt.nsteps, "stalled": stalled}


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


def _optimize_in_pool(refiner, structure: Structure, todo: list[tuple[int, np.ndarray]], workdir: Path,
                      workers: int, threads: int, settings: dict,
                      on_result: Callable[[dict], None]) -> list[dict[str, Any]]:
    """Futures are submitted one by one and never cancelled: on Python 3.11, ``pool.map`` cancels the
    pending futures when a worker dies, the pool's manager thread then fails on them
    (``InvalidStateError``) before terminating the surviving workers, and the job hangs at exit
    waiting for them. On any failure the workers still alive are killed here, before the shutdown.
    ``on_result`` sees every result as it comes back (in completion order)."""
    params = {**refiner.params, "threads": threads}
    context = multiprocessing.get_context(START_METHOD)
    before = set(multiprocessing.active_children())
    with _thread_env(threads):
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_init_worker,
                                   initargs=(type(refiner), params, structure, workdir, settings, context.Lock()))
    try:
        futures = [pool.submit(_optimize_in_worker, i, coords) for i, coords in todo]
        out = []
        for f in as_completed(futures):
            out.append(f.result())
            on_result(out[-1])
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
                        versions: dict[str, Any], holds: dict | None = None,
                        checkpoint: tuple[Path, str] | None = None) -> RefineResult:
    """Optimise every conformer with the refiner's ASE calculator; energies come back in Hartree.

    ``refiner.params`` supplies ``optimizer`` (a class of ``ase.optimize``: LBFGS, BFGS, FIRE), ``fmax``,
    ``max_steps``, ``keep_unconverged``, ``device``, ``workers`` and ``threads``. One ``opt_<i>.log`` per
    conformer is written to ``workdir``. A conformer whose residual force exceeds ``fmax`` after
    ``max_steps`` is kept with ``converged=False`` or dropped, after ``keep_unconverged``. The
    ``details`` record ``workers`` and ``threads`` (per worker) as resolved.

    ``checkpoint``: ``(path, key)`` of the file the results are appended to as they come (module
    docstring); the conformers already in it are not optimised again (``details["n_resumed"]``).
    """
    p = refiner.params
    fixed = refiner.fixed_atoms(structure)
    settings = {"optimizer": p["optimizer"], "fmax": p["fmax"], "max_steps": p["max_steps"], "fixed": fixed,
                "holds": holds, "stall_steps": p.get("stall_steps"), "stall_energy": p.get("stall_energy", 0.001)}
    done = load_checkpoint(*checkpoint) if checkpoint else {}
    todo = [(i, c.coords) for i, c in enumerate(conformers) if i not in done]
    on_result = _CheckpointWriter(checkpoint[0], checkpoint[1], resumed=bool(done)) if checkpoint else (lambda r: None)
    workers = resolve_workers(p["workers"], p["device"], max(1, len(todo)))
    threads = threads_per_worker(p["threads"], workers)
    workdir.mkdir(parents=True, exist_ok=True)   # e.g. preopt_<attempt> of start_preopt: uma
    try:
        if not todo:
            new = []
        elif workers == 1:
            calc = refiner.make_calculator(structure)
            new = []
            for i, coords in todo:
                new.append(_optimize(calc, structure, i, coords, workdir, **settings))
                on_result(new[-1])
        else:
            new = _optimize_in_pool(refiner, structure, todo, workdir, workers, threads, settings, on_result)
    finally:
        if checkpoint:
            on_result.close()
    raw = sorted([*done.values(), *new], key=lambda r: r["index"])

    out, n_unconverged, n_stalled = [], 0, 0
    for r in raw:
        converged = r["residual"] <= p["fmax"]
        n_unconverged += not converged
        n_stalled += bool(r.get("stalled"))
        if not converged and not p["keep_unconverged"]:
            continue
        info = {"converged": converged, "steps": r["steps"], "fmax": round(r["residual"], 5), "source_index": r["index"]}
        out.append(Conformer(r["coords"], r["energy_ev"] / HARTREE_TO_EV,
                             info | ({"stalled": True} if r.get("stalled") else {})))
    return RefineResult(out, versions=versions,
                        details={"n_input": len(conformers), "n_unconverged": n_unconverged,
                                 "workers": workers, "threads": threads} | ({"fixed_atoms": fixed} if fixed else {})
                        | ({"n_resumed": len(done)} if done else {})
                        | ({"n_stalled": n_stalled} if p.get("stall_steps") else {}))
