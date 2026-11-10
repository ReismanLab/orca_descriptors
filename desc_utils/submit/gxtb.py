"""Stage 6 with g-xTB instead of ORCA: one ``xtb --gxtb`` run per step, per conformer.

Selected with ``dft.engine: gxtb`` in the library config; the steps are the flags of the ``dft.gxtb``
block (:data:`desc_utils.submit.config.GXTB_PARAMS`). The outcome is recorded under the **same keys as
the ORCA engine** (``orca_terminated_normally``, ``orca_run_time_s``, ``orca_steps_s``, ``cpu_hours``,
...), so ``orcadesc status`` and ``orcadesc cost`` work for either engine without knowing which ran.

Each step runs in its own folder so its output files cannot collide (``xtb`` always writes ``charges``,
``wbo``, ``g98.out`` under fixed names)::

    conf_N/gxtb/{optimization,singlepoint,frequency,oxidation,reduction}/

``conf_N/optimization.xyz`` is then the geometry the properties are on — the optimised one with
``opt_geom``, otherwise a copy of ``input.xyz`` — because that is the file the morfeus/geometry
descriptors read, whichever engine produced the library.

**A failed optional step does not fail the conformer.** The frequency step is the fragile one: xtb's
numerical Hessian needs every displaced geometry to converge, which it does for small organics but not
reliably for transition-metal complexes (measured on an L-NiH2 complex: ~10 min, then
``hessian_numhess: Hessian element ... is not symmetric``). Losing the vibration descriptors should not
throw away the charges, orbitals and redox states that did come out, so a step that fails is noted in
``gxtb_failed_steps`` and the run carries on.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from desc_utils.submit import records
from desc_utils.submit.config import Config
from desc_utils.submit.tasks import TermWatch, allocated_cpus, run_watching_for_term, terminated_outcome

#: the steps of a g-xTB conformer job, in run order. ``singlepoint`` is the only one that must succeed:
#: without it there is nothing to condense.
STEPS = ("optimization", "singlepoint", "frequency", "oxidation", "reduction")
REQUIRED_STEP = "singlepoint"
#: what the molecule's ``dft.engine`` record says, next to ``"ORCA"`` for the other engine
ENGINE_LABEL = "g-xTB"


def thread_env(threads: int) -> dict[str, str]:
    """The environment one ``xtb`` step runs with, the same settings ``bceg.external.run`` uses.

    ``OPENBLAS_NUM_THREADS=1`` matters: the conda-forge xtb links the pthreads OpenBLAS, which prints
    ``OpenBLAS Warning : Detect OpenMP Loop`` once per BLAS call when it is called from inside an
    OpenMP region. One BLAS thread skips that check and xtb keeps its OpenMP threads.
    """
    return {"OMP_NUM_THREADS": f"{threads},1", "MKL_NUM_THREADS": str(threads),
            "OMP_STACKSIZE": "4G", "OPENBLAS_NUM_THREADS": "1"}


def uhf(multiplicity: int | None) -> int:
    """Unpaired electrons from the multiplicity; 0 when it is unknown."""
    return max(0, int(multiplicity) - 1) if multiplicity else 0


def step_command(step: str, exe: str, opts: dict, charge: int, multiplicity: int | None) -> list[str]:
    """The ``xtb`` command line for one step.

    ``--uhf`` is only passed for an open shell: with the flag g-xTB always runs unrestricted, even for
    0 unpaired electrons (the same rule as ``bceg.refine.gxtb``). The redox steps change the electron
    count by one, which flips the parity of the unpaired count.
    """
    cmd = [exe, "start.xyz", "--gxtb"]
    n_unpaired = uhf(multiplicity)
    if step == "optimization":
        cmd += ["--opt", opts["opt_level"]]
    elif step == "frequency":
        cmd += ["--hess"]
    elif step in ("oxidation", "reduction"):
        charge += 1 if step == "oxidation" else -1
        # removing or adding one electron turns a closed shell into a doublet and vice versa
        n_unpaired = n_unpaired + 1 if n_unpaired == 0 else n_unpaired - 1
    elif step == "singlepoint":
        if opts.get("fukui"):
            cmd += ["--vfukui"]
        if opts.get("molden"):
            cmd += ["--molden"]       # writes molden.input; the runner gzips it (gxtb_condense.compress_molden)
    cmd += ["--chrg", str(charge)]
    if n_unpaired:
        cmd += ["--uhf", str(n_unpaired)]
    if opts.get("accuracy"):
        cmd += ["--acc", str(opts["accuracy"])]
    if opts.get("etemp"):
        cmd += ["--etemp", str(opts["etemp"])]
    return cmd


def steps_to_run(opts: dict) -> list[str]:
    """The steps the ``dft.gxtb`` flags ask for, in run order."""
    wanted = {"singlepoint"}
    if opts.get("opt_geom"):
        wanted.add("optimization")
    if opts.get("freq_calc"):
        wanted.add("frequency")
    if opts.get("vertical_redox"):
        wanted |= {"oxidation", "reduction"}
    return [s for s in STEPS if s in wanted]


#: what :func:`run_gxtb_conformer` returns for a conformer it never started because the job was
#: already being stopped; :func:`run_gxtb_chunk` then records it as ``timeout`` (or ``failed``)
NOT_STARTED = "not_started"


def resolve_workers(workers: int | None, cpus: int, n_conformers: int) -> tuple[int, int]:
    """``(workers, threads per worker)`` for a chunk of ``n_conformers`` on ``cpus`` CPUs.

    ``workers`` None: one per CPU. Never more workers than conformers, so a chunk of one (``batch_size:
    1``, the default) runs exactly as before the batching: one xtb with every CPU of the allocation.
    """
    workers = max(1, min(int(workers or cpus), n_conformers))
    return workers, max(1, cpus // workers)


def run_gxtb_chunk(cfg: Config, inchikey: str, conformers: list[str], gxtb_cmd: str | None = None) -> dict[str, str]:
    """Run a chunk of one molecule's conformers in one array element; returns {conformer: status}.

    ``dft.gxtb.workers`` conformers run at once, each with ``allocated CPUs // workers`` OpenMP
    threads. The xtb runs are subprocesses, so threads are enough to keep them going in parallel; the
    one SIGTERM handler (:class:`desc_utils.submit.tasks.TermWatch`) lives in this, the main, thread.

    On SIGTERM (walltime or ``orcadesc cancel``) the running conformers are recorded like a lone
    conformer would be (``timeout``, or ``failed`` for a cancel), and so are the ones not started yet,
    with ``reason`` saying so: left at ``submitted`` they would look in progress and need ``--force``,
    while a ``timeout`` is rerun by the next plain ``submit``. A conformer whose run raises (no binary, a plain xtb, ...) is recorded as
    ``failed`` with the error and the others carry on: one bad folder does not cost the rest of the chunk.
    """
    lib = cfg.library_dir
    nprocs = (records.load_metadata(lib, inchikey).get("dft") or {}).get("nprocs")
    workers, threads = resolve_workers(cfg.gxtb.get("workers"), allocated_cpus(nprocs), len(conformers))
    print(f"{inchikey}: {len(conformers)} conformer(s), {workers} worker(s) x {threads} thread(s)")
    statuses: dict[str, str] = {}

    def one(conformer: str) -> str:
        try:
            return run_gxtb_conformer(cfg, inchikey, conformer, gxtb_cmd=gxtb_cmd, cpus=threads, watch=watch)
        except Exception as exc:  # noqa: BLE001 -- recorded, then the next conformer runs
            traceback.print_exc()
            records.update_metadata(lib, inchikey, {"conformers": {conformer: {
                "status": "failed", "reason": f"{type(exc).__name__}: {exc}", "finished_at": records.now()}}})
            return "failed"

    with TermWatch() as watch:
        if workers == 1:
            for conformer in conformers:
                statuses[conformer] = one(conformer)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for conformer, status in zip(conformers, pool.map(one, conformers)):
                    statuses[conformer] = status

    not_started = [c for c, st in statuses.items() if st == NOT_STARTED]
    if not_started:
        job_env = {"slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                   "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID")}
        outcome = terminated_outcome(lib, job_env)
        reason = "not started: the job was stopped before reaching it"
        if outcome.get("reason"):
            reason = f"{outcome['reason']}; {reason}"
        records.update_metadata(lib, inchikey, {"conformers": {
            c: {**outcome, "reason": reason, "finished_at": records.now()} for c in not_started}})
        for c in not_started:
            statuses[c] = outcome["status"]
        print(f"{inchikey}: {len(not_started)} conformer(s) not started, recorded as {outcome['status']}")
    return statuses


def run_gxtb_conformer(cfg: Config, inchikey: str, conformer: str, gxtb_cmd: str | None = None,
                       cpus: int | None = None, watch: TermWatch | None = None) -> str:
    """Run the g-xTB steps for one conformer folder and record the outcome. Returns its status.

    ``cpus``: the OpenMP threads xtb gets and the conformer's cost is counted on; default the whole
    allocation. ``watch``: the SIGTERM handler of a batched element (:func:`run_gxtb_chunk`); a conformer
    reached after it fired is not started and :data:`NOT_STARTED` is returned."""
    from desc_utils import gxtb_condense as gc
    from desc_utils import gxtb_parse as gp

    lib = cfg.library_dir
    conf_dir = lib / inchikey / conformer
    start = conf_dir / "input.xyz"
    if not start.is_file():
        raise FileNotFoundError(f"{conf_dir}: no input.xyz (prepare step missing)")
    exe = gxtb_cmd or cfg.slurm.get("gxtb_cmd") or cfg.gxtb.get("executable") or "xtb"
    resolved = shutil.which(exe) or (exe if Path(exe).is_file() else None)
    if not resolved:
        raise FileNotFoundError(
            f"{exe} not found on PATH; put the g-xTB build of xtb there in slurm.setup, or set "
            f"dft.gxtb.executable / slurm.gxtb_cmd to its path")

    meta = records.load_metadata(lib, inchikey)
    dft_meta = meta.get("dft") or {}
    opts = cfg.gxtb
    # dft.constrained: the atoms stages 2 to 4 placed stay put in the optimisation step. As for ORCA, the
    # constraints only bind a geometry optimisation; a single point has no geometry to hold.
    fixed = sorted(int(i) for i in (dft_meta.get("fixed_atoms") or {}))
    charge = dft_meta.get("charge") or 0
    multiplicity = dft_meta.get("multiplicity")
    if watch is not None and watch.terminated:
        return NOT_STARTED                # the job is being stopped: leave the record at submitted
    cpus = cpus or allocated_cpus(dft_meta.get("nprocs"))
    job_env = {"slurm_job_id": os.environ.get("SLURM_JOB_ID"),
               "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
               "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
               "host": os.uname().nodename, "cpus": cpus}
    records.update_metadata(lib, inchikey, {"conformers": {conformer: {
        "status": "running", "started_at": records.now(), "gxtb_cmd": resolved, **job_env}}})

    run_dir = conf_dir / gc.RUN_DIRNAME
    # A rerun (a failed conformer, or a completed one with --rerun-completed) starts from nothing. xtb writes
    # charges, wbo, g98.out and molden.input under fixed names, so a step that fails or is skipped this time
    # would otherwise leave the previous run's files to be condensed; and if the single point fails there is
    # no new condensed file to replace the old one.
    shutil.rmtree(run_dir, ignore_errors=True)
    (conf_dir / gc.CONDENSED_FILENAME).unlink(missing_ok=True)
    run_dir.mkdir(parents=True)
    geometry = start                      # the geometry each step runs on; the optimisation replaces it
    t0 = time.time()
    step_times: dict[str, dict] = {}
    failed: list[str] = []
    timed_out = False
    exit_code = 0
    terminated = True
    version = None

    for step in steps_to_run(opts):
        if step == "frequency" and "optimization" in failed:
            # the Hessian is the expensive step and aborts on a geometry that is not a stationary point
            # of this method; there is no point spending minutes to fail
            failed.append(step)
            continue
        sub = run_dir / step
        sub.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(geometry, sub / "start.xyz")
        log = sub / "xtb.out"
        s0 = time.time()
        cmd = step_command(step, resolved, opts, charge, multiplicity)
        if step == "optimization" and fixed:
            from bceg.xtb import fix_input      # writes xcontrol and selects the L-BFGS optimiser
            cmd += fix_input(sub, fixed)
        exit_code, timed_out = run_watching_for_term(cmd, sub, log, env=thread_env(cpus), watch=watch)
        text = log.read_text(errors="replace") if log.is_file() else ""
        gp.check_is_gxtb(text, resolved)  # a standard xtb would silently have run GFN2
        version = version or gp.version(text)
        ok = exit_code == 0 and gp.terminated_normally(text)
        step_times[step] = {"wall_s": round(time.time() - s0, 2), "ok": ok, "exit_code": exit_code}
        if timed_out:
            terminated = False
            failed.append(step)
            break
        if not ok:
            failed.append(step)
            if step == REQUIRED_STEP:
                terminated = False
                break
            continue                      # an optional step that failed costs its descriptors, not the run
        if step == "singlepoint":
            gc.compress_molden(sub)
        if step == "optimization":
            optimized = sub / "xtbopt.xyz"
            if optimized.is_file():
                geometry = optimized

    shutil.copyfile(geometry, conf_dir / "optimization.xyz")
    wall = round(time.time() - t0, 1)
    status = "completed" if REQUIRED_STEP not in failed and not timed_out else "failed"
    outcome = {
        "status": status,
        "exit_code": exit_code,
        "orca_terminated_normally": terminated,
        "orca_run_time": None,
        "orca_run_time_s": sum(s["wall_s"] for s in step_times.values()) or None,
        "orca_steps_s": {name: s["wall_s"] for name, s in step_times.items()},
        "gxtb_steps": sorted(step_times),
        "gxtb_failed_steps": failed or None,
        "wall_time_s": wall,
        "cpu_hours": round(wall / 3600 * cpus, 5),
        "finished_at": records.now(),
    }
    if timed_out:
        outcome |= terminated_outcome(lib, job_env)

    # the record the condenser reads for the job status and the per-step timings
    (run_dir / gc.RUN_RECORD).write_text(json.dumps({
        "status": outcome["status"], "detail": f"failed steps: {', '.join(failed)}" if failed else None,
        "steps": step_times, "wall_time_s": wall, "nprocs": cpus,
        "options": opts, "xtb_version": version, "charge": charge, "multiplicity": multiplicity}, indent=2))

    if cfg.condense and REQUIRED_STEP not in failed:
        outcome["condensed"] = gc.condense_folder(conf_dir, overwrite=True)
    records.update_metadata(lib, inchikey, {"conformers": {conformer: outcome}})
    print(f"{inchikey}/{conformer}: {outcome['status']} "
          f"({len(step_times)} steps, {wall:.1f}s" + (f", failed: {', '.join(failed)}" if failed else "") + ")")
    return outcome["status"]
