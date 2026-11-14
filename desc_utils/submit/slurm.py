"""SLURM scripts of the ORCA stage: the ``prepare`` array (one element per molecule) and one ORCA
array per molecule (one element per conformer, or per chunk of conformers packed onto one node with
``slurm.orca_batch``). Scripts and task lists live in ``_jobs/<timestamp>/``.

Both arrays run ``python -m desc_utils.submit <task> <frozen config> --tasks <jsonl> --index $SLURM_ARRAY_TASK_ID``
after the ``slurm.setup`` lines (activate ``gp_env``, load the ORCA module, NBO variables, ...).

The ORCA script runs the Python task in the background and forwards SIGTERM to it: SLURM delivers
``--signal=B:TERM@600`` to the batch shell only, and bash does not run traps while a foreground child
is running, so a walltime kill would otherwise never be recorded as ``timeout``.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

from desc_utils.submit.config import REPO_ROOT, Config

_SBATCH_FLAGS = {"partition", "account", "time", "mem", "mem_per_cpu", "ntasks", "cpus_per_task", "nodes",
                 "qos", "constraint", "gres"}


def directives(profile: dict) -> list[str]:
    lines = []
    for key, value in profile.items():
        if key == "extra":
            lines += [f"#SBATCH {x}" for x in value]
        elif key in _SBATCH_FLAGS and value is not None:
            lines.append(f"#SBATCH --{key.replace('_', '-')}={value}")
    return lines


def _header(name: str, array: str, profile: dict, log_dir: Path, extra: list[str] = ()) -> list[str]:
    return ["#!/bin/bash", f"#SBATCH --job-name={name}", f"#SBATCH --array={array}",
            f"#SBATCH --output={log_dir}/{name}_%A_%a.out", *directives(profile), *extra, ""]


def _setup(cfg: Config) -> list[str]:
    return [*cfg.slurm.get("setup", []), f'export PYTHONPATH={shlex.quote(str(REPO_ROOT))}"${{PYTHONPATH:+:$PYTHONPATH}}"', ""]


def write_tasks(path: Path, tasks: list[dict]) -> None:
    path.write_text("".join(json.dumps(t) + "\n" for t in tasks))


def read_task(path: Path, index: int) -> dict:
    lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip()]
    return json.loads(lines[index])


def write_prepare_array(cfg: Config, jobdir: Path, molecules: list[dict], force: bool = False,
                        rerun_completed: bool = False) -> tuple[Path, Path]:
    """``prepare.sbatch`` + ``prepare.jsonl``: one element per molecule ({inchikey, input_smiles})."""
    log_dir = jobdir / "slurm_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = jobdir / "prepare.jsonl"
    write_tasks(tasks_path, molecules)
    throttle = cfg.slurm.get("array_throttle")
    array = f"0-{len(molecules) - 1}" + (f"%{throttle}" if throttle else "")
    cmd = (f"python -m desc_utils.submit prepare-task {shlex.quote(str(cfg.path))} "
           f"--tasks {shlex.quote(str(tasks_path))} --index $SLURM_ARRAY_TASK_ID" + (" --force" if force else "")
           + (" --rerun-completed" if rerun_completed else ""))
    script = jobdir / "prepare.sbatch"
    script.write_text("\n".join([*_header("orca_prepare", array, cfg.slurm["prepare"], log_dir), *_setup(cfg), cmd, ""]))
    return script, tasks_path


def write_orca_array(cfg: Config, jobdir: Path, inchikey: str, conformers: list[str]) -> tuple[Path, Path]:
    """``orca_<InChIKey>.sbatch`` + ``.jsonl``: one element per conformer folder name.

    With ``slurm.orca_batch``, element ``i`` runs ``chunks(conformers, batch_size)[i]`` instead (task line
    ``{"inchikey", "conformers": [...]}``), several ORCA runs at once (``tasks.run_orca_chunk``), and the
    element asks for :func:`orca_batch_profile` rather than ``slurm.orca``.
    """
    batch = cfg.orca_batch
    if not batch:
        return _write_stage6_array(cfg, jobdir, inchikey, conformers, "orca", "orca-task", "ORCA")
    groups = chunks(conformers, batch["batch_size"])
    return _write_stage6_array(cfg, jobdir, inchikey, [{"inchikey": inchikey, "conformers": g} for g in groups],
                               "orca", "orca-task", "ORCA", profile_dict=orca_batch_profile(cfg))


#: the per-conformer keys of slurm.orca that do not describe a packed element's allocation
_PER_CONFORMER_KEYS = ("ntasks", "cpus_per_task", "mem", "mem_per_cpu", "nodes")


def orca_batch_profile(cfg: Config) -> dict:
    """The SLURM request of one packed ORCA element.

    ``slurm.orca.ntasks`` stays what ONE conformer gets (``%pal nprocs``, already in calcParam.inp).
    ``exclusive``: one whole node (``--nodes=1 --exclusive --mem=0``) and no ``--ntasks``, the set-up the
    ConfAnalysis DFT cost study measured at 1.03 billed CPU-h per single point; the node bills at its
    full core count whatever was asked, so the element fills it (workers = node CPUs // ntasks).
    Otherwise ``workers x ntasks`` tasks on one node, each with ``mem_per_cpu``.
    """
    orca = dict(cfg.slurm["orca"])
    batch = cfg.orca_batch
    out = {k: v for k, v in orca.items() if k not in _PER_CONFORMER_KEYS}
    out["nodes"] = 1
    if batch["exclusive"]:
        out["mem"] = 0
        out["extra"] = [*(orca.get("extra") or []), "--exclusive"]
    else:
        out["ntasks"] = batch["workers"] * int(orca.get("ntasks") or 1)
        if orca.get("mem_per_cpu"):
            out["mem_per_cpu"] = orca["mem_per_cpu"]
    return out


def chunks(conformers: list[str], size: int) -> list[list[str]]:
    """``conformers`` in consecutive groups of ``size`` (the last one shorter): one array element each."""
    size = max(1, int(size))
    return [conformers[i:i + size] for i in range(0, len(conformers), size)]


def write_gxtb_array(cfg: Config, jobdir: Path, inchikey: str, conformers: list[str]) -> tuple[Path, Path]:
    """``gxtb_<InChIKey>.sbatch`` + ``.jsonl``, the ``dft.engine: gxtb`` counterpart of the ORCA array.

    Same shape, the ``slurm.gxtb`` profile instead of ``slurm.orca``: g-xTB jobs are far shorter, so
    they are usually a few CPUs for a few hours rather than 16 CPUs for a week. A g-xTB single point
    takes seconds, less than a job's start-up, so ``dft.gxtb.batch_size`` conformers share one element
    (task line ``{"inchikey", "conformers": [...]}``); element ``i`` runs ``chunks(conformers, size)[i]``.
    Chunks never span molecules, so every array (and its cost) still belongs to one molecule.
    """
    groups = chunks(conformers, cfg.gxtb.get("batch_size") or 1)
    return _write_stage6_array(cfg, jobdir, inchikey, [{"inchikey": inchikey, "conformers": g} for g in groups],
                               "gxtb", "gxtb-task", "g-xTB")


def _write_stage6_array(cfg: Config, jobdir: Path, inchikey: str, conformers: list,
                        profile: str, subcommand: str, label: str,
                        profile_dict: dict | None = None) -> tuple[Path, Path]:
    """``conformers``: folder names (one task each), or ready-made task dicts (the batched chunks).
    ``profile_dict``: the SLURM request, when it is not ``slurm.<profile>`` as written (a packed ORCA element)."""
    log_dir = jobdir / "slurm_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    name = f"{profile}_{inchikey}"
    tasks_path = jobdir / f"{name}.jsonl"
    write_tasks(tasks_path, [c if isinstance(c, dict) else {"inchikey": inchikey, "conformer": c}
                             for c in conformers])
    throttle = cfg.slurm.get("array_throttle")
    array = f"0-{len(conformers) - 1}" + (f"%{throttle}" if throttle else "")
    cmd = (f"python -m desc_utils.submit {subcommand} {shlex.quote(str(cfg.path))} "
           f"--tasks {shlex.quote(str(tasks_path))} --index $SLURM_ARRAY_TASK_ID")
    body = [
        f"# {label} in the background so the TERM trap (walltime - 10 min) reaches the Python task,",
        f"# which stops {label} and records the conformer as timeout.",
        f"{cmd} &",
        "pid=$!",
        'trap \'kill -TERM "$pid" 2>/dev/null; wait "$pid"; exit 143\' TERM',
        'wait "$pid"',
        "exit $?",
    ]
    script = jobdir / f"{name}.sbatch"
    request = profile_dict if profile_dict is not None else cfg.slurm[profile]
    script.write_text("\n".join([*_header(name, array, request, log_dir, ["#SBATCH --signal=B:TERM@600"]),
                                 *_setup(cfg), *body, ""]))
    return script, tasks_path


def known_partitions() -> set[str] | None:
    """Partition names from ``sinfo``; None when sinfo is unavailable (laptop, tests)."""
    try:
        out = subprocess.run(["sinfo", "-h", "-o", "%P"], capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return {p.strip().rstrip("*") for p in out.stdout.split() if p.strip()}


def requested_partitions(cfg: Config) -> dict[str, str]:
    """{dotted setting: partition} for every partition named in the library's SLURM blocks."""
    found = {}
    for prof in ("prepare", "orca"):
        p = cfg.slurm.get(prof, {}).get("partition")
        if p:
            found[f"slurm.{prof}.partition"] = p
    for prof in ("cpu", "gpu"):
        p = (cfg.slurm.get("ensembles", {}).get(prof) or {}).get("partition")
        if p:
            found[f"slurm.ensembles.{prof}.partition"] = p
    p = (cfg.search.get("slurm") or {}).get("partition")
    if p:
        found["conformer_search.slurm.partition"] = p
    p = ((cfg.post_optimization or {}).get("slurm") or {}).get("partition")
    if p:
        found["post_optimization.slurm.partition"] = p
    return found


def check_partitions(cfg: Config) -> list[str]:
    """Problems with the partitions named in the config, checked against sinfo (empty when fine or unknowable)."""
    known = known_partitions()
    if known is None:
        return []
    return [f"{where}: partition {name!r} does not exist here (sinfo knows: {', '.join(sorted(known))})"
            for where, name in requested_partitions(cfg).items() if name not in known]


def sbatch(script: Path, dependency: list[str] | None = None) -> str:
    """Submit ``script`` and return its job id. ``dependency``: job ids for ``--dependency=afterany``."""
    cmd = ["sbatch", "--parsable"]
    if dependency:
        cmd.append("--dependency=afterany:" + ":".join(str(d) for d in dependency))
    cmd.append(str(script))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(script.parent))
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed ({result.returncode}): {result.stderr.strip()}")
    return result.stdout.strip().split(";")[0]


def queued_jobs() -> list[dict] | None:
    """The user's jobs in squeue, one per array element or plain job: {array_job_id, job_id, script};
    None when squeue is unavailable (laptop, tests). ``array_job_id`` is the job id of a plain job."""
    try:
        out = subprocess.run(["squeue", "-h", "--me", "-r", "-o", "%F|%A|%o"], capture_output=True, text=True,
                             timeout=60)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    jobs = []
    for line in out.stdout.splitlines():
        parts = line.strip().split("|", 2)
        if len(parts) == 3:
            jobs.append({"array_job_id": parts[0].strip(), "job_id": parts[1].strip(), "script": parts[2].strip()})
    return jobs


#: SLURM job states of a job that may still write its records (anything else has ended)
ALIVE_STATES = {"PENDING", "RUNNING", "REQUEUED", "REQUEUE_FED", "REQUEUE_HOLD", "RESIZING", "SUSPENDED",
                "CONFIGURING", "COMPLETING", "SIGNALING", "STAGE_OUT"}


def ended_jobs(job_ids) -> dict[str, str]:
    """``{job id: state}`` of the jobs among ``job_ids`` that sacct reports as ended (TIMEOUT,
    OUT_OF_MEMORY, CANCELLED, ...). Ids are the raw ones (``SLURM_JOB_ID`` of an array element). Empty
    when sacct is unavailable (laptop, tests) or fails: a job it does not report is taken as alive."""
    ids = sorted({str(j) for j in job_ids if j})
    if not ids:
        return {}
    try:
        out = subprocess.run(["sacct", "-n", "-X", "-P", "-j", ",".join(ids), "-o", "JobIDRaw,State"],
                             capture_output=True, text=True, timeout=60)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}
    if out.returncode != 0:
        return {}
    ended = {}
    for line in out.stdout.splitlines():
        raw, _, state = line.strip().partition("|")
        state = state.split()[0] if state.strip() else ""   # "CANCELLED by 123"
        if raw in ids and state and state not in ALIVE_STATES:
            ended[raw] = state
    return ended


def scancel(job_ids: list[str]) -> None:
    """Cancel ``job_ids`` (an array id cancels every element of it)."""
    result = subprocess.run(["scancel", *[str(j) for j in job_ids]], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"scancel {' '.join(job_ids)} failed ({result.returncode}): {result.stderr.strip()}")
