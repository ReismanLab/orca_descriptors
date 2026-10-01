"""SLURM scripts of the ORCA stage: the ``prepare`` array (one element per molecule) and one ORCA
array per molecule (one element per conformer). Scripts and task lists live in ``_jobs/<timestamp>/``.

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


def write_prepare_array(cfg: Config, jobdir: Path, molecules: list[dict], force: bool = False) -> tuple[Path, Path]:
    """``prepare.sbatch`` + ``prepare.jsonl``: one element per molecule ({inchikey, input_smiles})."""
    log_dir = jobdir / "slurm_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = jobdir / "prepare.jsonl"
    write_tasks(tasks_path, molecules)
    throttle = cfg.slurm.get("array_throttle")
    array = f"0-{len(molecules) - 1}" + (f"%{throttle}" if throttle else "")
    cmd = (f"python -m desc_utils.submit prepare-task {shlex.quote(str(cfg.path))} "
           f"--tasks {shlex.quote(str(tasks_path))} --index $SLURM_ARRAY_TASK_ID" + (" --force" if force else ""))
    script = jobdir / "prepare.sbatch"
    script.write_text("\n".join([*_header("orca_prepare", array, cfg.slurm["prepare"], log_dir), *_setup(cfg), cmd, ""]))
    return script, tasks_path


def write_orca_array(cfg: Config, jobdir: Path, inchikey: str, conformers: list[str]) -> tuple[Path, Path]:
    """``orca_<InChIKey>.sbatch`` + ``.jsonl``: one element per conformer folder name."""
    log_dir = jobdir / "slurm_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    name = f"orca_{inchikey}"
    tasks_path = jobdir / f"{name}.jsonl"
    write_tasks(tasks_path, [{"inchikey": inchikey, "conformer": c} for c in conformers])
    throttle = cfg.slurm.get("array_throttle")
    array = f"0-{len(conformers) - 1}" + (f"%{throttle}" if throttle else "")
    cmd = (f"python -m desc_utils.submit orca-task {shlex.quote(str(cfg.path))} "
           f"--tasks {shlex.quote(str(tasks_path))} --index $SLURM_ARRAY_TASK_ID")
    body = [
        "# ORCA in the background so the TERM trap (walltime - 10 min) reaches the Python task,",
        "# which stops ORCA and records the conformer as timeout.",
        f"{cmd} &",
        "pid=$!",
        'trap \'kill -TERM "$pid" 2>/dev/null; wait "$pid"; exit 143\' TERM',
        'wait "$pid"',
        "exit $?",
    ]
    script = jobdir / f"{name}.sbatch"
    script.write_text("\n".join([*_header(name, array, cfg.slurm["orca"], log_dir, ["#SBATCH --signal=B:TERM@600"]),
                                 *_setup(cfg), *body, ""]))
    return script, tasks_path


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
