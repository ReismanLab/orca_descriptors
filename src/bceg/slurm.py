"""SLURM job-array generation.

``mode: chained`` (default): one array per method; each array element runs the
search of one replicate and immediately its post-optimisation.
``mode: split``: one array per search method plus one per post-optimisation
level, the latter starting (``afterany``) once all search arrays have finished.

Config (``slurm:`` block)::

    slurm:
      mode: chained
      setup: ["module load orca", "source activate bceg"]   # run before each task
      tasks_per_job: 1          # tasks executed sequentially per array element
      array_throttle: 50        # max simultaneously running array elements
      cpu: {partition: ..., time: "24:00:00", cpus_per_task: 8, mem: 16G, account: ...}
      gpu: {partition: ..., time: "12:00:00", cpus_per_task: 8, mem: 32G, gres: "gpu:1"}

Methods and ``post_optimization`` may override the profile with their own ``slurm:`` mapping.
"""

from __future__ import annotations

import math
import shlex
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from bceg.config import Config
from bceg.tasks import Task

_SBATCH_KEYS = {"partition", "account", "time", "cpus_per_task", "mem", "gres", "qos", "constraint", "nodes", "ntasks"}


def _directives(profile: dict) -> list[str]:
    lines = []
    for k, v in profile.items():
        if k == "extra":
            lines += [f"#SBATCH {x}" for x in v]
        elif k in _SBATCH_KEYS and v is not None:
            lines.append(f"#SBATCH --{k.replace('_', '-')}={v}")
        elif k not in _SBATCH_KEYS:
            raise ValueError(f"Unknown slurm key {k!r}; allowed: {sorted(_SBATCH_KEYS | {'extra'})}")
    return lines


def new_jobdir(cfg: Config) -> Path:
    jobdir = cfg.results_dir / "_jobs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    (jobdir / "slurm_logs").mkdir(parents=True, exist_ok=True)
    return jobdir


def write_jobs(cfg: Config, tasks: list[Task], jobdir: Path | None = None) -> Path:
    """Write task files, sbatch scripts and submit.sh into ``jobdir``; returns it."""
    s = cfg.slurm
    per_job = int(s.get("tasks_per_job", 1))
    throttle = s.get("array_throttle")
    bceg = s.get("command", "bceg")
    jobdir = jobdir or new_jobdir(cfg)
    (jobdir / "slurm_logs").mkdir(parents=True, exist_ok=True)

    groups: dict[tuple[str, str], list[Task]] = defaultdict(list)
    for t in tasks:
        groups[(t.kind, cfg.refiner.lot if t.kind == "refine" else t.method)].append(t)

    submit = ["#!/bin/bash", "set -euo pipefail", f"cd {shlex.quote(str(jobdir))}", 'deps=""']
    refine_lines = []
    for (kind, label), group in sorted(groups.items()):
        name = f"{kind}_{label}"
        (jobdir / f"{name}.jsonl").write_text("".join(t.to_json() + "\n" for t in group))
        n_jobs = math.ceil(len(group) / per_job)
        overrides = cfg.refiner_slurm if kind == "refine" else cfg.methods[label].slurm
        resource = "gpu" if any(t.resource == "gpu" for t in group) else "cpu"
        profile = {**s.get(resource, {}), **overrides}
        array = f"0-{n_jobs - 1}" + (f"%{throttle}" if throttle else "")
        script = [
            "#!/bin/bash",
            f"#SBATCH --job-name=bceg_{name}",
            f"#SBATCH --array={array}",
            f"#SBATCH --output={jobdir}/slurm_logs/{name}_%A_%a.out",
            *_directives(profile),
            "",
            *s.get("setup", []),
            "",
            f"{bceg} run-task {shlex.quote(str(cfg.path))} --tasks {shlex.quote(str(jobdir / f'{name}.jsonl'))} "
            f"--chunk $SLURM_ARRAY_TASK_ID --chunk-size {per_job}",
        ]
        (jobdir / f"{name}.sbatch").write_text("\n".join(script) + "\n")
        if kind != "refine":
            submit += [f"jid=$(sbatch --parsable {name}.sbatch)", 'deps="$deps:${jid%%;*}"', f'echo "{name}: $jid"']
        else:
            refine_lines += [
                f'if [ -n "$deps" ]; then jid=$(sbatch --parsable --dependency=afterany$deps {name}.sbatch); '
                f"else jid=$(sbatch --parsable {name}.sbatch); fi",
                f'echo "{name}: $jid"',
            ]
    submit += refine_lines
    (jobdir / "submit.sh").write_text("\n".join(submit) + "\n")
    (jobdir / "submit.sh").chmod(0o755)
    return jobdir
