"""Report the state of every ORCA job of a computation folder: queued, running, done, failed or not started.

Expects the campaign layout ``<computation_folder>/<inchikey>/conf_*/`` created by launch_orca_all.py. Queued
and running jobs are found by asking SLURM (``squeue``) for the working directory of your jobs, so run this
on the cluster; elsewhere it falls back to how recently calcParam.out was written. Standard library only.

Usage:
    python3 scripts-for-orca-computations/check_jobs.py [computation_folder] [--all]
"""
from __future__ import annotations

import argparse
import getpass
import os
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

NORMAL = "ORCA TERMINATED NORMALLY"
ORCA_ERROR = re.compile(r"ORCA finished by error termination.*|aborting the run", re.IGNORECASE)
# Messages SLURM / Open MPI write to the job's error_<jobid>.txt when the job is killed before ORCA can.
SLURM_ERRORS = {
    "DUE TO TIME LIMIT": "killed by SLURM: time limit",
    "oom-kill": "killed by SLURM: out of memory",
    "Out Of Memory": "killed by SLURM: out of memory",
    "CANCELLED AT": "cancelled (scancel or node failure)",
    "not enough slots": "Open MPI: not enough slots for the requested processes",
}


def queued_jobs() -> dict[str, tuple[str, str]] | None:
    """Map the real working directory of each of your SLURM jobs to (job id, state); None without squeue."""
    try:
        out = subprocess.run(["squeue", "-h", "-u", getpass.getuser(), "-o", "%i|%T|%Z"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    jobs = {}
    for line in out.splitlines():
        job_id, state, workdir = line.split("|", 2)
        jobs[os.path.realpath(workdir)] = (job_id, state)
    return jobs


def latest_error_file(conf: Path) -> Path | None:
    """error_<jobid>.txt of the most recent submission (earlier ones stay in the folder after a resubmit)."""
    files = [(int(m.group(1)), p) for p in conf.glob("error_*.txt") if (m := re.fullmatch(r"error_(\d+)\.txt", p.name))]
    return max(files)[1] if files else None


def failure_reason(conf: Path, orca_error: str | None) -> str:
    err = latest_error_file(conf)
    if err is not None:
        text = err.read_text(errors="replace")
        for marker, reason in SLURM_ERRORS.items():
            if marker in text:
                return reason
    if orca_error:
        return orca_error
    return "no termination message: job killed or crashed (look at error_*.txt / output_*.txt)"


def job_status(conf: Path, jobs: dict[str, tuple[str, str]] | None, stale_minutes: float) -> tuple[str, str]:
    """(status, detail) for one conformer folder."""
    out = conf / "calcParam.out"
    if jobs is not None and (job := jobs.get(os.path.realpath(conf))):
        job_id, state = job
        return ("QUEUED" if state == "PENDING" else "RUNNING"), f"job {job_id} {state}"
    if not out.is_file():
        return "NOT STARTED", "no calcParam.out"

    # calcParam.out is appended to (>>) on every submission, so it can hold several runs: judge by the last one.
    last_normal = last_error = -1
    n_normal = 0
    orca_error = None
    with open(out, errors="replace") as fh:
        for i, line in enumerate(fh):
            if NORMAL in line:
                last_normal, n_normal = i, n_normal + 1
            elif m := ORCA_ERROR.search(line):
                if orca_error is None or last_error < last_normal:  # keep the first, most specific, line of a run
                    orca_error = m.group(0).strip()
                last_error = i

    if last_normal > last_error:
        if n_normal > 1:
            return "CHECK", f"{n_normal} normal terminations in calcParam.out (submitted several times?)"
        return "OK", ""
    if last_normal >= 0:
        return "CHECK", f"terminated normally, then a later run failed: {orca_error}"
    if last_error >= 0:
        return "FAILED", failure_reason(conf, orca_error)

    age_min = (time.time() - out.stat().st_mtime) / 60
    if jobs is None and age_min < stale_minutes:
        return "RUNNING?", f"calcParam.out written {age_min:.0f} min ago (squeue unavailable)"
    return "FAILED", failure_reason(conf, None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("computation_folder", type=Path, nargs="?", default=Path("."))
    parser.add_argument("--all", action="store_true", help="also list the jobs that finished normally")
    parser.add_argument("--stale-minutes", type=float, default=30,
                        help="without squeue, an unfinished calcParam.out written more recently than this "
                             "counts as still running (default: 30)")
    args = parser.parse_args()

    jobs = queued_jobs()
    if jobs is None:
        print("squeue not available: running jobs are guessed from calcParam.out modification times\n")

    counts = Counter()
    mol_dirs = sorted(p for p in args.computation_folder.iterdir() if p.is_dir() and not p.name.startswith("."))
    for mol in mol_dirs:
        confs = sorted((c for c in mol.glob("conf_*") if c.is_dir()), key=lambda c: int(c.name[5:]) if c.name[5:].isdigit() else 0)
        for conf in confs:
            status, detail = job_status(conf, jobs, args.stale_minutes)
            counts[status] += 1
            if status != "OK" or args.all:
                print(f"{status:<12} {mol.name}/{conf.name}" + (f"  {detail}" if detail else ""))

    order = ["OK", "RUNNING", "QUEUED", "RUNNING?", "FAILED", "CHECK", "NOT STARTED"]
    print("\n" + ", ".join(f"{s}: {counts[s]}" for s in order if counts[s]) + f"  (total {sum(counts.values())})")


if __name__ == "__main__":
    main()
