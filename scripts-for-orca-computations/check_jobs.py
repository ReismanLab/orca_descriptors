"""Report the state of every ORCA job of a computation folder: queued, running, done, failed or not started.

Expects the library layout ``<computation_folder>/<inchikey>/conf_*/`` that ``orcadesc`` (and the earlier
submission scripts) produce. Queued and running jobs are found by asking SLURM (``squeue``) for the working
directory of your jobs, so run this on the cluster; elsewhere it falls back to how recently calcParam.out was
written. Standard library only; the verdict on the files is ``desc_utils.orca_parse.job_status``, the same one
the condensed file records.

Usage:
    python3 scripts-for-orca-computations/check_jobs.py [computation_folder] [--all]
"""
from __future__ import annotations

import argparse
import getpass
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

# The verdict on a folder's files (termination lines of calcParam.out, SLURM messages in error_<jobid>.txt)
# is the one desc_utils.orca_parse.job_status gives, the same the condensed file records; this script only
# adds what squeue knows. The clone is the folder above scripts-for-orca-computations/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from desc_utils.orca_parse import job_status as file_job_status  # noqa: E402  (standard library only)


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


def job_status(conf: Path, jobs: dict[str, tuple[str, str]] | None, stale_minutes: float) -> tuple[str, str]:
    """(status, detail) for one conformer folder."""
    if jobs is not None and (job := jobs.get(os.path.realpath(conf))):
        job_id, state = job
        return ("QUEUED" if state == "PENDING" else "RUNNING"), f"job {job_id} {state}"
    js = file_job_status(conf)
    if js["status"] == "not_started":
        return "NOT STARTED", js["detail"]
    if js["status"] == "completed":
        return ("CHECK", js["detail"]) if js["n_normal_terminations"] > 1 else ("OK", "")
    if js["status"] == "failed":
        if js["failed_after_success"]:
            return "CHECK", f"terminated normally, then a later run failed: {js['detail']}"
        return "FAILED", js["detail"]
    # no termination line and not in the queue: still writing, or killed without a message
    age_min = (time.time() - (conf / "calcParam.out").stat().st_mtime) / 60
    if jobs is None and age_min < stale_minutes:
        return "RUNNING?", f"calcParam.out written {age_min:.0f} min ago (squeue unavailable)"
    return "FAILED", "no termination message: job killed or crashed (look at error_*.txt / output_*.txt)"


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
