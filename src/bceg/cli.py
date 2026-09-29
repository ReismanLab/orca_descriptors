"""Command line interface: ``bceg {plan,run,run-task,status}``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from bceg.config import Config
from bceg.inputs import load_inputs, update_index
from bceg.layout import method_dir
from bceg.provenance import read_record
from bceg.tasks import Task, expand, run_local, run_task


def cmd_plan(args) -> int:
    cfg = Config.load(args.config)
    molecules = load_inputs(cfg.inputs)
    if not args.dry_run:
        update_index(cfg.results_dir, molecules)
    tasks, problems = expand(cfg, molecules, write_params=not args.dry_run)
    for p in problems:
        print(f"PARAMS MISMATCH\n{p}\n", file=sys.stderr)
    if problems and not args.skip_mismatched:
        print(f"{len(problems)} method folder(s) have mismatching hyper-parameters; no job scripts written. "
              "Rename the methods, or pass --skip-mismatched to plan the rest.", file=sys.stderr)
        return 2
    n_search = sum(t.kind == "search" for t in tasks)
    print(f"{n_search} search and {len(tasks) - n_search} refine task(s) pending")
    if not tasks or args.dry_run:
        return 0
    from bceg.slurm import write_jobs

    jobdir = write_jobs(cfg, tasks)
    print(f"Job scripts written to {jobdir}\nSubmit with:  bash {jobdir / 'submit.sh'}")
    return 0


def cmd_run(args) -> int:
    counts = run_local(Config.load(args.config))
    print(counts)
    return int(counts.get("failed", 0) > 0)


def cmd_run_task(args) -> int:
    cfg = Config.load(args.config)
    lines = [ln for ln in Path(args.tasks).read_text().splitlines() if ln.strip()]
    start = args.chunk * args.chunk_size
    failed = False
    for line in lines[start : start + args.chunk_size]:
        task = Task.from_json(line)
        status = run_task(cfg, task)
        print(f"[{status:>18}] {task.id}", flush=True)
        failed |= status in ("failed", "waiting_for_search")
    return int(failed)


def cmd_status(args) -> int:
    cfg = Config.load(args.config)
    tasks, _ = expand(cfg, write_params=False, pending_only=False)
    rows = []
    for t in tasks:
        md = method_dir(cfg.results_dir, t.transformation, t.inchikey, t.method)
        lot = None if t.kind == "search" else cfg.refiner.lot
        rec = read_record(md.run_record(t.replicate, lot))
        rows.append({"transformation": t.transformation, "method": t.method,
                     "step": "search" if lot is None else lot, "status": rec["status"] if rec else "missing"})
    if not rows:
        print("No tasks.")
        return 0
    table = pd.DataFrame(rows).value_counts().unstack("status", fill_value=0)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(table)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bceg", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="check params logs, list pending tasks and write SLURM job arrays")
    p.add_argument("config")
    p.add_argument("--dry-run", action="store_true", help="only report pending tasks")
    p.add_argument("--skip-mismatched", action="store_true", help="plan the other methods despite mismatches")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("run", help="run all pending tasks locally, sequentially")
    p.add_argument("config")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("run-task", help="run a chunk of a task file (used inside SLURM arrays)")
    p.add_argument("config")
    p.add_argument("--tasks", required=True)
    p.add_argument("--chunk", type=int, required=True)
    p.add_argument("--chunk-size", type=int, default=1)
    p.set_defaults(func=cmd_run_task)

    p = sub.add_parser("status", help="summarise task status per method")
    p.add_argument("config")
    p.set_defaults(func=cmd_status)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
