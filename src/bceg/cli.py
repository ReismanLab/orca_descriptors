"""Command line interface: ``bceg {submit,run,run-task,status}``.

Typical use on the cluster::

    bceg submit configs/example.yaml --smiles ligands.txt --transformation NiF2
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from bceg.config import Config, read_raw, select, write_raw
from bceg.inputs import load_inputs, update_index
from bceg.layout import method_dir
from bceg.provenance import read_record
from bceg.tasks import Task, expand, run_local, run_task


def _selected_raw(args) -> dict:
    methods = [m for arg in args.methods or [] for m in arg.split(",") if m]
    return select(read_raw(args.config), smiles=args.smiles, transformations=args.transformation or None,
                  methods=methods or None, multiplicity=args.multiplicity)


def _selected_config(args) -> Config:
    return Config.from_raw(_selected_raw(args), Path(args.config).resolve())


def cmd_submit(args) -> int:
    from bceg.slurm import new_jobdir, write_jobs

    raw = _selected_raw(args)
    cfg = Config.from_raw(raw, Path(args.config).resolve())
    molecules = load_inputs(cfg.inputs)
    print(f"{len(molecules)} molecule(s) x transformations {list(cfg.transformations)} x methods {list(cfg.methods)} "
          f"x {cfg.replicates} replicates")
    tasks, problems = expand(cfg, molecules, write_params=False)
    for p in problems:
        print(f"PARAMS MISMATCH\n{p}\n", file=sys.stderr)
    if problems and not args.skip_mismatched:
        print(f"{len(problems)} method folder(s) have mismatching hyper-parameters; nothing submitted. "
              "Rename the methods/transformations, or pass --skip-mismatched to submit the rest.", file=sys.stderr)
        return 2
    print(f"{len(tasks)} task(s) pending ({'search + post-optimisation per job' if cfg.chained else 'split arrays'})")
    if not tasks or args.dry_run:
        return 0

    # Freeze the resolved configuration next to the job scripts; the jobs read this snapshot.
    jobdir = new_jobdir(cfg)
    write_raw(raw, jobdir / "config.yaml")
    cfg = Config.load(jobdir / "config.yaml")
    update_index(cfg.results_dir, molecules)
    tasks, _ = expand(cfg, molecules, write_params=True)
    write_jobs(cfg, tasks, jobdir)
    print(f"Job scripts: {jobdir}")
    if args.no_sbatch:
        print(f"Submit with:  bash {jobdir / 'submit.sh'}")
        return 0
    return subprocess.run(["bash", str(jobdir / "submit.sh")]).returncode


def cmd_run(args) -> int:
    counts = run_local(_selected_config(args))
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
    cfg = _selected_config(args)
    tasks, _ = expand(cfg, write_params=False, pending_only=False, chain=False)
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


def _add_selection(p: argparse.ArgumentParser) -> None:
    p.add_argument("config", help="YAML config (methods, transformations, post-optimisation, cluster)")
    p.add_argument("--smiles", help="text file, one SMILES per line (optional name after it), or CSV with a 'smiles' column")
    p.add_argument("--transformation", "-t", action="append",
                   help="transformation(s) from the config to apply, e.g. -t NiF2 -t none (default: all)")
    p.add_argument("--methods", "-m", action="append", help="comma-separated method names (default: all)")
    p.add_argument("--multiplicity", type=int,
                   help="spin multiplicity for the selected transformations; if it differs from the configured "
                        "value the folder becomes <transformation>_m<multiplicity>")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bceg", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("submit", help="check params logs, write SLURM job arrays and submit them")
    _add_selection(p)
    p.add_argument("--dry-run", action="store_true", help="only report what would be submitted; writes nothing")
    p.add_argument("--no-sbatch", action="store_true", help="write the job scripts but do not call sbatch")
    p.add_argument("--skip-mismatched", action="store_true", help="submit the other methods despite mismatches")
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser("run", help="run all pending tasks locally, sequentially")
    _add_selection(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("run-task", help="run a chunk of a task file (used inside SLURM arrays)")
    p.add_argument("config")
    p.add_argument("--tasks", required=True)
    p.add_argument("--chunk", type=int, required=True)
    p.add_argument("--chunk-size", type=int, default=1)
    p.set_defaults(func=cmd_run_task)

    p = sub.add_parser("status", help="summarise task status per transformation / method / step")
    _add_selection(p)
    p.set_defaults(func=cmd_status)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
