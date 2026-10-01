"""``python -m desc_utils.submit {submit,status,prepare-task,orca-task}``."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from bceg.provenance import ParamsMismatch, check_or_write

from desc_utils.submit import records, slurm
from desc_utils.submit.config import Config
from desc_utils.submit.source import make_source
from desc_utils.submit.status import print_status
from desc_utils.submit.tasks import prepare_molecule, run_orca_conformer

SKIP_STATES = {"complete", "in_progress"}


def cmd_submit(args) -> int:
    try:
        cfg = Config.load(args.config)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 2
    problems = slurm.check_partitions(cfg, cfg.bceg_raw())
    if problems:
        print("CONFIG ERROR:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 2
    lib = cfg.library_dir
    lib.mkdir(parents=True, exist_ok=True)

    # 1. the library's compared settings: written once, refused ever after if they differ
    try:
        verdict = check_or_write(cfg.library_params_path(), cfg.params(),
                                 rename_hint="Different settings mean a different library_dir.", write=not args.dry_run)
    except ParamsMismatch as exc:
        print(f"REFUSED\n{exc}", file=sys.stderr)
        return 2
    print(f"library {lib} ({verdict} params.json)")

    # 2. which molecules need work, from the records
    source = make_source(cfg.source)
    molecules = source.molecules()
    selected, skipped = [], []
    for i, mol in enumerate(molecules):
        state, reason = records.molecule_state(records.load_metadata(lib, mol.inchikey))
        if state in SKIP_STATES and not args.force:
            skipped.append((mol, state, reason))
            continue
        selected.append({"inchikey": mol.inchikey, "input_smiles": mol.input_smiles, "name": mol.name,
                         "state": state, "reason": reason, "smiles_line": i})
    for mol, state, reason in skipped:
        print(f"  skip  {mol.inchikey}  {state}: {reason}" + ("" if state == "complete" else " (use --force)"))
    for m in selected:
        print(f"  {'new' if m['state'] == 'absent' else m['state']:>5}  {m['inchikey']}  {m['input_smiles']}"
              + (f"  ({m['reason']})" if m["state"] != "absent" else ""))

    tasks: list = []
    if cfg.source.get("run_bceg"):
        tasks, problems = source.pending_tasks()
        for p in problems:
            print(f"bceg PARAMS MISMATCH\n{p}", file=sys.stderr)
        if problems:
            return 2
        print(f"bceg: {len(tasks)} task(s) pending for this selection")
    print(f"orca: {'on' if cfg.orca_on else 'off'}; {len(selected)} molecule(s) to prepare, {len(skipped)} skipped")
    if args.dry_run:
        return 0
    if not selected and not (cfg.source.get("run_bceg") and tasks):
        print("Nothing to submit.")
        return 0

    # 3. freeze, then bceg, then the prepare array afterany bceg's jobs
    jobdir = lib / "_jobs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    frozen = cfg.freeze(jobdir)
    print(f"frozen config: {frozen.path}")
    bceg_ids: list[str] = []
    if cfg.source.get("run_bceg"):
        try:
            bceg_ids = source.submit(jobdir, no_sbatch=args.no_sbatch)
        except (RuntimeError, ParamsMismatch) as exc:
            print(f"bceg submission failed: {exc}\nNothing was queued; the ORCA stage was not submitted. "
                  f"Fix the cause and run the same command again ({jobdir} can be deleted).", file=sys.stderr)
            return 1
    if not cfg.orca_on:
        print("orca: off; ensembles only. Re-run with orca: on to refine them.")
        return 0
    if not selected:
        print("No molecule to prepare.")
        return 0
    for m in selected:
        records.register_molecule(lib, m["inchikey"], {"smiles": m["input_smiles"], "smiles_line": m["smiles_line"],
                                                       "name": m["name"]})
    script, _ = slurm.write_prepare_array(frozen, jobdir, [{"inchikey": m["inchikey"], "input_smiles": m["input_smiles"]}
                                                           for m in selected], force=args.force)
    if args.no_sbatch:
        print(f"prepare array written, not submitted: sbatch {script}")
        return 0
    job_id = slurm.sbatch(script, dependency=bceg_ids or None)
    for m in selected:
        records.update_metadata(lib, m["inchikey"], {
            "inchikey": m["inchikey"], "smiles": m["input_smiles"],
            "orca_stage": {"status": "queued", "prepare_job_id": job_id, "after_bceg_jobs": bceg_ids,
                           "submitted_at": records.now()}})
    print(f"prepare array {job_id} ({len(selected)} molecule(s))" + (f" after bceg jobs {', '.join(bceg_ids)}" if bceg_ids else ""))
    return 0


def cmd_status(args) -> int:
    print_status(Config.load(args.config))
    return 0


def cmd_prepare_task(args) -> int:
    cfg = Config.load(args.config)
    task = slurm.read_task(Path(args.tasks), args.index)
    try:
        status = prepare_molecule(cfg, task["inchikey"], task["input_smiles"], force=args.force)
    except ParamsMismatch as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0 if status == "prepared" else 1


def cmd_orca_task(args) -> int:
    cfg = Config.load(args.config)
    task = slurm.read_task(Path(args.tasks), args.index)
    status = run_orca_conformer(cfg, task["inchikey"], task["conformer"])
    return 0 if status == "completed" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m desc_utils.submit", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("submit", help="run bceg for the input list, then queue the ORCA stage")
    p.add_argument("config")
    p.add_argument("--dry-run", action="store_true", help="report what would be submitted; writes nothing")
    p.add_argument("--force", action="store_true", help="also resubmit molecules recorded complete or in progress")
    p.add_argument("--no-sbatch", action="store_true", help="write the job scripts but do not call sbatch")
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser("status", help="state of every molecule of the input list, from the records")
    p.add_argument("config")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("prepare-task", help="(cluster) split one molecule's ensemble and submit its ORCA array")
    p.add_argument("config")
    p.add_argument("--tasks", required=True)
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_prepare_task)

    p = sub.add_parser("orca-task", help="(cluster) run ORCA for one conformer and record the outcome")
    p.add_argument("config")
    p.add_argument("--tasks", required=True)
    p.add_argument("--index", type=int, required=True)
    p.set_defaults(func=cmd_orca_task)

    args = parser.parse_args(argv)
    return args.func(args)
