#!/usr/bin/env python3
"""
aggregate_metadata.py

Walk a set of directories, find metadata.json files (this pipeline's
per-molecule/per-run metadata records), collect them into a single
batch-level document, and flag runs whose "should-be-constant" pipeline
settings (RDKit version, CREST method, DFT level of theory, calc_param_md5,
UMA model, etc.) don't match the majority value seen across the batch.

This is run automatically as part of the workflow: launch_workflow.sh
submits slurm-aggregate.sh (via submit_aggregate.py) with a SLURM
dependency on that ligand's ORCA jobs, so the summary is rebuilt after each
ligand's DFT finishes and the last rebuild sees the whole finished batch.
Concurrent rebuilds are serialised with an flock on <out>.lock.

USAGE
-----
    # Scan every immediate subdirectory of a parent directory:
    python aggregate_metadata.py --root /path/to/batch_parent

    # Scan a specific list of directories:
    python aggregate_metadata.py --dirs runA runB runC

    # Recurse arbitrarily deep looking for metadata.json:
    python aggregate_metadata.py --root /path/to/batch_parent --recursive

    # Change output location / also emit the flat CSV:
    python aggregate_metadata.py --root . --out mybatch_metadata.json \
        --csv metadata_summary.csv

OUTPUT
------
By default a JSON file named after the scanned directory --
<main_directory>_metadata.json, written inside that directory (e.g.
scanning .../test2 produces .../test2/test2_metadata.json). It holds:

    batch_directory    - absolute path of the scanned parent
    generated_at       - local timestamp of this aggregation
    n_runs / n_flagged - counts
    consistency        - per settings field: the majority ("expected") value,
                          the full value->[directories] breakdown, and the
                          directories where the field is missing
    runs               - one entry per directory, keyed by directory name,
                          each with the run's full (nested) metadata plus
                          issues / mismatched_fields / flagged / error

A run is flagged for either of two independent reasons: "issues" (something
went wrong inside the run -- a stage that failed or was left "running", a
conformer that failed/timed out, DFT configured with no conformers) or
"mismatched_fields" (its settings disagree with the rest of the batch).

With --csv, the flat one-row-per-directory table is written as well; its
columns are the union of every dot-notation field found across all
metadata.json files (e.g. "crest.method", "dft.conformers.conf_1.status").

Directories with a missing or malformed metadata.json still appear, with an
explanatory note in "error", so nothing silently disappears.
"""

import argparse
import csv
import fcntl
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Fields that are expected to be IDENTICAL across every run in a batch
# (i.e. pipeline configuration / method choices), as opposed to fields that
# are naturally per-molecule (smiles, inchikey, energies, job ids, dates...).
# Edit this list to match what your batch actually holds constant.
# Dot-paths mirror the flattened column names.
# ---------------------------------------------------------------------------
CONSISTENCY_FIELDS = [
    "conformer_gen.embed_method",
    "conformer_gen.rdkit_version",
    "conformer_gen.xtb_preopt",
    "conformer_gen.xtb_preopt_method",
    "crest.method",
    "crest.crest_version",
    "crest.n_conf",
    "crest.ewin_kcal_mol",
    "crest.quick",
    "crest.norotmd",
    "crest.noreftopo",
    "dft.calc_param_md5",
    "dft.engine",
    "dft.orca_module",
    "dft.level_of_theory.opt_method",
    "dft.level_of_theory.opt_solvent",
    "dft.level_of_theory.svp_method",
    "dft.level_of_theory.nmr_method",
    "dft.level_of_theory.property_solvent",
    "dft.level_of_theory.opt_geom",
    "dft.level_of_theory.freq_calc",
    "dft.level_of_theory.nmr_calc",
    "dft.level_of_theory.nbo_run",
    "dft.level_of_theory.vertical_redox",
    "dft.level_of_theory.conf_search",
    "dft.level_of_theory.goat_method",
    "dft.compound_steps.optimization.extra_basis",
    "dft.compound_steps.frequency.extra_basis",
    "uma.model_name",
    "uma.optimizer",
    "uma.fmax",
    "uma.task_name",
]

# Subdirectories that are never runs (scratch / bookkeeping), skipped in
# --root mode so they don't show up as bogus "no metadata file found" rows.
SKIP_DIR_NAMES = {"__pycache__"}

# Stage / conformer outcomes that mean the run finished its work. Anything
# else is reported as an issue: "running" means the batch job died mid-stage,
# "failed" speaks for itself, and a conformer left at "submitted" once the
# batch is done means its ORCA job vanished without calling mark_orca_done.py.
# Shared with submit_all_smiles.py, which uses the same definition to decide
# whether resubmitting a ligand would accomplish anything.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metadata_utils import OK_CONFORMER_STATUSES, OK_STEP_STATUSES  # noqa: E402


def default_out_name(directory):
    """<main_directory>/<main_directory_name>_metadata.json"""
    directory = Path(directory).resolve()
    return directory / f"{directory.name}_metadata.json"


def flatten(obj, prefix=""):
    """Flatten a nested dict/list into {dot.path: scalar_value}."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten(v, key))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            key = f"{prefix}[{i}]" if prefix else f"[{i}]"
            out.update(flatten(v, key))
    else:
        out[prefix] = obj
    return out


def find_metadata_files(dirs, pattern, recursive):
    """Return {directory_label: Path_to_json_or_None}."""
    results = {}
    for d in dirs:
        d = Path(d)
        label = str(d)
        if not d.is_dir():
            results[label] = None
            continue
        if recursive:
            matches = sorted(d.rglob(pattern))
        else:
            matches = sorted(d.glob(pattern))
        results[label] = matches[0] if matches else None
        if len(matches) > 1:
            print(
                f"WARNING: {label} has {len(matches)} files matching "
                f"'{pattern}'; using {matches[0]}",
                file=sys.stderr,
            )
    return results


def load_json_safely(path):
    """Return (raw_dict, error_message_or_None)."""
    if path is None:
        return {}, "no metadata file found"
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as e:
        return {}, f"failed to parse JSON: {e}"
    return data, None


def write_csv(out_path, file_map, raws, flats, errors, present_fields, majority):
    """Flat one-row-per-directory table (union of all dot-paths as columns)."""
    all_columns = []
    seen = set()
    for flat in flats.values():
        for k in flat.keys():
            if k not in seen:
                seen.add(k)
                all_columns.append(k)

    fieldnames = (["directory"] + all_columns
                  + ["issues", "mismatched_fields", "flagged", "error"])
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label in file_map:
            flat = flats[label]
            row = {"directory": label}
            for col in all_columns:
                row[col] = flat.get(col, "")
            mismatches = mismatched_fields(flat, errors[label], present_fields, majority)
            issues = run_issues(raws[label], errors[label])
            row["issues"] = "; ".join(issues)
            row["mismatched_fields"] = "; ".join(mismatches)
            row["flagged"] = "TRUE" if (mismatches or issues or errors[label]) else "FALSE"
            row["error"] = errors[label] or ""
            writer.writerow(row)


def run_issues(raw, err):
    """Things that went wrong *inside* this run, as opposed to settings that
    disagree with the rest of the batch. Driven by the stage statuses
    record_step.py writes and the conformer statuses mark_orca_done.py writes.
    """
    if err is not None:
        return []
    issues = []

    steps = raw.get("steps") or {}
    for name in sorted(steps):
        info = steps[name] or {}
        status = info.get("status")
        if status not in OK_STEP_STATUSES:
            detail = f"step {name}={status}"
            if info.get("exit_code") is not None:
                detail += f" (exit {info['exit_code']})"
            issues.append(detail)

    conformers = (raw.get("dft") or {}).get("conformers") or {}
    for name in sorted(conformers):
        status = (conformers[name] or {}).get("status")
        if status not in OK_CONFORMER_STATUSES:
            issues.append(f"conformer {name}={status}")

    # A run that got as far as configuring DFT but has no conformers at all
    # never produced anything to compute on.
    if raw.get("dft") and not conformers:
        issues.append("dft configured but no conformers recorded")

    # launch_orca_all.py noticed calcParam.inp had been edited since these
    # conformers were computed.
    dft = raw.get("dft") or {}
    if dft.get("previous_calc_param_md5"):
        issues.append(f"calcParam.inp changed since these results "
                      f"({dft['previous_calc_param_md5']} -> {dft.get('calc_param_md5')})")

    return issues


def mismatched_fields(flat, err, present_fields, majority):
    """Settings fields where this run disagrees with the batch majority."""
    if err is not None:
        return []
    mismatches = []
    for field in present_fields:
        if field not in flat:
            mismatches.append(f"{field} (missing)")
            continue
        val = str(flat[field])
        if val != majority.get(field):
            mismatches.append(f"{field}={val} (expected {majority[field]})")
    return mismatches


def aggregate(dirs, pattern, recursive, out_path, csv_path=None):
    file_map = find_metadata_files(dirs, pattern, recursive)

    # Load everything first, so we can compute the majority ("expected")
    # value for each consistency field before judging any single run.
    raws, flats, errors = {}, {}, {}
    for label, path in file_map.items():
        raw, err = load_json_safely(path)
        raws[label] = raw
        flats[label] = flatten(raw)
        errors[label] = err

    seen = {k for flat in flats.values() for k in flat}
    present_fields = [f for f in CONSISTENCY_FIELDS if f in seen]

    # Majority value per consistency field, plus the full value -> dirs
    # breakdown, ignoring directories where the field is simply absent.
    majority = {}
    breakdown = {}
    missing_in = {}
    for field in present_fields:
        by_value = defaultdict(list)
        for label, flat in flats.items():
            if field in flat and flat[field] is not None:
                by_value[str(flat[field])].append(Path(label).name)
        if by_value:
            counts = Counter({v: len(labels) for v, labels in by_value.items()})
            majority[field] = counts.most_common(1)[0][0]
            breakdown[field] = dict(by_value)
            missing_in[field] = [
                Path(label).name for label, flat in flats.items()
                if errors[label] is None and flat.get(field) is None
            ]

    runs = {}
    n_flagged = 0
    for label, path in file_map.items():
        mismatches = mismatched_fields(flats[label], errors[label], present_fields, majority)
        issues = run_issues(raws[label], errors[label])
        flagged = bool(mismatches or issues or errors[label])
        n_flagged += flagged
        runs[Path(label).name] = {
            "directory": str(Path(label).resolve()),
            "metadata_file": str(path.resolve()) if path else None,
            "flagged": flagged,
            "issues": issues,
            "mismatched_fields": mismatches,
            "error": errors[label],
            "metadata": raws[label],
        }

    document = {
        "batch_directory": str(Path(out_path).resolve().parent),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_runs": len(runs),
        "n_flagged": n_flagged,
        "consistency": {
            field: {
                "expected": majority[field],
                "values": breakdown[field],
                "missing_in": missing_in[field],
            }
            for field in majority
        },
        "runs": runs,
    }

    with open(out_path, "w") as f:
        json.dump(document, f, indent=2, sort_keys=True)

    if csv_path:
        write_csv(csv_path, file_map, raws, flats, errors, present_fields, majority)

    return document


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--root", type=str,
                        help="Parent directory; each immediate subdirectory is treated as one run.")
    group.add_argument("--dirs", nargs="+",
                        help="Explicit list of directories to scan, one row each.")
    ap.add_argument("--pattern", default="metadata.json",
                     help="Filename (glob) to look for in each directory. Default: metadata.json")
    ap.add_argument("--recursive", action="store_true",
                     help="Search each directory recursively (rglob) instead of only its top level.")
    ap.add_argument("--out", default=None,
                     help="Output JSON path. Default: <main_directory>/<main_directory>_metadata.json")
    ap.add_argument("--csv", default=None,
                     help="Optionally also write the flat one-row-per-directory CSV here.")
    args = ap.parse_args()

    if args.root:
        parent = Path(args.root)
        if not parent.is_dir():
            sys.exit(f"ERROR: --root '{parent}' is not a directory")
        dirs = sorted(p for p in parent.iterdir()
                      if p.is_dir()
                      and p.name not in SKIP_DIR_NAMES
                      and not p.name.startswith("."))
        if not dirs:
            sys.exit(f"ERROR: no subdirectories found under '{parent}'")
        out_path = Path(args.out) if args.out else default_out_name(parent)
    else:
        dirs = args.dirs
        out_path = Path(args.out) if args.out else default_out_name(Path.cwd())

    # Several ligands' aggregation jobs can come due at the same time; the
    # last one to run is the one that sees the complete batch, so just make
    # sure they don't interleave writes to the same file.
    lock_path = out_path.with_name(out_path.name + ".lock")
    with open(lock_path, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            document = aggregate(dirs, args.pattern, args.recursive, out_path, args.csv)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)

    print(f"Wrote {document['n_runs']} runs to {out_path} "
          f"({document['n_flagged']} flagged).")
    if args.csv:
        print(f"Wrote flat table to {args.csv}")
    if document["n_flagged"]:
        print("Flagged directories:")
        for name, run in document["runs"].items():
            if run["flagged"]:
                reason = (run["error"]
                          or "; ".join(run["issues"] + run["mismatched_fields"]))
                print(f"  - {name}: {reason}")


if __name__ == "__main__":
    main()
