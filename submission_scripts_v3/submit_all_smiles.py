"""Submit one launch_workflow_<i>.sh per line of smiles.txt.

Ligands that are already finished (or still running) are skipped, so adding
a line to smiles.txt and re-running this only costs an allocation for the
new molecule. Use --force to resubmit regardless, and --dry-run to see what
would be submitted without queueing anything.

Before actually submitting (or re-submitting) a ligand, this also
cross-checks workflow_inputs.yaml against BOTH that ligand's own
metadata.json AND every other ligand's metadata.json already in this batch
directory -- see check_inputs.py -- and asks whether to proceed if they've
drifted apart. This is what catches a brand new ligand about to be submitted
under different settings than the rest of an in-progress batch, since a new
ligand has no history of its own to disagree with. Ligands skipped by the
run_state() check above never reach this, since there's nothing about to be
computed for them to disagree with. Pass --yes to answer that prompt
automatically instead of blocking on input() (for scripted/cron use).

The per-ligand script index tracks the line number in smiles.txt, so
launch_workflow_2.sh is always the third ligand whether or not the ones
before it were skipped.

SLURM resources for the submitted job come from workflow_inputs.yaml's
slurm.main_job block, passed as `sbatch --time=... --mem-per-cpu=...` flags
-- these override launch_workflow.sh's own #SBATCH defaults, since #SBATCH
lines are parsed before this script (or workflow_inputs.yaml) is ever read.
"""

import argparse
import os
import subprocess
import sys

from rdkit import Chem

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metadata_utils
import workflow_inputs as wi
import check_inputs

PLACEHOLDER = "smiles_ligand"
TEMPLATE = "launch_workflow.sh"

# States that mean "there is nothing useful to do by resubmitting".
SKIP_STATES = {"complete", "in_progress"}


def write_workflow_script(template, dest, ligand):
    """Substitute the SMILES into a copy of the workflow template.

    Done in Python rather than with `sed s/PLACEHOLDER/$ligand/`: SMILES
    routinely contain '/' (double-bond stereochemistry, e.g. F/C=C/F), which
    closes a sed expression early, and '&' and '\\' have their own meanings in
    a sed replacement. str.replace() has no such metacharacters.
    """
    with open(template) as f:
        text = f.read()
    if PLACEHOLDER not in text:
        sys.exit(f"ERROR: {template} does not contain the placeholder {PLACEHOLDER!r}")
    with open(dest, "w") as f:
        f.write(text.replace(PLACEHOLDER, ligand))


def sbatch_resource_overrides(cfg):
    """--time / --mem-per-cpu / --ntasks-per-node for the main launch_workflow.sh
    job, read from workflow_inputs.yaml's slurm.main_job block."""
    main_job = wi.get(cfg, "slurm.main_job", default={})
    flags = []
    if "time" in main_job:
        flags.append(f"--time={main_job['time']}")
    if "mem_per_cpu" in main_job:
        flags.append(f"--mem-per-cpu={main_job['mem_per_cpu']}")
    if "ntasks_per_node" in main_job:
        flags.append(f"--ntasks-per-node={main_job['ntasks_per_node']}")
    return flags


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smiles-file", default="smiles.txt",
                        help="One SMILES per line. Default: smiles.txt")
    parser.add_argument("--template", default=TEMPLATE,
                        help=f"Workflow script to specialise. Default: {TEMPLATE}")
    parser.add_argument("--force", action="store_true",
                        help="Resubmit every ligand, including ones already complete "
                             "or still marked in progress.")
    parser.add_argument("--yes", action="store_true",
                        help="Automatically proceed when workflow_inputs.yaml disagrees "
                             "with a ligand's recorded metadata, instead of prompting.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be submitted, but don't sbatch anything "
                             "or prompt about input disagreements.")
    args = parser.parse_args()

    if not os.path.exists(args.template):
        sys.exit(f"ERROR: {args.template} not found in {os.getcwd()}")

    cfg = wi.load_inputs()
    overrides = sbatch_resource_overrides(cfg)

    # Computed once per run (rendering the DFT template and reading every
    # <inchikey>/metadata.json in this directory), then filtered per ligand
    # below -- avoids re-scanning the whole batch directory once per line of
    # smiles.txt.
    full_batch_mismatches = check_inputs.compute_batch_mismatches(cfg)

    with open(args.smiles_file) as f:
        lines = f.readlines()

    submitted, skipped = 0, 0
    for i, line in enumerate(lines):
        ligand = line.strip()
        if not ligand or ligand.startswith("#"):
            continue

        mol = Chem.MolFromSmiles(ligand)
        if mol is None:
            print(f"[{i}] {ligand}: INVALID SMILES -- skipping", file=sys.stderr)
            skipped += 1
            continue
        inchi_key = Chem.rdinchi.MolToInchiKey(mol)

        state, reason = metadata_utils.run_state(inchi_key)
        if state in SKIP_STATES and not args.force:
            print(f"[{i}] {ligand} ({inchi_key}): {state} -- {reason}; skipping"
                  + (" (use --force to resubmit)" if state == "in_progress" else ""))
            skipped += 1
            continue

        # Only reachable for ligands about to be (re)submitted -- a ligand
        # skipped above has nothing pending that current settings could
        # disagree with. Checks BOTH this ligand's own metadata (if any) and
        # every other ligand already in this batch. In --dry-run this
        # reports disagreements without blocking on input(), since a dry run
        # shouldn't wait on a person.
        batch_mismatches_here = {k: v for k, v in full_batch_mismatches.items() if k != inchi_key}
        if args.dry_run:
            own_metadata = metadata_utils.load_metadata(inchi_key) if os.path.isdir(inchi_key) else {}
            own_mismatches = check_inputs.compute_mismatches(cfg, own_metadata)
            had_mismatches = bool(own_mismatches) or bool(batch_mismatches_here)
            if own_mismatches:
                check_inputs.print_mismatches(inchi_key, ligand, own_mismatches)
            if batch_mismatches_here:
                check_inputs.print_batch_mismatches(batch_mismatches_here)
        else:
            proceed, had_mismatches = check_inputs.confirm_or_skip(
                ligand, cfg=cfg, assume_yes=args.yes, batch_mismatches=full_batch_mismatches)
            if not proceed:
                print(f"[{i}] {ligand} ({inchi_key}): user declined to proceed "
                      f"with the changed settings; skipping")
                skipped += 1
                continue

        note = f"{state} -- {reason}" if state != "absent" else "new"
        script = f"launch_workflow_{i}.sh"
        write_workflow_script(args.template, script, ligand)

        if args.dry_run:
            print(f"[{i}] {ligand} ({inchi_key}): {note}; would submit {script}"
                  + (" (with unresolved input differences)" if had_mismatches else ""))
            submitted += 1
            continue

        sbatch_cmd = ["sbatch"] + overrides
        if had_mismatches:
            # A human already confirmed the disagreement above; tell the
            # batch job's own (non-interactive) safety check not to re-ask.
            sbatch_cmd.append("--export=ALL,FORCE_INPUT_MISMATCH=1")
        sbatch_cmd.append(script)

        result = subprocess.run(sbatch_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[{i}] {ligand}: sbatch failed: {result.stderr.strip()}", file=sys.stderr)
            continue
        print(f"[{i}] {ligand} ({inchi_key}): {note}; {result.stdout.strip()}")
        submitted += 1

    verb = "would submit" if args.dry_run else "submitted"
    print(f"\n{verb} {submitted}, skipped {skipped}.")


if __name__ == "__main__":
    main()
