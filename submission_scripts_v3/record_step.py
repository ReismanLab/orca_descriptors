"""Record the outcome of one workflow stage into <inchikey>/metadata.json.

Called by the run_step() wrapper in launch_workflow.sh, once before a stage
and once after it:

    python record_step.py --lig "$SMILES" --step uma --status running
    python record_step.py --lig "$SMILES" --step uma --status failed --exit-code 137

Recording from the *driver* rather than from inside each stage script is
deliberate: the stage that broke test3 was SIGKILLed by the OOM killer, and
no in-process try/except can survive that. The shell still sees the exit
code, so the shell is the only place that can reliably write it down.

A stage left at "running" means the batch job itself died (walltime, node
failure) partway through that stage.

The pre-record is skipped when <inchikey> doesn't exist yet -- the first
stage is the one that creates it, and pre-creating the directory here would
make ligand_only_create_folder.py think the molecule had already been built.
"""

import argparse
import os
import sys
import time

from rdkit import Chem

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metadata_utils


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lig", required=True,
                        help="SMILES string of the molecule.")
    parser.add_argument("--step", required=True,
                        help="Stage name, e.g. build / crest / uma / orca_submit.")
    parser.add_argument("--status", required=True,
                        choices=["running", "completed", "failed", "skipped"],
                        help="Outcome to record for this stage.")
    parser.add_argument("--exit-code", type=int, default=None,
                        help="Exit status of the stage command.")
    args = parser.parse_args()

    mol = Chem.MolFromSmiles(args.lig)
    if mol is None:
        sys.exit(f"ERROR: invalid SMILES {args.lig!r}")
    inchi_key = Chem.rdinchi.MolToInchiKey(mol)

    if not os.path.isdir(inchi_key):
        if args.status == "running":
            return  # first stage hasn't created the directory yet
        sys.exit(f"ERROR: no directory {inchi_key} to record '{args.step}' into")

    entry = {"status": args.status}
    if args.status == "running":
        entry["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    else:
        entry["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        entry["exit_code"] = args.exit_code
        entry["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")

    metadata_utils.update_metadata(inchi_key, {"steps": {args.step: entry}})
    print(f"{inchi_key}: step {args.step} -> {args.status}"
          + (f" (exit {args.exit_code})" if args.exit_code is not None else ""))


if __name__ == "__main__":
    main()
