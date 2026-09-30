"""Queue the batch-level metadata aggregation for one ligand's workflow.

Called at the tail of launch_workflow.sh, from the main (batch) directory,
after launch_orca_all.py has submitted this ligand's ORCA jobs:

    python submit_aggregate.py --lig "$SMILES"

The ORCA jobs are separate SLURM jobs, so by the time launch_workflow.sh
exits every conformer is still only "submitted" -- aggregating right here
would freeze that half-finished picture into the summary. Instead this
submits slurm-aggregate.sh with --dependency=afterany on this ligand's ORCA
job ids (read back out of <inchikey>/metadata.json, where launch_orca_all.py
recorded them), so aggregation runs once those jobs have finished, whatever
their outcome.

Every ligand in the batch queues one of these, and each rebuilds the whole
<main_directory>_metadata.json from scratch, so the last one to come due
captures the finished batch. aggregate_metadata.py holds an flock while it
writes, so simultaneous rebuilds don't interleave.
"""

import argparse
import os
import re
import subprocess
import sys

from rdkit import Chem

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metadata_utils


def submitted_job_ids(inchikey_dir):
    """ORCA job ids this run just submitted, from <inchikey>/metadata.json."""
    meta = metadata_utils.load_metadata(inchikey_dir)
    conformers = meta.get("dft", {}).get("conformers", {})
    ids = []
    for conf in sorted(conformers):
        info = conformers[conf] or {}
        # Only "submitted" conformers are still in the queue: mark_orca_done.py
        # rewrites the status once ORCA exits, and filtered/skipped conformers
        # never had a job at all.
        if info.get("status") == "submitted" and info.get("slurm_job_id"):
            ids.append(str(info["slurm_job_id"]))
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lig", type=str, required=True,
                        help="SMILES string of the molecule (same value the rest "
                             "of the workflow was run with).")
    parser.add_argument("--script", default="slurm-aggregate.sh",
                        help="Aggregation batch script to submit. Default: slurm-aggregate.sh")
    args = parser.parse_args()

    inchi_key = Chem.rdinchi.MolToInchiKey(Chem.MolFromSmiles(args.lig))
    if not os.path.isdir(inchi_key):
        sys.exit(f"ERROR: no directory {inchi_key} to aggregate from")
    if not os.path.exists(args.script):
        sys.exit(f"ERROR: {args.script} not found in {os.getcwd()}")

    job_ids = submitted_job_ids(inchi_key)
    cmd = ["sbatch"]
    if job_ids:
        cmd.append("--dependency=afterany:" + ":".join(job_ids))
    else:
        # No ORCA job to wait on (all conformers filtered out, or they were
        # already running from an earlier submission) -- aggregate now.
        print(f"No pending ORCA jobs for {inchi_key}; aggregating immediately.")
    cmd.append(args.script)

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout.strip() or result.stderr.strip())
    if result.returncode != 0:
        sys.exit(f"ERROR: {' '.join(cmd)} failed with code {result.returncode}")

    m = re.search(r'Submitted batch job (\d+)', result.stdout)
    if m and job_ids:
        print(f"Aggregation job {m.group(1)} will run after ORCA jobs "
              f"{', '.join(job_ids)} finish.")


if __name__ == "__main__":
    main()
