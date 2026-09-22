#!/usr/bin/bash
#SBATCH -J condense
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH -o condense_%j.out
#SBATCH -e condense_%j.err

# Condense a whole library on a compute node, so that nothing heavy runs on the login node:
#
#     sbatch slurm-condense.sh /path/to/<LIB>
#
# Writes <LIB>/<molecule>/conf_N/orca_condensed.json, one small JSON per conformer instead of
# ~70 MB of raw ORCA output, then checks every file it wrote. Only those files, plus the
# optimization.xyz next to them, have to be copied to your laptop (step C of the README).
#
# Standard-library Python only: no conda environment, no pip install, just a git clone of
# orca_descriptors. Folders that already hold an orca_condensed.json are skipped, so re-running
# this after a new batch of ORCA jobs only does the new ones.
#
# The job fails (non-zero exit, FAILED in sacct) if any conformer could not be condensed or if a
# written file does not check out; condense_<jobid>.out names the offending folders.

set -euo pipefail

REPO=/path/to/orca_descriptors      # <- edit: the clone on the cluster, the folder holding desc_utils/
LIB=${1:-}                          # <- or edit a default library path here instead of passing one

if [[ -z $LIB ]]; then
    echo "usage: sbatch $0 /path/to/<LIB>" >&2
    exit 2
fi
if [[ ! -d $LIB ]]; then
    echo "no such library folder: $LIB" >&2
    exit 2
fi
if [[ ! -d $REPO/desc_utils ]]; then
    echo "REPO does not hold desc_utils/, edit it in $0: $REPO" >&2
    exit 2
fi

module load python                  # any python 3.8 or newer

# One worker per allocated core, whatever --cpus-per-task above ends up being.
JOBS=${SLURM_CPUS_PER_TASK:-4}

echo "library: $LIB"
echo "repo:    $REPO"
echo "python:  $(python -V 2>&1), $JOBS worker(s)"

# The check still runs when a conformer failed: one dead ORCA job out of three hundred should not
# hide the state of the other 299. The job itself fails if either command did.
status=0
PYTHONPATH=$REPO python -m desc_utils.condense --root "$LIB" --jobs "$JOBS" || status=1
PYTHONPATH=$REPO python -m desc_utils.condense --root "$LIB" --check || status=1
exit $status
