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
#     sbatch slurm-condense.sh /path/to/<LIB> --overwrite     # rebuild every file, completed ones too
#
# Anything after the library path is handed to `python -m desc_utils.condense` as is (--overwrite,
# --nbo-full all, --mo-states ..., see `python -m desc_utils.condense -h`); the check that follows
# takes none of them.
#
# Writes <LIB>/<molecule>/conf_N/orca_condensed.json, one small JSON per conformer instead of
# ~70 MB of raw ORCA output, then checks every file it wrote. The log of the run,
# condense_<jobid>.out and .err, lands in the directory you submit from, which is not necessarily
# the library: cd there first, or pass sbatch -o /somewhere/condense_%j.out -e ... . Only those files, plus the
# optimization.xyz next to them, have to be copied to your laptop (docs/condense.md, Transfer).
#
# Standard-library Python only: no conda environment, no pip install, just a git clone of
# orca_descriptors. Every file records the state of its ORCA job (job.status: completed, running,
# failed) and the log ends with a count of them. Folders whose file says "completed" are skipped, the
# others are rebuilt, so re-running this after a new batch or a resubmission only does what changed.
#
# The job fails (non-zero exit, FAILED in sacct) if any conformer could not be condensed or if a
# written file does not check out; condense_<jobid>.out names the offending folders.

set -euo pipefail

# The clone of orca_descriptors on the cluster, the folder that holds desc_utils/. Nothing to edit
# as long as this script sits in it, which is the normal case: it is worked out below from the path
# the job was submitted with. Set it here, or with `sbatch --export=ALL,REPO=...`, if you moved the
# script out of the clone.
REPO=${REPO:-}
LIB=${1:-}                          # <- or edit a default library path here instead of passing one
shift $(( $# > 0 ? 1 : 0 ))
EXTRA=("$@")                        # forwarded to desc_utils.condense, e.g. --overwrite

if [[ -z $LIB ]]; then
    echo "usage: sbatch slurm-condense.sh /path/to/<LIB> [condense options, e.g. --overwrite]" >&2
    exit 2
fi
if [[ ! -d $LIB ]]; then
    echo "no such library folder: $LIB" >&2
    exit 2
fi

if [[ -z $REPO ]]; then
    # sbatch runs a *copy* of this script out of /var/spool, so $BASH_SOURCE says nothing about
    # where the clone is. Slurm remembers the path the job was submitted with; the clone is the
    # folder above scripts-for-orca-computations/.
    if [[ -n ${SLURM_JOB_ID:-} ]] && command -v scontrol >/dev/null 2>&1; then
        submitted=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | tr ' ' '\n' | sed -n 's/^Command=//p' | head -1)
        [[ -n ${submitted:-} && -e $submitted ]] && REPO=$(cd -- "$(dirname -- "$submitted")/.." && pwd)
    fi
    # Not under slurm (running it by hand): the script itself is the original.
    [[ -n $REPO ]] || REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
fi

if [[ ! -d $REPO/desc_utils ]]; then
    echo "no desc_utils/ in $REPO, so that is not the orca_descriptors clone." >&2
    echo "Submit the script from inside the clone, or name the clone yourself:" >&2
    echo "  sbatch --export=ALL,REPO=\$HOME/orca_descriptors slurm-condense.sh /path/to/<LIB>" >&2
    exit 2
fi

module load python                  # any python 3.8 or newer

# One worker per allocated core, whatever --cpus-per-task above ends up being.
JOBS=${SLURM_CPUS_PER_TASK:-4}

echo "library: $LIB"
echo "repo:    $REPO"
echo "python:  $(python -V 2>&1), $JOBS worker(s)"
(( ${#EXTRA[@]} )) && echo "options: ${EXTRA[*]}"

# The check still runs when a conformer failed: one dead ORCA job out of three hundred should not
# hide the state of the other 299. The job itself fails if either command did.
status=0
PYTHONPATH=$REPO python -m desc_utils.condense --root "$LIB" --jobs "$JOBS" ${EXTRA[@]+"${EXTRA[@]}"} || status=1
PYTHONPATH=$REPO python -m desc_utils.condense --root "$LIB" --check || status=1
exit $status
