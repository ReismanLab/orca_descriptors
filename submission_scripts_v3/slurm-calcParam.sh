#!/usr/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH -t 160:00:00
#SBATCH -o output_%j.txt
#SBATCH -e error_%j.txt
#SBATCH --signal=B:TERM@600
#SBATCH --mem-per-cpu=32G

# this helps remove silent quit from NBO
ulimit -s unlimited

module load orca/6.1.0
module load python

#  orca uses environment variables to find nbo
#  even though they have exe extensions they are linux binaries
export NBOBIN=/resnick/groups/ReismanLab/nbo7/bin
export NBOEXE=/resnick/groups/ReismanLab/nbo7/bin/nbo7.i8.exe
export GENEXE=/resnick/groups/ReismanLab/nbo7/bin/gennbo.i8.exe
export PATH="$PATH:$NBOBIN"

# ---------------------------------------------------------------------------
# Provenance: record this conformer's outcome into <inchikey>/metadata.json.
# This script runs from inside conf_N, so the launch directory holding
# mark_orca_done.py / metadata_utils.py is two levels up.
# ---------------------------------------------------------------------------
MARK_SCRIPT="$(cd ../.. && pwd)/mark_orca_done.py"

mark() {
    if [ -f "$MARK_SCRIPT" ]; then
        python "$MARK_SCRIPT" "$@" || echo "WARNING: could not update metadata.json" >&2
    else
        echo "WARNING: $MARK_SCRIPT not found; metadata.json not updated" >&2
    fi
}

# --signal=B:TERM@600 (above) delivers SIGTERM to this script 10 minutes before
# the walltime expires. Catch it so a job that runs out of time is recorded as
# "timeout" instead of being left at "submitted" indefinitely.
on_term() {
    trap - TERM
    kill "$orca_pid" 2>/dev/null
    mark --status timeout --exit-code 143
    exit 143
}
trap on_term TERM

# Backgrounded + wait, rather than run in the foreground, so bash can service
# the TERM trap while ORCA is still running.
$(which orca) calcParam.inp >> calcParam.out &
orca_pid=$!
wait "$orca_pid"
rc=$?

mark --exit-code "$rc"
exit "$rc"
