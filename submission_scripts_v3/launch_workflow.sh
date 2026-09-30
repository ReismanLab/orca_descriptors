#! /bin/bash
# These #SBATCH defaults are fallbacks only -- they're read by SLURM before
# this script (or workflow_inputs.yaml) is ever opened. The real values come
# from workflow_inputs.yaml's slurm.main_job block, passed as `sbatch
# --time=... --mem-per-cpu=...` flags by submit_all_smiles.py, which override
# whatever is written here.
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH -t 60:00:00
#SBATCH --mem-per-cpu=4G

# Abort on the first failing stage rather than feeding an empty directory to
# the next one. ('-u' is deliberately not set: the HPC `module` shell
# functions reference unset variables and would trip it.)
set -eo pipefail

SMILES="smiles_ligand"
# SLURM copies the batch script to /var/spool/slurmd/job<id>/slurm_script before
# running it, so ${BASH_SOURCE[0]} points into the spool, NOT at the directory
# holding record_step.py and submit_aggregate.py. SLURM_SUBMIT_DIR is where
# sbatch was invoked, which is the batch directory; fall back to the script's
# own location when this is run outside SLURM.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# ---------------------------------------------------------------------------
# Load every user-chosen pipeline parameter from workflow_inputs.yaml (or
# $WORKFLOW_INPUTS if set) via get_input.py, once, up front. See
# workflow_inputs.yaml for what each of these means and workflow_inputs.py
# for the loader itself.
# ---------------------------------------------------------------------------
get_input() { python "$SCRIPT_DIR/get_input.py" "$1"; }

OPTIMIZE=$(get_input conformer_gen.optimize)
XTB_GFN_LEVEL=$(get_input conformer_gen.xtb_gfn_level)

CREST_CHARGE=$(get_input crest.charge)
CREST_SPIN=$(get_input crest.spin_multiplicity)
CREST_EWIN=$(get_input crest.ewin_kcal_mol)
CREST_NCONF=$(get_input crest.n_conf)
CREST_QUICK=$(get_input crest.quick)
CREST_NOROTMD=$(get_input crest.norotmd)
CREST_NOREFTOPO=$(get_input crest.noreftopo)
CREST_CORES=$(get_input crest.cores)

UMA_MODEL=$(get_input uma.model_name)
UMA_FMAX=$(get_input uma.fmax)
UMA_TASK=$(get_input uma.task_name)

DFT_EWIN=$(get_input dft.ewin_kcal_mol)
DFT_CALC_PARAM_TEMPLATE=$(get_input dft.calc_param_template)

CONDA_ENV=$(get_input environment.conda_env)
OMOL_ENV_PATH=$(get_input environment.omol_env_path)

# bash-friendly booleans -> the flag or nothing, for the CLI args below
crest_flag() { [ "$1" = "true" ] && echo "$2" || echo ""; }
CREST_QUICK_FLAG=$(crest_flag "$CREST_QUICK" --quick)
CREST_NOROTMD_FLAG=$(crest_flag "$CREST_NOROTMD" --norotmd)
CREST_NOREFTOPO_FLAG=$(crest_flag "$CREST_NOREFTOPO" --noreftopo)

# ---------------------------------------------------------------------------
# run_step <name> <command...>
#
# Marks the stage "running" in <inchikey>/metadata.json, runs it, then records
# the outcome and exit code -- including for stages the shell only learns
# about after the fact, like an OOM kill (137). Aborts the workflow if the
# stage failed, having written the failure down first.
# ---------------------------------------------------------------------------
record() {
    python "$SCRIPT_DIR/record_step.py" --lig "$SMILES" "$@" \
        || echo "WARNING: could not record step status" >&2
}

run_step() {
    local name="$1"; shift
    echo "=== [$name] $* ==="
    record --step "$name" --status running
    local rc=0
    "$@" || rc=$?
    if [ "$rc" -eq 0 ]; then
        record --step "$name" --status completed --exit-code 0
    else
        record --step "$name" --status failed --exit-code "$rc"
        echo "ERROR: stage '$name' failed with exit code $rc; stopping workflow." >&2
        exit "$rc"
    fi
}

# Rebuild the batch summary however this job ends -- including when a stage
# above aborts it. A failed ligand has no pending ORCA jobs, so aggregation
# runs immediately and the failure lands in <main_directory>_metadata.json
# instead of leaving the batch looking clean.
on_exit() {
    local rc=$?
    trap - EXIT
    echo "=== [aggregate] queueing batch metadata aggregation (workflow rc=$rc) ==="
    python "$SCRIPT_DIR/submit_aggregate.py" --lig "$SMILES" \
        || echo "WARNING: could not queue metadata aggregation" >&2
    exit "$rc"
}
trap on_exit EXIT

### conda environment for crest
# `conda activate` is a shell function, not the conda binary, and a
# non-interactive sbatch shell has not sourced it -- without this line the
# activation silently fails and every stage below runs in the login
# environment (which is what happened in test3).
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# ---------------------------------------------------------------------------
# Cross-check workflow_inputs.yaml against any metadata already recorded for
# this ligand, before doing anything else. There's no TTY inside a SLURM
# batch job to ask interactively -- if submit_all_smiles.py already confirmed
# a disagreement (FORCE_INPUT_MISMATCH=1, exported via --export on sbatch),
# proceed; otherwise fail loudly here rather than silently mixing old and new
# settings. See check_inputs.py.
# ---------------------------------------------------------------------------
CHECK_ARGS=(--lig "$SMILES")
if [ "${FORCE_INPUT_MISMATCH:-0}" = "1" ]; then
    CHECK_ARGS+=(--force-non-interactive)
fi
python "$SCRIPT_DIR/check_inputs.py" "${CHECK_ARGS[@]}"

### MAKE directory from SMILES
run_step build python ligand_only_create_folder.py --lig "$SMILES" \
    --optimize "$OPTIMIZE" --xtb-gfn-level "$XTB_GFN_LEVEL"

### CREST
run_step crest python crest.py --lig "$SMILES" \
    --charge "$CREST_CHARGE" --spin-multiplicity "$CREST_SPIN" \
    --ewin "$CREST_EWIN" --n-conf "$CREST_NCONF" --cores "$CREST_CORES" \
    $CREST_QUICK_FLAG $CREST_NOROTMD_FLAG $CREST_NOREFTOPO_FLAG
conda deactivate

### UMA CALCULATION
source "$OMOL_ENV_PATH/bin/activate"
# HF_TOKEN must be exported by your shell environment / cluster secrets
# store, NOT hardcoded here -- this file (and workflow_inputs.yaml) may be
# shared or version-controlled.
: "${HF_TOKEN:?HF_TOKEN is not set -- export it before submitting, do not hardcode it here}"

module load openbabel

# pre-optimize structures using uma
run_step uma python uma_opt_confs.py --lig "$SMILES" \
    --model-name "$UMA_MODEL" --fmax "$UMA_FMAX" --task-name "$UMA_TASK"

### ORCA CALCULATION
module load orca/6.1.0

# launch the DFT computations using orca
run_step orca_submit python launch_orca_all.py --lig "$SMILES" \
    --ewin "$DFT_EWIN" --calc-param-template "$DFT_CALC_PARAM_TEMPLATE"

### AGGREGATE METADATA
# Handled by the EXIT trap above, so it runs whether or not the stages
# succeeded. It queues slurm-aggregate.sh behind this ligand's ORCA jobs;
# that rebuilds <main_directory>_metadata.json from every
# <inchikey>/metadata.json in this directory.
