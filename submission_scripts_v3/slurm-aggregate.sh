#! /bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:20:00
#SBATCH --mem-per-cpu=2G
#SBATCH -o aggregate_%j.out
#SBATCH -e aggregate_%j.err

# Rebuild the batch-level metadata summary for this directory.
#
# Submitted by submit_aggregate.py at the end of launch_workflow.sh, with a
# --dependency=afterany on that ligand's ORCA jobs, so it only runs once the
# DFT for that ligand has finished. One of these is queued per ligand; each
# rebuilds the whole summary, so the last to run captures the finished batch.
#
# Output: <main_directory>/<main_directory>_metadata.json
# (aggregate_metadata.py's default when given --root .)

module load python

python aggregate_metadata.py --root .
