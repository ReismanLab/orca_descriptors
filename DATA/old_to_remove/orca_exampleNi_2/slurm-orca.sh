#!/usr/bin/bash
#SBATCH -n 16
#SBATCH -N 1
#SBATCH --time=7-0:00:00
#SBATCH -o output_%j.txt
#SBATCH -e error_%j.txt
#SBATCH --signal=B:TERM@600

# Jobname below is set automatically when submitting like this: sbatch -J jobname job-orca.sh

job=${SLURM_JOB_NAME}
job=$(echo ${job%%.*})

module load orca/6.1.0
module load python

#  orca uses environment variables to find nbo
#  even though they have exe extensions they are linux binaries
export NBOBIN=/resnick/groups/ReismanLab/nbo7/bin
export NBOEXE=/resnick/groups/ReismanLab/nbo7/bin/nbo7.i8.exe
export GENEXE=/resnick/groups/ReismanLab/nbo7/bin/gennbo.i8.exe
export PATH="$PATH:$NBOBIN"

$(which orca) $job.inp >> $job.out