#!/bin/bash
#SBATCH --job-name=CLARA_ERB
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=CLARA_ERB_%j.out

# Frequency and variable are taken as arguments so that the four combinations can
# run as parallel jobs:
#
#   sbatch compute_stats.slurm monthly_mean outgoing_longwave_radiation
#   sbatch compute_stats.slurm monthly_mean outgoing_shortwave_radiation
#   sbatch compute_stats.slurm daily_mean   outgoing_longwave_radiation
#   sbatch compute_stats.slurm daily_mean   outgoing_shortwave_radiation
#
# The climatological P0.1/P99.9 thresholds are derived from the monthly means and
# shared by both frequencies, so submit the monthly jobs first and make the daily
# ones depend on them (--dependency=afterok:<jobid>) to avoid two jobs computing
# the same threshold file at once.

module load conda
conda activate work

FREQ=${1:-monthly_mean}
VAR=${2:-outgoing_longwave_radiation}

if [ "$FREQ" = "daily_mean" ]; then
    DEND=20260821
else
    DEND=20260701
fi

echo "Running on $(hostname): $FREQ $VAR (1979-01-01 to $DEND)"

python compute_stats.py --frequency $FREQ --dstart 19790101 --dend $DEND --variable $VAR --do-spatial
