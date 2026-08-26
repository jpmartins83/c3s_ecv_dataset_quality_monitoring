#!/bin/bash
#SBATCH --job-name=claraa3
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=claraa3_%j.out

# run 'sbatch download_claraa3.slurm' in the command line
module load conda
conda activate work

echo "Running on $(hostname)"
# pwd
# ls -l
python3 download_ERB.py --frequency 'monthly_mean' --dstart 19790101 --dend 20261231