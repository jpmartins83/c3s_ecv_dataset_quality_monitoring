#!/bin/bash
#SBATCH --job-name=clara
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=clara_%j.out

# run 'sbatch download_clara.slurm' in the command line
module load conda
conda activate work

# echo "Running on $(hostname)"
# pwd
# ls -l
# python3 download_SRB.py --frequency 'monthly_mean' --dstart 19790101 --dend 20261231
python3 download_SRB.py --frequency 'daily_mean' --dstart 19790101 --dend 20261231