#!/bin/bash
#SBATCH --job-name=uth_stats
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=uth_stats_%j.out

module load conda
conda activate work

python3 compute_stats.py --dstart 1994-07-01 --dend 2018-12-31 --do-spatial