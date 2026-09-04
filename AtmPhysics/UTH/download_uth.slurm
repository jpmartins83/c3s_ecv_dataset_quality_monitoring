#!/bin/bash
#SBATCH --job-name=uth_download
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=uth_download_%j.out

module load conda
conda activate work

python3 download_dataset.py --dstart 1994-07-01 --dend 2018-12-31