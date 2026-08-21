#!/bin/bash
#SBATCH --job-name=girafe
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=girafe_%j.out

# run 'sbatch compute_girafe_stats.slurm' in the command line
module load conda
conda activate work

# python3 /etc/ecmwf/nfs/dh1_home_a/cxjo/C3S_stuff/c3s_ecv_dataset_quality_monitoring/AtmPhysics/Precipitation/download_girafe_daily.py
# python3 compute_GIRAFE_stats_daily.py monthly
# python3 compute_GIRAFE_stats.py daily
python compute_GIRAFE_stats.py --frequency 'monthly' --dstart 20020101 --dend 20260630
python compute_GIRAFE_stats.py --frequency 'daily' --dstart 20020101 --dend 20260630