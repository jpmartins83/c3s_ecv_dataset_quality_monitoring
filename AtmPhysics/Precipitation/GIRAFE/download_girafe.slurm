#!/bin/bash
#SBATCH --job-name=girafe
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=girafe_%j.out

# run 'sbatch download_girafe.slurm' in the command line
module load conda
conda activate work

echo "Running on $(hostname)"
pwd
ls -l
# python3 /etc/ecmwf/nfs/dh1_home_a/cxjo/C3S_stuff/c3s_ecv_dataset_quality_monitoring/AtmPhysics/Precipitation/download_girafe_daily.py
python3 download_girafe_daily.py