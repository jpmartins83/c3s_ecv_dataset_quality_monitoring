#!/bin/bash
#SBATCH --job-name=CLARA
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=CLARA_%j.out

# run 'sbatch compute_stats.slurm' in the command line
module load conda
conda activate work
# python compute_stats.py --frequency monthly_mean --dstart 19790101 --dend 20260730 --variable surface_downwelling_shortwave_flux
python compute_stats.py --frequency monthly_mean --dstart 19790101 --dend 20260730 --variable surface_downwelling_longwave_flux --do-spatial
# python compute_stats.py --frequency monthly_mean --dstart 19790101 --dend 20260730 --variable surface_net_downward_shortwave_flux
# python compute_stats.py --frequency monthly_mean --dstart 19790101 --dend 20260730 --variable surface_net_downward_longwave_flux
# python compute_stats.py --frequency monthly_mean --dstart 19790101 --dend 20260730 --variable surface_net_downward_radiative_flux
# python compute_stats.py --frequency daily_mean --dstart 19790101 --dend 20260731 --variable surface_downwelling_shortwave_flux