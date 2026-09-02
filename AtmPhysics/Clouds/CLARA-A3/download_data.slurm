#!/bin/bash
#SBATCH --job-name=clara_clouds_dl
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=clara_clouds_dl_%j.out

# Usage:  sbatch download_data.slurm <frequency> [product ...]
#
# One job per product is how this is parallelised: the products are independent
# and each is a separate CDS request, so
#
#   for P in CFC CPH CTO LWP IWP JCH; do sbatch download_data.slurm monthly_mean $P; done
#   for P in CFC CPH CTO LWP IWP;     do sbatch download_data.slurm daily_mean   $P; done
#
# runs six (then five) downloads side by side. With no product given, the job
# walks every product of the frequency in turn.
#
# Monthly means are requested a year at a time (47 requests per product, ~60 GB
# for the whole record). Daily means are requested a month at a time, and are
# very much larger: roughly 120 GB for CFC or CPH and several hundred GB for
# CTO, LWP and IWP, so about 1.5 TB for the five daily products together.
#
# The download is resumable: a block whose files are all on disk is skipped, so
# a job that runs out of time can simply be resubmitted.

module load conda
conda activate work

FREQ=${1:-monthly_mean}
shift 1 2>/dev/null || true

VARARGS=""
for p in "$@"; do VARARGS="$VARARGS --variable $p"; done

echo "Running on $(hostname): $FREQ ${*:-all products}"

python -u download_data.py --frequency "$FREQ" --dstart 1979-01-01 --dend 2026-12-31 $VARARGS
