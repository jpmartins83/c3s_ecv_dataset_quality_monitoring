#!/bin/bash
#SBATCH --job-name=CLARA_SRB
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=CLARA_SRB_%j.out

# Usage:  sbatch compute_stats.slurm <frequency> <variable> [spatial] [p999] [notseries]
#
# Frequency and variable are arguments so the combinations can run as parallel
# jobs. The site sbatch wrapper parses its own arguments and rejects "--flags"
# placed after the script, so the optional switches are passed as bare keywords:
#   spatial    -> --do-spatial   (recompute the spatial consistency, slow)
#   p999       -> --do-p999      (recompute the climatological thresholds)
#   notseries  -> --no-tseries   (skip the time series)
#   stride=N   -> --clim-stride N (pool every Nth file into the climatology)
#
# The climatological P0.1/P99.9 thresholds are derived from the monthly means and
# shared by both frequencies, so run the monthly job of a variable first, or make
# the daily one depend on it (-d afterok:<jobid>).
#
# The time series uses one worker process per reserved CPU, so --cpus-per-task
# sets the parallelism. Throughput saturates around 4-8 workers because reading
# the files from NFS, not the arithmetic, is the limit.

module load conda
conda activate work

FREQ=${1:-monthly_mean}
VAR=${2}
shift 2 2>/dev/null || true

if [ -z "$VAR" ]; then echo "usage: sbatch compute_stats.slurm <frequency> <variable> [spatial] [p999] [notseries]"; exit 1; fi

EXTRA=""
for arg in "$@"; do
    case "$arg" in
        spatial)   EXTRA="$EXTRA --do-spatial" ;;
        p999)      EXTRA="$EXTRA --do-p999" ;;
        notseries) EXTRA="$EXTRA --no-tseries" ;;
        stride=*)  EXTRA="$EXTRA --clim-stride ${arg#stride=}" ;;
        *) echo "Unknown option '$arg' (expected: spatial, p999, notseries, stride=N)"; exit 1 ;;
    esac
done

if [ "$FREQ" = "daily_mean" ]; then DEND=20260820; else DEND=20260730; fi

echo "Running on $(hostname): $FREQ $VAR (1979-01-01 to $DEND) options:$EXTRA"

python -u compute_stats.py --frequency $FREQ --dstart 19790101 --dend $DEND --variable $VAR $EXTRA
