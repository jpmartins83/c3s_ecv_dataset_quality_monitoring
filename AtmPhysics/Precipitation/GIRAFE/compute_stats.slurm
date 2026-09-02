#!/bin/bash
#SBATCH --job-name=GIRAFE
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=GIRAFE_%j.out

# Usage:  sbatch compute_stats.slurm <frequency> [spatial] [p999] [notseries]
#
#   sbatch compute_stats.slurm monthly spatial
#   sbatch compute_stats.slurm daily            # time series only (the usual case)
#
# The site sbatch wrapper parses its own arguments and rejects "--flags" placed
# after the script, so the optional switches are passed as bare keywords:
#   spatial    -> --do-spatial   (recompute the spatial consistency, slow)
#   p999       -> --do-p999      (recompute the climatological thresholds)
#   notseries  -> --no-tseries   (skip the time series)
#   stride=N   -> --clim-stride N (pool every Nth file into the climatology)
#
# The climatological P0.1/P99.9 thresholds are derived from the monthly means and
# shared by both frequencies, so run the monthly job first, or make the daily one
# depend on it (-d afterok:<jobid>), to avoid two jobs writing them at once.
#
# The time series uses one worker process per reserved CPU, so --cpus-per-task
# sets the parallelism. Throughput saturates around 4-8 workers because reading
# the files from NFS, not the arithmetic, is the limit.

module load conda
conda activate work

FREQ=${1:-daily}
shift 1 2>/dev/null || true

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

echo "Running on $(hostname): $FREQ (2002-01-01 to 2026-06-30) options:$EXTRA"

python -u compute_stats.py --frequency $FREQ --dstart 20020101 --dend 20260630 $EXTRA
