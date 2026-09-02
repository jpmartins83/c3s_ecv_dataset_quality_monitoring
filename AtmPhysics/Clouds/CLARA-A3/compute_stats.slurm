#!/bin/bash
#SBATCH --job-name=CLARA_Clouds
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=CLARA_Clouds_%j.out

# Usage:  sbatch compute_stats.slurm <frequency> <variable> [spatial] [p999] [notseries] [stride=N]
#
# Frequency and variable are arguments so the combinations can run as parallel
# jobs. The site sbatch wrapper parses its own arguments and rejects "--flags"
# placed after the script, so the optional switches are passed as bare keywords:
#   spatial    -> --do-spatial   (recompute the spatial consistency, slow)
#   p999       -> --do-p999      (recompute the climatological thresholds)
#   notseries  -> --no-tseries   (skip the time series)
#   stride=N   -> --clim-stride N (pool every Nth file into the climatology)
#
# The variable is a monitored quantity, not a product file: one CLARA-A3 cloud
# file holds several of them. The eleven are
#
#   CFC      cloud fraction                       [%]        from the CFC files
#   CPH      liquid cloud fraction                [%]        from the CPH files
#   CTT      cloud top temperature                [K]        from the CTO files
#   CTH      cloud top height                     [m]        from the CTO files
#   CTP      cloud top pressure                   [hPa]      from the CTO files
#   LWP      cloud liquid water path              [kg m-2]   from the LWP files
#   COT_liq  water cloud optical thickness        [1]        from the LWP files
#   CRE_liq  liquid droplet effective radius      [m]        from the LWP files
#   IWP      cloud ice water path                 [kg m-2]   from the IWP files
#   COT_ice  ice cloud optical thickness          [1]        from the IWP files
#   CRE_ice  ice particle effective radius        [m]        from the IWP files
#
# The joint cloud property histogram (JCH) has no entry: it is a 6-D field on a
# 1x1 degree grid, so it is covered by the gallery's integrity and metadata
# tables only.
#
# The climatological P0.1/P99.9 thresholds are derived from, and named after,
# the frequency they are compared against, so each (frequency, variable) pair
# computes its own on first use. A daily climatology pools ~1300 files per
# calendar month, which needs both a stride and more memory:
#
#   sbatch --mem=64G compute_stats.slurm daily_mean CFC p999 stride=4
#
# The time series uses one worker process per reserved CPU, so --cpus-per-task
# sets the parallelism. Throughput saturates around 4-8 workers because reading
# the files from NFS, not the arithmetic, is the limit.

module load conda
conda activate work

FREQ=${1:-monthly_mean}
VAR=${2}
shift 2 2>/dev/null || true

if [ -z "$VAR" ]; then echo "usage: sbatch compute_stats.slurm <frequency> <variable> [spatial] [p999] [notseries] [stride=N]"; exit 1; fi

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

# Last date each product is delivered for, from the dataset constraints as of
# 2026-09-01: the cloud physical properties stop five months short of the cloud
# mask, phase and cloud top products, and the final daily month of the latter
# holds a single day. Bump these when a new ICDR release lands.
case "$VAR" in
    LWP|IWP|COT_liq|COT_ice|CRE_liq|CRE_ice) LAST_DAILY=20250531; LAST_MONTHLY=20250501 ;;
    CFC|CPH|CTT|CTH|CTP)                     LAST_DAILY=20251001; LAST_MONTHLY=20251001 ;;
    *) echo "Unknown variable '$VAR' (see the list at the top of this script)"; exit 1 ;;
esac

if [ "$FREQ" = "daily_mean" ]; then DEND=$LAST_DAILY; else DEND=$LAST_MONTHLY; fi

echo "Running on $(hostname): $FREQ $VAR (1979-01-01 to $DEND) options:$EXTRA"

python -u compute_stats.py --frequency $FREQ --dstart 19790101 --dend $DEND --variable $VAR $EXTRA
