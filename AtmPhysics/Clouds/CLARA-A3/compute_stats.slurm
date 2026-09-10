#!/bin/bash
#SBATCH --job-name=CLARA_Clouds
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=CLARA_Clouds_%j.out

# Usage:  sbatch compute_stats.slurm <frequency> <variable(s)> [spatial] [p999] [notseries] [stride=N]
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
# Quantities that share a product file are worth computing together: reading the
# file from NFS, not the arithmetic, is the cost, so one job over the CTO group
# costs what one CTO quantity used to and writes all three sets of aux files.
# all_<PREFIX> names the whole group held in that file,
#
#   all_CFC  ->  CFC                      all_LWP  ->  LWP,COT_liq,CRE_liq
#   all_CPH  ->  CPH                      all_IWP  ->  IWP,COT_ice,CRE_ice
#   all_CTO  ->  CTT,CTH,CTP
#
# and an explicit comma-separated list works too, as long as the quantities come
# from one file:
#
#   sbatch compute_stats.slurm daily_mean all_CTO
#   sbatch compute_stats.slurm daily_mean CTT,CTP
#
# The climatological P0.1/P99.9 thresholds are derived from, and named after,
# the frequency they are compared against, so each (frequency, variable) pair
# computes its own on first use. A daily climatology pools ~1300 files per
# calendar month, i.e. ~1.3e9 values; they are read in parallel and reduced to
# the outer 0.1% of the distribution as they go, so the memory below covers it
# and stride= is only worth it to shorten the read:
#
#   sbatch compute_stats.slurm daily_mean CFC p999
#
# The time series uses one worker process per reserved CPU, so --cpus-per-task
# sets the parallelism. Throughput saturates around 4-8 workers because reading
# the files from NFS, not the arithmetic, is the limit.

# sbatch compute_stats.slurm daily_mean all_CPH   # 1 quantity
# sbatch compute_stats.slurm daily_mean all_CTO   # 3
# sbatch compute_stats.slurm daily_mean all_LWP   # 3
# sbatch compute_stats.slurm daily_mean all_IWP   # 3

module load conda
conda activate work

FREQ=${1:-monthly_mean}
VAR=${2}
shift 2 2>/dev/null || true

if [ -z "$VAR" ]; then echo "usage: sbatch compute_stats.slurm <frequency> <variable(s)> [spatial] [p999] [notseries] [stride=N]"; exit 1; fi

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

# One group name expands to every quantity held in that product file; anything
# else is passed through, so an explicit comma-separated list still works.
case "$VAR" in
    all_CFC) VARS=CFC ;;
    all_CPH) VARS=CPH ;;
    all_CTO) VARS=CTT,CTH,CTP ;;
    all_LWP) VARS=LWP,COT_liq,CRE_liq ;;
    all_IWP) VARS=IWP,COT_ice,CRE_ice ;;
    *)       VARS=$VAR ;;
esac

# Last date each product is delivered for, from the dataset constraints as of
# 2026-09-01: the cloud physical properties stop five months short of the cloud
# mask, phase and cloud top products, and the final daily month of the latter
# holds a single day. Bump these when a new ICDR release lands.
#
# Checked for every requested quantity rather than just the first: the ones in a
# single file do share a delivery, but a hand-written list that crosses files
# would otherwise silently take the end date of whichever came first and cut the
# other short. Which quantities may legitimately share a run is settled by
# compute_stats.py.
LAST_DAILY=""
LAST_MONTHLY=""
for v in ${VARS//,/ }; do
    case "$v" in
        LWP|IWP|COT_liq|COT_ice|CRE_liq|CRE_ice) d=20250531; m=20250501 ;;
        CFC|CPH|CTT|CTH|CTP)                     d=20251001; m=20251001 ;;
        *) echo "Unknown variable '$v' (see the list at the top of this script)"; exit 1 ;;
    esac
    if [ -n "$LAST_DAILY" ] && [ "$LAST_DAILY" != "$d" ]; then
        echo "'$VARS' mixes quantities that are not delivered up to the same date; submit them as separate jobs"
        exit 1
    fi
    LAST_DAILY=$d
    LAST_MONTHLY=$m
done

if [ "$FREQ" = "daily_mean" ]; then DEND=$LAST_DAILY; else DEND=$LAST_MONTHLY; fi

echo "Running on $(hostname): $FREQ $VARS (1979-01-01 to $DEND) options:$EXTRA"

python -u compute_stats.py --frequency $FREQ --dstart 19790101 --dend $DEND --variable $VARS $EXTRA
