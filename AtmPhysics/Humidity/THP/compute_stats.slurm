#!/bin/bash
#SBATCH --job-name=THP_stats
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=THP_stats_%j.out

# Usage:  sbatch compute_stats.slurm [release] [source] [variable(s)] [p999]
#
# With no arguments it recomputes every aux file for every release on disk,
# which takes a couple of minutes: the product is a monthly zonal mean, so the
# whole 19-year record is ~100 MB and is read in one pass. There is no
# incremental mode and none is needed -- when new months arrive, download them
# and run this again.
#
#   release     v1 (default: every release configured in compute_stats.py)
#   source      RO | MOD | DIFF | RDIFF (default: all four)
#                 RO    the radio-occultation retrievals
#                 MOD   the reanalysis collocated at the occultation locations
#                 DIFF  RO minus MOD, in the variable's own units
#                 RDIFF RO minus MOD, in per cent of MOD
#   variable(s) comma-separated, from
#                 Q          monthly mean humidity              [g/kg]
#                 Q_stdev    standard deviation of humidity     [g/kg]
#                 Q_obssig   measurement uncertainty of the mean[g/kg]
#                 Q_samperr  sampling error of the mean         [g/kg]
#                 Q_num      monthly data number                [1]
#                 WQ         a priori fraction                  [percent]
#               (default: all six. DIFF and RDIFF only carry Q and Q_stdev --
#               the other four are not independent between the two product
#               types; see VARIABLES in compute_stats.py.)
#   p999        re-derive the climatological P0.1/P99.9 thresholds
#
# Examples
# --------
#   sbatch compute_stats.slurm                     # everything
#   sbatch compute_stats.slurm v1 RO Q
#   sbatch compute_stats.slurm v1 DIFF Q,Q_stdev p999
#
# What it writes into aux_files/, per release, source and variable:
#
#   zonal_record_<VAR>_<SOURCE>_<REL>_<dstart>_<dend>.nc       the record as a cube
#   tseries_stats_<VAR>_<SOURCE>_<REL>_<dstart>_<dend>.nc      the QC time series
#   zonal_consistency_<VAR>_<SOURCE>_<REL>_<dstart>_<dend>.nc  per-cell aggregates
#   <VAR>_p999_<SOURCE>_<REL>_20061201_20161201.nc             outlier thresholds
#
# The thresholds are derived from the reprocessed part of the record (the CDR,
# December 2006 to December 2016) and are reused once written, so the interim
# record is judged against a fixed reference rather than against itself. Pass
# p999 to force them to be derived again, which is for a new release or a new
# reference period, not for months being appended.
#
# Afterwards, rebuild the gallery with
#
#   python build_QC_gallery.py

module load conda
conda activate work

RELEASE=${1:-}
SOURCE=${2:-}
VARS=${3:-}
shift 3 2>/dev/null || true

EXTRA=""
[ -n "$RELEASE" ] && EXTRA="$EXTRA --release $RELEASE"
[ -n "$SOURCE" ]  && EXTRA="$EXTRA --source $SOURCE"
[ -n "$VARS" ]    && EXTRA="$EXTRA --variable $VARS"

for arg in "$@"; do
    case "$arg" in
        p999) EXTRA="$EXTRA --do-p999" ;;
        *) echo "Unknown option '$arg' (expected: p999)"; exit 1 ;;
    esac
done

echo "Running on $(hostname): compute_stats.py$EXTRA"

python -u compute_stats.py $EXTRA
