#!/bin/bash
#SBATCH --job-name=THP_download
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=THP_download_%j.out

# Usage:  sbatch download_data.slurm [release] [source] [dstart] [dend]
#
# With no arguments it downloads every release in RELEASES and both product
# types, which is the whole record: ~230 monthly files per product type, about
# 50 MB each way. One CDS request covers a calendar year, so the whole thing is
# ~40 requests and finishes in well under an hour.
#
#   release   v1 (default: every release configured in download_data.py)
#   source    RO | MOD (default: both)
#             RO  = radio_occultation_data, the retrievals
#             MOD = reanalysis_data, the reanalysis collocated at the same
#                   occultation locations
#   dstart    first month, YYYY-MM-DD (default: the release start)
#   dend      last month,  YYYY-MM-DD (default: the release end)
#
# Examples
# --------
#   sbatch download_data.slurm                          # everything
#   sbatch download_data.slurm v1 RO                    # only the retrievals
#   sbatch download_data.slurm v1 MOD 2025-01-01 2025-12-01
#
# Adding a release
# ----------------
# When version 2 appears in the Climate Data Store:
#
#   1. Read its request keys off the download form and its first/last month off
#      the constraints, both under
#      https://cds-dev-cci2.copernicus-climate.eu/api/catalogue/v1/collections/
#      satellite-humidity-profiles/{form,constraints}.json
#   2. Uncomment and fill in the "v2" entry in RELEASES in download_data.py,
#      and the matching entries in compute_stats.py and build_QC_gallery.py.
#   3. sbatch download_data.slurm v2
#
# The data lands in datasets/Humidity/THP/v2/ next to v1, and nothing else in
# the workflow needs changing: the statistics and the gallery iterate over the
# releases they find on disk, and the gallery starts showing a v2-minus-v1
# difference panel on its own.

module load conda
conda activate work

RELEASE=${1:-}
SOURCE=${2:-}
DSTART=${3:-}
DEND=${4:-}

EXTRA=""
[ -n "$RELEASE" ] && EXTRA="$EXTRA --release $RELEASE"
[ -n "$SOURCE" ]  && EXTRA="$EXTRA --source $SOURCE"
[ -n "$DSTART" ]  && EXTRA="$EXTRA --dstart $DSTART"
[ -n "$DEND" ]    && EXTRA="$EXTRA --dend $DEND"

echo "Running on $(hostname): download_data.py$EXTRA"

python -u download_data.py $EXTRA
