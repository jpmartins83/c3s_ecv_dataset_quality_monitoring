from pathlib import Path
import cdsapi
import logging
import random
import time
import zipfile
import calendar
import argparse
import pandas as pd

# =============================================================================
# Configuration
# =============================================================================

# parse command line arguments
parser = argparse.ArgumentParser(description="Compute statistics for a dataset")
parser.add_argument("--frequency", type=str, default="daily_mean", help="Frequency of the data (daily or monthly)")
parser.add_argument("--dstart", type=str, default="1979-01-01", help="Start date for the data")
parser.add_argument("--dend", type=str, default="2026-12-31", help="End date for the data")
args = parser.parse_args()
FREQUENCY = args.frequency
dstart = args.dstart
dend = args.dend

DATASET = "satellite-surface-radiation-budget"
PRODUCT = "CLARA-A3"

if FREQUENCY == "monthly_mean":
    VARIABLE = [
        "surface_downwelling_shortwave_flux",
        "surface_downwelling_longwave_flux",
        "surface_net_downward_shortwave_flux",
        "surface_net_downward_longwave_flux",
        "surface_net_downward_radiative_flux"
    ]
elif FREQUENCY == "daily_mean": 
    VARIABLE = [
    "surface_downwelling_shortwave_flux",
]

cdr_start_year = 1979
cdr_end_year   = 2020
dates = pd.date_range(start=dstart, end=dend, freq='MS')
YEARS = dates.year.unique()
OUTPUT_DIR = Path(f"../../../datasets/SRB/{PRODUCT}/{FREQUENCY}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_RETRIES = 8

# read cds key from aux file
fkey = "../../../cds_key_dev.txt"
with open(fkey, 'r') as f:
    key = f.read().strip()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

client = cdsapi.Client(
    url='https://cds-dev-cci2.copernicus-climate.eu/api', 
    key=key,
    progress=True,
    quiet=False,
    timeout=3600
)

for year in YEARS:

    yyyy = str(year)
    if year <= cdr_end_year:
        CDR_TYPE = "thematic_climate_data_record"
    else:
        CDR_TYPE = "interim_climate_data_record"

    for month in range(1, 13):
        month = f"{month:02d}"
        n_days = calendar.monthrange(int(year), int(month))[1]

        zip_file = OUTPUT_DIR / f"{PRODUCT}_{FREQUENCY}_{year}_{month}.zip"
        extract_dir = OUTPUT_DIR / f"{year}/{month}"
        print(extract_dir)

        # Skip if already extracted
        if extract_dir.exists():
            logging.info(f"{year}: already extracted, skipping.")
            continue

        # Download if needed
        if not zip_file.exists():

            request = {
                # "day": [day],
                "year": [year],
                "month": [month],
                "origin": ["eumetsat"],
                "variable": VARIABLE,
                "time_aggregation": [FREQUENCY],
                "product_family": ["clara_a3"],
                "day": [f"{d:02d}" for d in range(1, n_days + 1)],
                "climate_data_record_type": [CDR_TYPE],
            }

            success = False

            for attempt in range(MAX_RETRIES):

                try:

                    logging.info(f"{year}: downloading...")

                    client.retrieve(
                        DATASET,
                        request,
                        str(zip_file)
                    )

                    success = True
                    break

                except Exception as e:

                    wait = min(600, 30 * (2 ** attempt))
                    wait += random.randint(0, 30)

                    logging.warning(
                        f"{year}: attempt {attempt+1} failed ({e}) "
                        f"Retrying in {wait} s."
                    )

                    time.sleep(wait)

            if not success:
                logging.error(f"{year}: failed after {MAX_RETRIES} attempts.")
                continue

        # Extract
        try:

            logging.info(f"{year}: extracting...")

            extract_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(zip_file, "r") as zf:
                zf.extractall(extract_dir)

            zip_file.unlink()

            logging.info(f"{year}: extraction complete.")

        except Exception as e:

            logging.error(f"{year}: extraction failed ({e})")