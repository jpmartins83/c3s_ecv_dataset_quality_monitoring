from pathlib import Path
import argparse
import calendar
import logging
import random
import re
import shutil
import time
import zipfile

import cdsapi
import pandas as pd

# =============================================================================
# Configuration
# =============================================================================

DATASET = "satellite-cloud-properties"
PRODUCT = "CLARA-A3"
ECV = "Clouds"
PRODUCT_FAMILY = "clara_a3"
ORIGIN = "eumetsat"

# Each CDS variable is delivered as its own file, whose name starts with the
# three-letter product token below. That token is what the rest of the workflow
# (compute_stats.py, build_QC_gallery.py) keys on.
VARIABLES = {
    "cloud_fraction": "CFC",
    "cloud_phase": "CPH",
    "cloud_top_level": "CTO",
    "cloud_physical_properties_of_the_liquid_phase": "LWP",
    "cloud_physical_properties_of_the_ice_phase": "IWP",
    "joint_cloud_property_histogram": "JCH",
}

# The joint histogram is only produced as monthly means (and on a 1x1 degree
# grid rather than 0.25 degree), so it is not part of the daily request.
FREQUENCY_VARIABLES = {
    "monthly_mean": list(VARIABLES),
    "daily_mean": [v for v in VARIABLES if v != "joint_cloud_property_histogram"],
}

# Last date each variable is available for, per frequency. Read from the dataset
# constraints (.../api/catalogue/v1/collections/satellite-cloud-properties/
# constraints.json) on 2026-09-01: the cloud mask, phase and cloud top products
# run five months further than the physical properties, and the final ICDR month
# can be partial (October 2025 holds a single day). Asking for dates beyond
# these fails the constraint check, so the loop stops at them instead. Extend
# these as new ICDR releases appear.
AVAILABILITY_END = {
    "cloud_fraction": {"monthly_mean": "2025-10-01", "daily_mean": "2025-10-01"},
    "cloud_phase": {"monthly_mean": "2025-10-01", "daily_mean": "2025-10-01"},
    "cloud_top_level": {"monthly_mean": "2025-10-01", "daily_mean": "2025-10-01"},
    "cloud_physical_properties_of_the_liquid_phase":
        {"monthly_mean": "2025-05-01", "daily_mean": "2025-05-31"},
    "cloud_physical_properties_of_the_ice_phase":
        {"monthly_mean": "2025-05-01", "daily_mean": "2025-05-31"},
    "joint_cloud_property_histogram": {"monthly_mean": "2025-05-01"},
}

CDR_END_YEAR = 2020  # 1979-2020 is the TCDR, later years are the ICDR

# The token that follows the product name in a filename. The joint histogram
# uses 'mh' where every other monthly product uses 'mm'.
FREQ_TOKENS = {"monthly_mean": "mm", "daily_mean": "dm"}
FILE_TOKENS = {("JCH", "monthly_mean"): "mh"}

# <PRODUCT><token><YYYYMMDD>000000<platform>AVPOS<01|I1>GL.nc — the tail varies
# with the platform and with TCDR/ICDR, so files are matched, never constructed.
FILE_PATTERN = re.compile(r"(?P<prefix>[A-Z]{3})(?P<token>dm|mm|mh)(?P<date>\d{8})")

MAX_RETRIES = 8

# =============================================================================


def file_token(prefix, frequency):
    return FILE_TOKENS.get((prefix, frequency), FREQ_TOKENS[frequency])


def resolve_variables(requested, frequency):
    """Turn the --variable arguments into CDS variable names.

    Both the CDS name and the short product token are accepted, so
    '--variable CFC' and '--variable cloud_fraction' mean the same thing.
    """
    available = FREQUENCY_VARIABLES[frequency]
    if not requested:
        return available

    by_token = {token: name for name, token in VARIABLES.items()}
    resolved = []
    for item in requested:
        for name in (s.strip() for s in item.split(",") if s.strip()):
            key = by_token.get(name.upper(), name)
            if key not in VARIABLES:
                raise SystemExit(
                    f"Unknown variable '{name}'. Choose from: "
                    + ", ".join(f"{t} ({n})" for n, t in VARIABLES.items())
                )
            if key not in available:
                raise SystemExit(
                    f"{VARIABLES[key]} ({key}) is not delivered as {frequency}."
                )
            if key not in resolved:
                resolved.append(key)
    return resolved


def month_dir(output_dir, date):
    return output_dir / f"{date.year}" / f"{date.month:02d}"


def dates_on_disk(output_dir, prefix, token, dates):
    """The subset of dates whose file is already extracted."""
    found = set()
    for year, month in sorted({(d.year, d.month) for d in dates}):
        directory = output_dir / f"{year}" / f"{month:02d}"
        if not directory.is_dir():
            continue
        for path in directory.glob(f"{prefix}{token}*.nc"):
            match = FILE_PATTERN.match(path.name)
            if match and match.group("prefix") == prefix:
                found.add(match.group("date"))
    return {d for d in dates if d.strftime("%Y%m%d") in found}


def place_extracted_files(staging_dir, output_dir):
    """Move the NetCDF files of one download into their <year>/<month> folder.

    A monthly request covers a whole year, so one zip carries files for up to
    twelve different months; the destination is taken from the date in each
    filename rather than from the request.
    """
    moved = 0
    for path in sorted(staging_dir.rglob("*.nc")):
        match = FILE_PATTERN.match(path.name)
        if match is None:
            logging.warning(f"Unexpected filename, left in place: {path.name}")
            continue
        date = pd.to_datetime(match.group("date"), format="%Y%m%d")
        destination = month_dir(output_dir, date)
        destination.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(destination / path.name))
        moved += 1
    return moved


def retrieve(client, request, zip_file, label):
    """One CDS request, retried with exponential backoff. True if it landed."""
    for attempt in range(MAX_RETRIES):
        try:
            logging.info(f"{label}: downloading...")
            client.retrieve(DATASET, request, str(zip_file))
            return True
        except Exception as exception:
            wait = min(600, 30 * (2 ** attempt)) + random.randint(0, 30)
            logging.warning(
                f"{label}: attempt {attempt + 1} failed ({exception}) "
                f"Retrying in {wait} s."
            )
            time.sleep(wait)
    logging.error(f"{label}: failed after {MAX_RETRIES} attempts.")
    return False


def requests_for(cds_variable, frequency, dstart, dend):
    """The (label, tag, dates, request) blocks needed to cover one variable.

    Monthly means are requested a year at a time, which keeps the number of CDS
    requests down; daily means are requested a month at a time, because a single
    month of one product is already around a gigabyte.
    """
    end = pd.Timestamp(AVAILABILITY_END[cds_variable][frequency])
    if dend > end:
        logging.info(
            f"{VARIABLES[cds_variable]}: {frequency} ends {end:%Y-%m-%d}, "
            f"clipping the requested end date {dend:%Y-%m-%d}."
        )
    end = min(dend, end)
    if dstart > end:
        logging.warning(
            f"{VARIABLES[cds_variable]}: nothing to do, {dstart:%Y-%m-%d} is "
            f"after the last available date {end:%Y-%m-%d}."
        )
        return []

    resolution = "MS" if frequency == "monthly_mean" else "D"
    dates = pd.date_range(dstart, end, freq=resolution)
    if dates.empty:
        return []

    # Monthly: one block per year. Daily: one block per calendar month.
    if frequency == "monthly_mean":
        keys = sorted({(d.year,) for d in dates})
    else:
        keys = sorted({(d.year, d.month) for d in dates})

    blocks = []
    for key in keys:
        group = pd.DatetimeIndex(
            [d for d in dates if (d.year,) == key or (d.year, d.month) == key])
        year = key[0]
        tag = f"{year}" if len(key) == 1 else f"{year}-{key[1]:02d}"
        request = {
            "product_family": [PRODUCT_FAMILY],
            "origin": [ORIGIN],
            "variable": [cds_variable],
            "time_aggregation": [frequency],
            "climate_data_record_type": [
                "thematic_climate_data_record" if year <= CDR_END_YEAR
                else "interim_climate_data_record"
            ],
            "year": [str(year)],
            "month": sorted({f"{m:02d}" for m in group.month}),
            # Monthly means are indexed by the first of the month only.
            "day": ["01"] if frequency == "monthly_mean"
                   else [f"{d:02d}" for d in group.day],
        }
        blocks.append((f"{VARIABLES[cds_variable]} {tag}", tag, group, request))
    return blocks


def main():
    parser = argparse.ArgumentParser(
        description=f"Download {PRODUCT} cloud properties from the CDS")
    parser.add_argument("--frequency", type=str, default="monthly_mean",
                        choices=list(FREQUENCY_VARIABLES),
                        help="Frequency of the data (daily_mean or monthly_mean)")
    parser.add_argument("--dstart", type=str, default="1979-01-01",
                        help="Start date for the data")
    parser.add_argument("--dend", type=str, default="2026-12-31",
                        help="End date for the data (clipped to what is available)")
    parser.add_argument("--variable", action="append", default=[],
                        help="Product to download, by token (CFC) or CDS name "
                             "(cloud_fraction). Repeatable or comma-separated; "
                             "the default is every product of the frequency. One "
                             "job per product is the way to parallelise this.")
    args = parser.parse_args()

    frequency = args.frequency
    dstart = pd.to_datetime(args.dstart)
    dend = pd.to_datetime(args.dend)
    variables = resolve_variables(args.variable, frequency)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    output_dir = Path(f"../../../datasets/{ECV}/{PRODUCT}/{frequency}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open("../../../cds_key_dev.txt", "r") as f:
        key = f.read().strip()

    client = cdsapi.Client(
        url="https://cds-dev-cci2.copernicus-climate.eu/api",
        key=key,
        progress=True,
        quiet=False,
        timeout=3600,
    )

    logging.info(
        f"{frequency} {dstart:%Y-%m-%d} to {dend:%Y-%m-%d}: "
        + ", ".join(VARIABLES[v] for v in variables)
    )

    for cds_variable in variables:
        prefix = VARIABLES[cds_variable]
        token = file_token(prefix, frequency)

        for label, tag, dates, request in requests_for(
                cds_variable, frequency, dstart, dend):

            # Only the dates that are not on disk yet, so an interrupted run can
            # be restarted without downloading everything again. The check is per
            # product: a folder holding the other products is not evidence that
            # this one has been extracted.
            present = dates_on_disk(output_dir, prefix, token, dates)
            if len(present) == len(dates):
                logging.info(f"{label}: already extracted ({len(dates)} files), skipping.")
                continue
            if present:
                logging.info(
                    f"{label}: {len(present)} of {len(dates)} files present, "
                    f"re-requesting the whole block."
                )

            staging_dir = output_dir / f".staging_{prefix}_{tag}"
            zip_file = output_dir / f"{prefix}_{frequency}_{tag}.zip"

            if not zip_file.exists():
                if not retrieve(client, request, zip_file, label):
                    continue

            try:
                logging.info(f"{label}: extracting...")
                if staging_dir.exists():
                    shutil.rmtree(staging_dir)
                staging_dir.mkdir(parents=True)
                with zipfile.ZipFile(zip_file, "r") as zf:
                    zf.extractall(staging_dir)
                moved = place_extracted_files(staging_dir, output_dir)
                shutil.rmtree(staging_dir, ignore_errors=True)
                zip_file.unlink()
                logging.info(f"{label}: extraction complete, {moved} file(s).")
                if moved != len(dates):
                    logging.warning(
                        f"{label}: expected {len(dates)} file(s) but the download "
                        f"held {moved}."
                    )
            except Exception as exception:
                logging.error(f"{label}: extraction failed ({exception})")
                # A download that was interrupted leaves a truncated zip behind,
                # and keeping it would make every later run try to unpack the
                # same broken file instead of asking for the block again.
                shutil.rmtree(staging_dir, ignore_errors=True)
                zip_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
