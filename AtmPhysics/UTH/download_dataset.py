"""Download the daily UTH edition-2 record from the Climate Data Store."""

import argparse
import logging
from pathlib import Path
import random
import re
import shutil
import time
import zipfile

import cdsapi
import pandas as pd


DATASET = "satellite-upper-troposphere-humidity"
VARIABLE = "mean_uth"
VERSION = "v2"
OUTPUT_DIR = Path("../../../datasets/UTH/daily")
MAX_RETRIES = 8
DATE_PATTERN = re.compile(r"(\d{8})")


def place_extracted_files(staging_dir):
    """Move files into daily/<year>/<month>, based on their date token."""
    moved = 0
    for path in staging_dir.rglob("*.nc"):
        match = DATE_PATTERN.search(path.name)
        if match is None:
            logging.warning("Unexpected filename, left in place: %s", path.name)
            continue
        date = pd.to_datetime(match.group(1), format="%Y%m%d")
        destination = OUTPUT_DIR / f"{date.year}" / f"{date.month:02d}"
        destination.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(destination / path.name))
        moved += 1
    return moved


def dates_on_disk(dates):
    """Return the requested dates that are already extracted."""
    found = set()
    for date in dates:
        directory = OUTPUT_DIR / f"{date.year}" / f"{date.month:02d}"
        if not directory.is_dir():
            continue
        for path in directory.glob("*.nc"):
            match = DATE_PATTERN.search(path.name)
            if match is not None:
                found.add(match.group(1))
    return {date.strftime("%Y%m%d") for date in dates} <= found


def retrieve(client, request, zip_file, label):
    for attempt in range(MAX_RETRIES):
        try:
            logging.info("%s: downloading...", label)
            client.retrieve(DATASET, request, str(zip_file))
            return True
        except Exception as exception:
            wait = min(600, 30 * 2 ** attempt) + random.randint(0, 30)
            logging.warning("%s: attempt %s failed (%s); retrying in %s s.",
                            label, attempt + 1, exception, wait)
            time.sleep(wait)
    logging.error("%s: failed after %s attempts.", label, MAX_RETRIES)
    return False


def main():
    parser = argparse.ArgumentParser(description="Download daily UTH edition-2 data")
    parser.add_argument("--dstart", default="1994-07-01")
    parser.add_argument("--dend", default="2018-12-31")
    args = parser.parse_args()

    dstart = max(pd.Timestamp(args.dstart), pd.Timestamp("1994-07-01"))
    dend = min(pd.Timestamp(args.dend), pd.Timestamp("2018-12-31"))
    if dstart > dend:
        raise SystemExit("Requested range does not overlap the UTH edition-2 record.")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open("../../cds_key_dev.txt") as key_file:
        key = key_file.read().strip()
    client = cdsapi.Client(
        url="https://cds-dev-cci2.copernicus-climate.eu/api",
        key=key,
        progress=True,
        timeout=3600,
    )

    for month in pd.date_range(dstart.replace(day=1), dend.replace(day=1), freq="MS"):
        block_start = max(month, dstart)
        block_end = min(month + pd.offsets.MonthEnd(), dend)
        dates = pd.date_range(block_start, block_end, freq="D")
        label = f"UTH {month:%Y-%m}"
        zip_file = OUTPUT_DIR / f"UTH_{month:%Y%m}.zip"
        staging_dir = OUTPUT_DIR / f".staging_{month:%Y%m}"

        if dates_on_disk(dates):
            logging.info("%s: already extracted, skipping.", label)
            continue

        if not zip_file.exists():
            request = {
                "variable": [VARIABLE],
                "version": [VERSION],
                "year": [f"{month.year}"],
                "month": [f"{month.month:02d}"],
                "day": [f"{day:02d}" for day in dates.day],
            }
            if not retrieve(client, request, zip_file, label):
                continue

        try:
            logging.info("%s: extracting...", label)
            shutil.rmtree(staging_dir, ignore_errors=True)
            staging_dir.mkdir(parents=True)
            with zipfile.ZipFile(zip_file) as archive:
                archive.extractall(staging_dir)
            moved = place_extracted_files(staging_dir)
            shutil.rmtree(staging_dir)
            zip_file.unlink()
            logging.info("%s: extraction complete, %s file(s).", label, moved)
        except Exception as exception:
            logging.error("%s: extraction failed (%s)", label, exception)
            shutil.rmtree(staging_dir, ignore_errors=True)
            zip_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()