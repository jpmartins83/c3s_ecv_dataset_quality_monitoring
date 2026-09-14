"""Download the Tropospheric Humidity Profiles record from the Climate Data Store.

The dataset is monthly and zonally averaged, so one file covers the globe: a
36-band latitude by 251-level altitude grid holding a single month. That makes
the whole record small (~220 kB per file, two product types, ~230 months), and
one request per year returns the twelve monthly files in a zip.

Everything that changes from one release of the dataset to the next lives in
RELEASES below. The record on disk is laid out release-first,

    datasets/Humidity/THP/<release>/<source>/<YYYY>/<file>.nc

so a second release is downloaded alongside the first rather than over it, and
the QC gallery can show them side by side.
"""

from pathlib import Path
import argparse
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

DATASET = "satellite-humidity-profiles"
PRODUCT = "THP"
ECV = "Humidity"

OUTPUT_ROOT = Path(f"../../../datasets/{ECV}/{PRODUCT}")
CDS_URL = "https://cds-dev-cci2.copernicus-climate.eu/api"
CDS_KEY_FILE = Path("../../../cds_key_dev.txt")

# The two product types are the radio-occultation retrievals and the reanalysis
# sampled at the same occultation locations. They are requested separately and
# kept in separate folders, because the QC compares them against each other.
#
# 'token' is what the delivered filename carries in place of the source name:
#   zgrid_<r|i>hgmet_<token>_<YYYYMM>_<R|I>_<processing>_<version>.nc
# The reanalysis token changes across the record -- ERA-Interim backs the CDR
# and ERA5 the ICDR -- so it is matched as a set, never constructed.
SOURCES = {
    "RO": dict(
        product_type="radio_occultation_data",
        tokens=("metop",),
        label="Radio occultation",
    ),
    "MOD": dict(
        product_type="reanalysis_data",
        tokens=("erai@metop", "era5@metop"),
        label="Collocated reanalysis",
    ),
}

# One entry per release of the dataset.
#
# 'request' holds the extra CDS keys that select the release. Version 1 needs
# none: the download form (checked 2026-09-11) offers only variable, product
# type, year and month, so the request selects the one release there is. When
# version 2 appears the form is expected to grow a version widget, exactly as
# satellite-upper-troposphere-humidity has, and the entry below is then all that
# has to be filled in.
#
# 'product_versions' is the product_version attribute the files of that release
# are expected to carry. It is not used to route the download -- the request
# decides that -- but the gallery reports any file whose attribute falls outside
# it, which is how a release served under the wrong name would show up.
#
# 'start'/'end' come from the dataset constraints
# (.../api/catalogue/v1/collections/satellite-humidity-profiles/
# constraints.json) read on 2026-09-11: the record opens with December 2006 and
# runs to December 2025. Requesting a month outside them fails the constraint
# check, so the loop stops at them instead.
RELEASES = {
    "v1": dict(
        request={},
        product_versions=("1.0", "1.1", "1.2", "1.3"),
        start="2006-12-01",
        end="2025-12-01",
        label="Version 1",
    ),
    # "v2": dict(
    #     request={"version": ["v2"]},
    #     product_versions=("2.0",),
    #     start="2006-12-01",
    #     end=None,          # None -> read the last month from the constraints
    #     label="Version 2",
    # ),
}

# <YYYYMM> is the only part of a delivered filename this workflow constructs an
# expectation about; everything else (CDR vs ICDR, processing and version
# tokens) varies across the record and is matched.
FILE_PATTERN = re.compile(
    r"zgrid_(?P<grid>[ri]hgmet)_(?P<source>[a-z0-9@]+)_(?P<date>\d{6})_"
    r"(?P<record>[RI])_(?P<processing>\d{4})_(?P<version>\d{4})\.nc"
)

MAX_RETRIES = 8

# =============================================================================


def source_dir(release, source):
    return OUTPUT_ROOT / release / source


def file_date(name):
    """The YYYYMM a delivered file covers, or None if the name is unexpected."""
    match = FILE_PATTERN.match(name)
    return match.group("date") if match else None


def months_on_disk(release, source, months):
    """The subset of `months` whose file is already extracted."""
    found = set()
    for year in sorted({m.year for m in months}):
        directory = source_dir(release, source) / f"{year}"
        if not directory.is_dir():
            continue
        for path in directory.glob("*.nc"):
            date = file_date(path.name)
            if date is not None:
                found.add(date)
    return {m.strftime("%Y%m") for m in months} & found


def place_extracted_files(release, source, staging_dir):
    """Move the extracted files into <release>/<source>/<YYYY>/."""
    moved = 0
    for path in sorted(staging_dir.rglob("*.nc")):
        date = file_date(path.name)
        if date is None:
            logging.warning("Unexpected filename, left in place: %s", path.name)
            continue
        destination = source_dir(release, source) / date[:4]
        destination.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(destination / path.name))
        moved += 1
    return moved


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


def resolve(requested, available, what):
    if not requested:
        return list(available)
    resolved = []
    for item in requested:
        for name in (s.strip() for s in item.split(",") if s.strip()):
            if name not in available:
                raise SystemExit(f"Unknown {what} '{name}'. Choose from: "
                                 + ", ".join(available))
            if name not in resolved:
                resolved.append(name)
    return resolved


def main():
    parser = argparse.ArgumentParser(
        description="Download the Tropospheric Humidity Profiles record")
    parser.add_argument("--release", action="append", default=[],
                        help="Release to download (default: every release in "
                             "RELEASES). Repeatable, or comma-separated.")
    parser.add_argument("--source", action="append", default=[],
                        help="RO, MOD, or both (default: both)")
    parser.add_argument("--dstart", default=None,
                        help="First month, YYYY-MM-DD (default: the release start)")
    parser.add_argument("--dend", default=None,
                        help="Last month, YYYY-MM-DD (default: the release end)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    releases = resolve(args.release, RELEASES, "release")
    sources = resolve(args.source, SOURCES, "source")

    client = cdsapi.Client(url=CDS_URL, key=CDS_KEY_FILE.read_text().strip(),
                           progress=False, timeout=3600)

    for release in releases:
        config = RELEASES[release]
        if config["end"] is None:
            raise SystemExit(
                f"Release '{release}' has no end date yet. Read the last "
                f"available month from the dataset constraints and set it in "
                f"RELEASES before downloading.")
        start = pd.Timestamp(args.dstart or config["start"])
        end = pd.Timestamp(args.dend or config["end"])
        start = max(start, pd.Timestamp(config["start"]))
        end = min(end, pd.Timestamp(config["end"]))
        if start > end:
            logging.warning("%s: requested range does not overlap the release "
                            "(%s to %s), skipping.",
                            release, config["start"], config["end"])
            continue

        for source in sources:
            source_config = SOURCES[source]
            months = pd.date_range(start.replace(day=1), end.replace(day=1), freq="MS")

            # One request per calendar year: the zip comes back holding one file
            # per month, so a year costs a single queue slot instead of twelve.
            for year, year_months in months.groupby(months.year).items():
                year_months = pd.DatetimeIndex(year_months)
                label = f"{release} {source} {year}"
                have = months_on_disk(release, source, year_months)
                wanted = {m.strftime("%Y%m") for m in year_months}
                if have == wanted:
                    logging.info("%s: already extracted (%s months), skipping.",
                                 label, len(have))
                    continue

                staging_dir = source_dir(release, source) / f".staging_{year}"
                zip_file = source_dir(release, source) / f".{source}_{year}.zip"
                zip_file.parent.mkdir(parents=True, exist_ok=True)

                request = {
                    "variable": "all",
                    "product_type": source_config["product_type"],
                    "year": f"{year}",
                    "month": sorted({f"{m.month:02d}" for m in year_months}),
                    **config["request"],
                }
                if not zip_file.exists() and not retrieve(client, request, zip_file, label):
                    continue

                try:
                    shutil.rmtree(staging_dir, ignore_errors=True)
                    staging_dir.mkdir(parents=True)
                    with zipfile.ZipFile(zip_file) as archive:
                        archive.extractall(staging_dir)
                    moved = place_extracted_files(release, source, staging_dir)
                    shutil.rmtree(staging_dir)
                    zip_file.unlink()
                    logging.info("%s: extracted %s file(s).", label, moved)
                except Exception as exception:
                    logging.error("%s: extraction failed (%s)", label, exception)
                    shutil.rmtree(staging_dir, ignore_errors=True)
                    zip_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
