"""Compute UTH time-series and spatial-consistency statistics."""

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar


DATA_DIR = Path("../../../datasets/UTH/daily")
DATE_PATTERN = re.compile(r"(\d{8})")

# The aux outputs follow the same convention as every other dataset in this
# repository, which is what build_QC_gallery.py looks them up by:
#   aux_files/tseries_stats_<KEY>_<FREQUENCY>_<start>_<end>.nc
#   aux_files/spatial_consistency_<KEY>_<FREQUENCY>_<start>_<end>.nc
AUX_DIR = Path("aux_files")
VAR_KEY = "UTH"        # the monitored quantity, as the aux filenames carry it
FREQUENCY = "daily"    # edition 2 is delivered as daily means only


def timestamp_for(path, dataset):
    """The nominal date of a daily mean.

    The time coordinate is the START of the 24-hour averaging window (D-1 23:30
    to D 23:30), so reading it verbatim labelled every daily mean one calendar
    day early: the series ran 1994-07-05 to 2018-12-30 for files dated
    1994-07-06 to 2018-12-31, and the statistics then disagreed with the file
    dates the QC gallery reports alongside them. The centre of time_bnds is the
    nominal date, and it agrees with both the filename and the reference date of
    the time:units attribute.
    """
    if "time_bnds" in dataset:
        bounds = pd.DatetimeIndex(np.ravel(dataset.time_bnds.values))
        return (bounds[0] + (bounds[-1] - bounds[0]) / 2).normalize()
    match = DATE_PATTERN.search(path.name)
    if match is not None:
        return pd.to_datetime(match.group(1), format="%Y%m%d")
    if "time" in dataset.coords:
        return pd.Timestamp(dataset.time.values[0]).normalize()
    raise ValueError(f"No time bounds, time coordinate or YYYYMMDD date in {path}")


def data_files(dstart, dend):
    files = []
    for path in DATA_DIR.rglob("*.nc"):
        match = DATE_PATTERN.search(path.name)
        if match is not None and dstart <= pd.Timestamp(match.group(1)) <= dend:
            files.append(path)
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Compute statistics for UTH")
    parser.add_argument("--dstart", default="1994-07-01")
    parser.add_argument("--dend", default="2018-12-31")
    parser.add_argument("--variable", default="mean_uth")
    parser.add_argument("--do-spatial", action="store_true")
    args = parser.parse_args()

    dstart = pd.Timestamp(args.dstart)
    dend = pd.Timestamp(args.dend)
    files = data_files(dstart, dend)
    if not files:
        raise SystemExit(f"No UTH files found in {DATA_DIR} for the requested period.")

    records = []
    for path in files:
        with xr.open_dataset(path) as dataset:
            data = dataset[args.variable].squeeze(drop=True)
            timestamp = timestamp_for(path, dataset)
            weights = np.cos(np.deg2rad(data.latitude))
            records.append({
                "time": timestamp,
                "mean": float(data.weighted(weights).mean(("latitude", "longitude"))),
                "median": float(data.median()),
                "std": float(data.weighted(weights).std(("latitude", "longitude"))),
                "minimum": float(data.min()),
                "maximum": float(data.max()),
                "p01": float(data.quantile(0.01)),
                "p99": float(data.quantile(0.99)),
                "number_of_values": int(data.count()),
                "missing": int(data.isnull().sum()),
                "missing_fraction": float(
                    xr.where(data.isnull(), 1.0, 0.0).weighted(weights).mean(("latitude", "longitude"))
                ),
            })

    stats = pd.DataFrame(records).set_index("time").sort_index()

    AUX_DIR.mkdir(exist_ok=True)
    # The period in the filename is the one actually covered by the statistics,
    # not the one requested: a run asking for more than has been delivered must
    # not claim the wider range.
    period = f"{stats.index[0]:%Y%m%d}_{stats.index[-1]:%Y%m%d}"
    tseries_path = AUX_DIR / f"tseries_stats_{VAR_KEY}_{FREQUENCY}_{period}.nc"
    xr.Dataset.from_dataframe(stats).to_netcdf(tseries_path)
    print(f"Wrote {tseries_path} ({len(stats)} days)")

    if not args.do_spatial:
        return

    dataset = xr.open_mfdataset([str(path) for path in files], combine="nested",
                                concat_dim="time", chunks={"time": 30})
    data = dataset[args.variable]
    spatial = xr.Dataset({
        "Missing_values": data.isnull().sum("time"),
        "Number_of_values": data.notnull().sum("time"),
        "Max_value": data.max("time"),
        "Min_value": data.min("time"),
    })
    spatial_path = AUX_DIR / f"spatial_consistency_{VAR_KEY}_{FREQUENCY}_{period}.nc"
    with ProgressBar():
        spatial.to_netcdf(spatial_path)
    print(f"Wrote {spatial_path}")
    dataset.close()


if __name__ == "__main__":
    main()