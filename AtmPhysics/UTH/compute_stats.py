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


def timestamp_for(path, dataset):
    if "time" in dataset.coords:
        return pd.Timestamp(dataset.time.values[0])
    match = DATE_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"No time coordinate or YYYYMMDD date in {path}")
    return pd.to_datetime(match.group(1), format="%Y%m%d")


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
            weights = np.cos(np.deg2rad(data.lat))
            records.append({
                "time": timestamp,
                "mean": float(data.weighted(weights).mean(("lat", "lon"))),
                "median": float(data.median()),
                "std": float(data.weighted(weights).std(("lat", "lon"))),
                "minimum": float(data.min()),
                "maximum": float(data.max()),
                "p01": float(data.quantile(0.01)),
                "p99": float(data.quantile(0.99)),
                "number_of_values": int(data.count()),
                "missing": int(data.isnull().sum()),
                "missing_fraction": float(
                    xr.where(data.isnull(), 1.0, 0.0).weighted(weights).mean(("lat", "lon"))
                ),
            })

    stats = pd.DataFrame(records).set_index("time").sort_index()
    xr.Dataset.from_dataframe(stats).to_netcdf("UTH_daily_stats.nc")

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
    with ProgressBar():
        spatial.to_netcdf("UTH_daily_spatial_consistency.nc")
    dataset.close()


if __name__ == "__main__":
    main()