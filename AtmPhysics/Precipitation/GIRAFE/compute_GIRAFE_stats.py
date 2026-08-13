import xarray as xr
import numpy as np
import pandas as pd
import os
import sys
from dask.diagnostics import ProgressBar
from pathlib import Path

frequency = sys.argv[1] if len(sys.argv) > 1 else "daily"

records = []
var = "precipitation"
stats_file = f"GIRAFE_{frequency}_stats.nc"
print(f"Computing stats for {frequency} data, saving to {stats_file}")

    
dates = pd.date_range("2002-01-01", "2026-06-30", freq="D" if frequency == "daily" else "MS")
fnames = [
    f"../../../datasets/GIRAFE/{frequency}/"
    + (f"{date.year}/{date.month:02d}/" if frequency == "daily" else f"{date.year}/")
    + (f"PREdm{date:%Y%m%d}000000120" if frequency == "daily" else f"PREmm{date:%Y%m%d}000000120")
    + f"{'IMPGS01GL' if date < pd.Timestamp('2023-01-01') else 'IMPGSI1GL'}.nc"
    for date in dates
]

print("Computing missing values for all files, this might take a while...")

ds = xr.open_mfdataset(
    fnames,
    combine="nested",
    concat_dim="time",
    chunks={"time": 1}
)

# raise SystemExit("Stopping here for testing, remove this line to continue processing.")
missing = (
    ds[var]
    .isnull()
    .sum("time")
    .rename("Missing_values")
)
# print(missing)
file_path = Path(f"GIRAFE_{frequency}_missing.nc")
print(f"Writing missing values to {file_path}")
missing.to_netcdf(file_path)
# 2. Check if file exists and is not empty
if file_path.is_file() and file_path.stat().st_size > 0:
    print("File was written successfully.")
else:
    print("File write failed or file is empty.")  

print("Computing stats for all files...")
raise SystemExit
for f in fnames:
    print(f"Processing {f}")
    if not os.path.exists(f): 
        print(f"File {f} does not exist, skipping.")
        continue
    ds = xr.open_dataset(f)

    x = ds[var]
    weights = np.cos(np.deg2rad(ds.lat))
    threshold = x.quantile(0.999)

    records.append(

        {

        "time": pd.Timestamp(ds.time.values[0]),

        "mean": float(x.weighted(weights).mean(("lat", "lon"))),

        "std": float(x.weighted(weights).std(("lat", "lon"))),

        "minimum": float(x.min()),

        "maximum": float(x.max()),

        "missing_fraction":

            float(xr.where(np.isnan(x), 1.0, 0.0).weighted(weights).mean(("lat", "lon"))),

        "negative_fraction":

            float((x < 0).astype(float).weighted(weights).mean(("lat","lon"))),

        "outliers_fraction":

            float((x > threshold).astype(float).weighted(weights).mean(("lat","lon"))),

        "p01":

            float(x.quantile(0.01)),

        "p99":

            float(x.quantile(0.99))

        }

    )

stats = pd.DataFrame(records)
stats = stats.set_index("time")

if os.path.exists(stats_file):

    old_stats = xr.open_dataset(stats_file).to_dataframe()

    # Combine old and new
    stats = pd.concat([old_stats, stats])

    # Remove duplicate dates
    stats = stats[~stats.index.duplicated(keep="last")]

    # Sort chronologically
    stats = stats.sort_index()

new = xr.Dataset.from_dataframe(stats)

new.to_netcdf(stats_file)