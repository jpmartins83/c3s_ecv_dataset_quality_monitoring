import xarray as xr
import numpy as np
import pandas as pd
import os
import sys
from dask.diagnostics import ProgressBar
from pathlib import Path

frequency = sys.argv[1] if len(sys.argv) > 1 else "daily"
do_compute_missing = False
do_compute_climatology = False

records = []
var = "precipitation"
stats_file = f"GIRAFE_{frequency}_stats.nc"
    
dates = pd.date_range("2002-01-01", "2026-06-30", freq="D" if frequency == "daily" else "MS")
fnames = [
    f"../../../datasets/GIRAFE/{frequency}/"
    + (f"{date.year}/{date.month:02d}/" if frequency == "daily" else f"{date.year}/") # month folder for daily, year folder for monthly
    + (f"PREdm{date:%Y%m%d}000000120" if frequency == "daily" else f"PREmm{date:%Y%m%d}000000120")  # different file naming for daily vs monthly
    + f"{'IMPGS01GL' if date < pd.Timestamp('2023-01-01') else 'IMPGSI1GL'}.nc" # CDR vs ICDR naming
    for date in dates
]

###### compute map of the frequency of missing values for all files
if do_compute_missing:
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


#### compute climatology thresholds for outlier detection
climatology_file = "GIRAFE_climatology_thresholds.nc"
climatology_start = "2002-01-01"
climatology_end = "2022-12-31"
if do_compute_climatology:
    print("Computing monthly P99.9 thresholds for 2002–2022 using daily files...")

    # --------------------------------------------------------
    # Extract dates from filenames
    # --------------------------------------------------------

    dates_d = pd.date_range("2002-01-01", "2026-06-30", freq="D" )
    fnames_d = [
        f"../../../datasets/GIRAFE/daily/"
        + (f"{date.year}/{date.month:02d}/") # month folder for daily, year folder for monthly
        + (f"PREdm{date:%Y%m%d}000000120")  # different file naming for daily vs monthly
        + f"{'IMPGS01GL'}.nc" # CDR vs ICDR naming
        for date in dates_d
    ]

    file_dates = pd.to_datetime(
        pd.Series(fnames_d)
        .str.extract(r"PREdm(\d{8})", expand=False),  # using daily files here
        format="%Y%m%d"
    )

    # Files within climatology period
    clim_mask = (
        (file_dates >= climatology_start) &
        (file_dates <= climatology_end)
    )

    clim_fnames = np.array(fnames_d)[clim_mask]

    # --------------------------------------------------------
    # Calculate threshold for each calendar month
    # --------------------------------------------------------

    thresholds = []

    for month in range(1, 13):

        print(f"\nProcessing month {month:02d}...")

        # Select files belonging to this calendar month
        month_mask = file_dates[clim_mask].dt.month == month

        month_files = clim_fnames[month_mask.values]

        print(f"Number of files: {len(month_files)}")

        # Open the files lazily with Dask
        ds_clim = xr.open_mfdataset(
            month_files.tolist(),
            combine="nested",
            concat_dim="time",
            chunks={"time": 30}
        )

        x = ds_clim[var]

        # P99.9 over all time, latitude and longitude
        threshold = x.quantile(
            0.999,
            dim=("time", "lat", "lon")
        ).compute()

        threshold_value = float(threshold)

        print(f"P99.9 = {threshold_value}")

        thresholds.append(threshold_value)

        ds_clim.close()

    # --------------------------------------------------------
    # Save thresholds
    # --------------------------------------------------------

    threshold_ds = xr.Dataset(
        {
            f"{var}_p999": (
                ["month"],
                thresholds
            )
        },
        coords={
            "month": np.arange(1, 13)
        }
    )

    threshold_ds[f"{var}_p999"].attrs = {
        "long_name": "Monthly climatological 99.9th percentile",
        "description": (
            "99.9th percentile of precipitation values "
            "for each calendar month over 2002–2022"
        ),
        "climatology_period": "2002-2022",
        "quantile": 0.999
    }

    threshold_ds.to_netcdf(climatology_file)

    print(f"\nSaved to {climatology_file}")


# ============================================================
# Load thresholds
# ============================================================

threshold_ds = xr.open_dataset(climatology_file)

monthly_thresholds = threshold_ds[f"{var}_p999"]

print("\nMonthly climatological thresholds:")
print(monthly_thresholds)


###### computing statistics for each file
print(f"Computing stats for {frequency} data, saving to {stats_file}")

for f in fnames:

    print(f"Processing {f}")

    if not os.path.exists(f):
        print(f"File {f} does not exist, skipping.")
        continue

    ds = xr.open_dataset(f)

    x = ds[var]
    month = pd.Timestamp(ds.time.values[0]).month
    threshold = float(monthly_thresholds.sel(month=month))
    # Latitude weighting
    weights = np.cos(np.deg2rad(ds.lat))

    records.append({

        "time":
            pd.Timestamp(ds.time.values[0]),

        # ----------------------------------------------------
        # Distribution
        # ----------------------------------------------------

        "mean":
            float(
                x.weighted(weights).mean(("lat", "lon"))
            ),

        "median":
            float(x.median()),

        "std":
            float(
                x.weighted(weights).std(("lat", "lon"))
            ),

        "minimum":
            float(x.min()),

        "maximum":
            float(x.max()),

        "p01":
            float(x.quantile(0.01)),

        "p99":
            float(x.quantile(0.99)),

        # ----------------------------------------------------
        # Missing data
        # ----------------------------------------------------

        "missing":
            int(x.isnull().sum()),

        "missing_fraction":
            float(
                xr.where(
                    np.isnan(x),
                    1.0,
                    0.0
                ).weighted(weights).mean(("lat", "lon"))
            ),

        # ----------------------------------------------------
        # Negative values
        # ----------------------------------------------------

        "negative_fraction":
            float(
                (x < 0)
                .astype(float)
                .weighted(weights)
                .mean(("lat", "lon"))
            ),

        # ----------------------------------------------------
        # Outliers relative to CDR climatology
        # ----------------------------------------------------

        "outliers_fraction":
            float(
                (x > threshold)
                .astype(float)
                .weighted(weights)
                .mean(("lat", "lon"))
            ),
    })

    ds.close()    

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