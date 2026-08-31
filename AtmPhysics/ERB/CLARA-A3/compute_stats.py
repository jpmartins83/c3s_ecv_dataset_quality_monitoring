import xarray as xr
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
import argparse
import glob
import re
import datetime

def need_to_update_existing_file(stats_dir, filestring, frequency, dstart, dend,climatology_start):    
    existing_files = glob.glob(os.path.join(stats_dir, f"{filestring}_{frequency}_*.nc"))
    # print(existing_files)
    # If multiple summary files exist, pick the most recent one by modification time
    old_stats_path = max(existing_files, key=os.path.getmtime) if existing_files else None
    if old_stats_path:
        print(f"Found existing summary file: {old_stats_path}")

        # Extract dates from filename (expects format: summary_stats_frequency_YYYYMMDD_to_YYYYMMDD.nc)
        date_matches = re.findall(r"\d{8}", os.path.basename(old_stats_path))
        if len(date_matches) != 2:
                raise ValueError(
                    f"Could not parse start and end dates from filename: {old_stats_path}"
                )
        old_start_str, old_end_str = date_matches
        if old_start_str == dstart and old_end_str == dend:
            print(f"Existing summary file already covers the requested date range {dstart} to {dend}. It will be overwritten.")
            os.remove(old_stats_path)
            update_existing_file = False
        elif dstart < old_end_str:
            print(f"Existing summary file covers the date range {old_start_str} to {old_end_str}.")
            print(f"Requested date range is {dstart} to {dend}.")
            print("The requested start date is before the end date of the existing summary file. This indicates a recalculation rather than an extension. File will be deleted and recalculated.")
            os.remove(old_stats_path)
            update_existing_file = False
        else:
            print(f"Existing summary file covers the date range {old_start_str} to {old_end_str}.")
            print(f"Requested date range is {dstart} to {dend}.")
            update_existing_file = True
    else:
        print("No existing summary file found. A new summary file will be created.")
        update_existing_file = False


    if not update_existing_file:
        new_start_dt = pd.to_datetime(dstart)
        new_end_dt = pd.to_datetime(dend)
        if pd.Timestamp(new_start_dt) > climatology_start:
            raise ValueError(
                            f"The start date {new_start_dt} is after the climatology period start date {climatology_start}. Please adjust "
                            f"the start date to be within the climatology period ({climatology_start:%Y-%m-%d} onwards) for spatial consistency computation."
                        )

    else:
        
        old_end_dt = pd.to_datetime(old_end_str, format="%Y%m%d")
        old_start_dt = pd.to_datetime(old_start_str, format="%Y%m%d")
        new_start_dt = old_start_dt
        new_end_dt = pd.to_datetime(dend)

        # 3. GAP CHECK: Verify the new data directly follows the old data
        time_gap = pd.to_datetime(dstart) - old_end_dt
        # If the gap is longer than allowed (e.g., more than 1 day), raise an error
        if time_gap > datetime.timedelta(days=1):
            raise ValueError(
                f"Gap detected! Previous stats data ended on {old_end_dt.strftime('%Y-%m-%d')}, "
                f"but new data starts on {dstart}. "
                f"Gap size: {time_gap.days} days."
                f"Please ensure that the new data is continuous with the existing data or adjust the date range accordingly."
            )
    print(new_start_dt, new_end_dt)
    return update_existing_file, old_stats_path, new_start_dt, new_end_dt


"""
Compute statistics of the dataset offline.

This script computes various statistics such as spatial consistency,
missing values, and climatology thresholds for outlier detection.

Accepts as arguments the frequency of the data (daily or monthly), start date, and end date.

Hardcoded variables:
- var: The variable to analyze (outgoing_longwave_radiation or outgoing_shortwave_radiation).
- path patterns
- output file names for statistics and climatology thresholds.
- climatological period for threshold computation.

no need to change code below end of configuration section, unless you know what you are doing.

"""

# parse command line arguments
parser = argparse.ArgumentParser(description="Compute statistics for a dataset")
parser.add_argument("--frequency", type=str, default="daily_mean", help="Frequency of the data (daily_mean or monthly_mean)")
parser.add_argument("--dstart", type=str, default="1979-01-01", help="Start date for the data")
parser.add_argument("--dend", type=str, default="2026-07-31", help="End date for the data")
parser.add_argument("--variable", type=str, default="outgoing_longwave_radiation", help="Variable to analyze")
parser.add_argument("--do-spatial", action="store_true", help="Compute the spatial consistency maps")
parser.add_argument("--do-p999", action="store_true", help="Compute the climatological P0.1/P99.9 thresholds")
args = parser.parse_args()
frequency = args.frequency
dstart = args.dstart
dend = args.dend
variable = args.variable

#  --- script options - we may skip certain parts if needed ----
# ideally do_compute_p999 should only be used in case of a new CDR
do_compute_spatial_consistency = False or args.do_spatial
do_compute_p999 = False or args.do_p999  # compute p99.9. this only needs to be done once, then the thresholds are saved to a file and can be reused. no need to change if more recent files become available
do_compute_stats_tseries = True  # compute stats timeseries 
climatology_start = pd.Timestamp("1979-01-01")  
climatology_end   = pd.Timestamp("2020-12-31")  # end of climatology period for threshold computation

DATASET = "satellite-earth-radiation-budget-eumdac"
PRODUCT = "CLARA-A3"
ECV = "ERB"

# Both products are delivered at both frequencies.
variables = [
    "outgoing_longwave_radiation",
    "outgoing_shortwave_radiation",
]

# var_strings maps the dataset variable names to the strings used in the filenames.
var_strings = {
    "outgoing_longwave_radiation": "OLR",
    "outgoing_shortwave_radiation": "RSF"
}
# For ERB the name of the variable inside the file is NOT the string used in the
# filename, so the two have to be mapped separately.
nc_variables = {
    "OLR": "LW_flux",
    "RSF": "SW_flux"
}
freq_strings = {
    'daily_mean': 'dm',
    'monthly_mean': 'mm'
}

dir_dataset = Path(f"../../../datasets/{ECV}/{PRODUCT}/{frequency}")

# variable = "outgoing_shortwave_radiation"
var_string = var_strings[variable]
nc_var = nc_variables[var_string]
freq_string = freq_strings[frequency]
    
dates = pd.date_range(dstart, dend, freq="D" if frequency == "daily_mean" else "MS")

fnames = [
    f"{dir_dataset}/"
    + (f"{date.year}/{date.month:02d}/") # month folder for daily, year folder for monthly
    + (f"{var_string}{freq_string}{date:%Y%m%d}000000319AVPOS")
    + (["01GL.nc" if date <= climatology_end else "I1GL.nc"][0])  # CDR vs ICDR naming
    
    for date in dates
]

existing_fnames = [f for f in fnames if Path(f).exists()]
missing_fnames = [f for f in fnames if not Path(f).exists()]

print(f"Expected: {len(fnames)}")
print(f"Existing: {len(existing_fnames)}")
print(f"Missing:  {len(missing_fnames)}")

for f in missing_fnames:
    print(f)

stats_dir = "aux_files"


# ------- end of configuration section --------


if not os.path.exists(stats_dir):
    os.makedirs(stats_dir)

###### compute map of the frequency of missing values for all files
if do_compute_spatial_consistency:

    # check if there are existing summary files for the given frequency.
    # if not, we will compute a new summary file for the entire date range.
    # if yes, we test further to check if it is a recalculation or an extension
    # if it is an extension, we will use the most recent one to avoid recomputing statistics for files that have already been processed. if not, we delete theold file and recompute
    
    filestring = 'spatial_consistency'    
    update_existing_file, old_stats_path, new_start_dt, new_end_dt = need_to_update_existing_file(stats_dir, filestring, frequency, dstart, dend,climatology_start)
        
    print(f"Computing spatial consistency for all files between {new_start_dt} and {new_end_dt}, this might take a while...")

    # open mf dataset between dstart and dend
    # Only the files that exist: a date missing from the delivery would
    # otherwise abort the whole run when open_mfdataset reaches it.
    ds = xr.open_mfdataset(
        existing_fnames,
        combine="nested",
        concat_dim="time",
        chunks={"time": 1}
    )

    new_missing    = ds[nc_var].isnull().sum("time").rename("Missing_values")
    new_num_values = ds[nc_var].notnull().sum("time").rename("Number_of_values")
    new_max_values = ds[nc_var].max("time").rename("Max_value")
    new_min_values = ds[nc_var].min("time").rename("Min_value")

    if update_existing_file:

        # Load old stats into memory
        old_stats = xr.open_dataset(old_stats_path).load()
        # Combine old and new statistics using safe Xarray tools
        updated_ds = xr.Dataset()
        updated_ds["Missing_values"] = old_stats["Missing_values"].fillna(0) + new_missing.fillna(0)
        updated_ds["Number_of_values"] = old_stats["Number_of_values"].fillna(0) + new_num_values.fillna(0)
        updated_ds["Max_value"] = xr.concat([old_stats["Max_value"], new_max_values], dim="temp").max("temp")
        updated_ds["Min_value"] = xr.concat([old_stats["Min_value"], new_min_values], dim="temp").min("temp")
    else:
        updated_ds = xr.Dataset({
            "Missing_values": new_missing,
            "Number_of_values": new_num_values,
            "Max_value": new_max_values,
            "Min_value": new_min_values
        })

    summary_ds = xr.merge([updated_ds["Missing_values"], updated_ds["Number_of_values"], updated_ds["Max_value"], updated_ds["Min_value"]])
    outfile = os.path.join(stats_dir, f"spatial_consistency_{var_string}_{frequency}_{new_start_dt:%Y%m%d}_{new_end_dt:%Y%m%d}.nc")
    print(f"Writing spatial stats to {outfile}")
    summary_ds.to_netcdf(outfile)
 


#### compute climatology thresholds for outlier detection
climatology_file = Path(stats_dir,f"{var_string}_p999_{climatology_start:%Y%m%d}_{climatology_end:%Y%m%d}.nc")
if os.path.exists(climatology_file):
    print(f"Climatology thresholds file {climatology_file} already exists. It will be reused.")
    do_compute_p999 = False
elif do_compute_stats_tseries:
    # The time series statistics need the thresholds, so compute them on the fly.
    print(f"Climatology thresholds file {climatology_file} does not exist. It will be computed.")
    do_compute_p999 = True

if do_compute_p999:
    print(f"Computing monthly P0.01 and P99.9 thresholds for {climatology_start:%Y}-{climatology_end:%Y%} using monthly files...")

    # --------------------------------------------------------
    # Extract dates from filenames
    # --------------------------------------------------------

    dates_d = pd.date_range(climatology_start, climatology_end, freq="MS") # the thresholds are always derived from the monthly means
    # print(dates_d)

    # The thresholds come from the monthly means whatever the frequency of this
    # run, so that both frequencies share one threshold file. Point at the
    # monthly directory and use the monthly filename token explicitly.
    dir_monthly = Path(f"../../../datasets/{ECV}/{PRODUCT}/monthly_mean")

    fnames_d = [
        f"{dir_monthly}/"
        + (f"{date.year}/{date.month:02d}/")
        + (f"{var_string}mm{date:%Y%m%d}000000319AVPOS")
        + (["01GL.nc" if date <= climatology_end else "I1GL.nc"][0])  # CDR vs ICDR naming

        for date in dates_d
    ]
    # print(fnames_d)

    file_dates = pd.to_datetime(
        pd.Series(fnames_d)
        .str.extract(rf"{var_string}mm(\d{{8}})", expand=False),  # using monthly files here
        format="%Y%m%d"
    )
    # print(file_dates)

    # Files within climatology period
    clim_mask = (
        (file_dates >= climatology_start) &
        (file_dates <= climatology_end)
    )

    clim_fnames = np.array([f for f in np.array(fnames_d)[clim_mask] if Path(f).exists()])
    print(clim_fnames)

    # --------------------------------------------------------
    # Calculate threshold for each calendar month
    # --------------------------------------------------------

    monthly_p001 = []
    monthly_p999 = []

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

        x = ds_clim[nc_var]

        # P99.9 over all time, latitude and longitude
        p001,p999 = x.quantile(
            [0.001, 0.999],
            dim=("time", "lat", "lon")
        ).compute()

        print(f"P0.1 = {p001}, P99.9 = {p999}")

        monthly_p001.append(float(p001))
        monthly_p999.append(float(p999))

        ds_clim.close()

    # --------------------------------------------------------
    # Save thresholds
    # --------------------------------------------------------

    threshold_ds = xr.Dataset(
        {
            f"{var_string}_p001": (
                ["month"],
                monthly_p001
            ),f"{var_string}_p999": (
                ["month"],
                monthly_p999
            )

        },
        coords={
            "month": np.arange(1, 13)
        }
    )

    threshold_ds[f"{var_string}_p001"].attrs = {
        "long_name": "Monthly climatological 0.1th percentile",
        "quantile": 0.001
    }

    threshold_ds[f"{var_string}_p999"].attrs = {
            "long_name": "Monthly climatological 99.9th percentile",
            "quantile": 0.999
        }

    threshold_ds.to_netcdf(climatology_file)

    print(f"\nSaved to {climatology_file}")



if do_compute_stats_tseries:

    # ============================================================
    # Load thresholds
    # ============================================================

    threshold_ds = xr.open_dataset(climatology_file)

    monthly_p001 = threshold_ds[f"{var_string}_p001"]
    monthly_p999 = threshold_ds[f"{var_string}_p999"]

    # print("\nMonthly climatological thresholds:")
    # print(monthly_thresholds)

    # stats_tseries_file = Path("aux_files", f"tseries_stats_{frequency}.nc")
    filestring = f'tseries_stats_{var_string}'
    update_existing_file, old_stats_path, new_start_dt, new_end_dt = need_to_update_existing_file(stats_dir, filestring, frequency, dstart, dend,climatology_start)

    if update_existing_file:
        print(f"Updating existing stats time series file: {old_stats_path}")
    
    stats_tseries_file = os.path.join(stats_dir, f"{filestring}_{frequency}_{new_start_dt:%Y%m%d}_{new_end_dt:%Y%m%d}.nc")
    print(f"Creating new stats time series file for {frequency} data.\nNew file will be saved to: {stats_tseries_file}")
    ###### computing statistics for each file
    print(f"Computing stats for {frequency} data, saving to {stats_tseries_file}")

    records = []

    for f in fnames:

        print(f"Processing {f}")

        if not os.path.exists(f):
            print(f"File {f} does not exist, skipping.")
            continue

        ds = xr.open_dataset(f)

        x = ds[nc_var]
        month = pd.Timestamp(ds.time.values[0]).month
        p001 = float(monthly_p001.sel(month=month))
        p999 = float(monthly_p999.sel(month=month))
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

            "negative_outliers_fraction":
                float(
                    (x < p001)
                    .astype(float)
                    .weighted(weights)
                    .mean(("lat", "lon"))
                ),

            # ----------------------------------------------------
            # Outliers relative to CDR climatology
            # ----------------------------------------------------

            "positive_outliers_fraction":
                float(
                    (x > p999)
                    .astype(float)
                    .weighted(weights)
                    .mean(("lat", "lon"))
                ),
        })

        ds.close()    

    stats = pd.DataFrame(records)
    stats = stats.set_index("time")

    if update_existing_file:
        old_stats = xr.open_dataset(old_stats_path).to_dataframe()
        # Combine old and new
        stats = pd.concat([old_stats, stats])

        # Remove duplicate dates
        stats = stats[~stats.index.duplicated(keep="last")]
        # Sort chronologically
        stats = stats.sort_index()

    new = xr.Dataset.from_dataframe(stats)
    new.to_netcdf(stats_tseries_file)