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
import time
from concurrent.futures import ProcessPoolExecutor


def compute_file_stats(task):
    """Compute the statistics of a single file.

    Runs in a worker process, so it takes everything it needs as arguments and
    returns plain Python types. The reductions are done in numpy on the loaded
    array rather than through xarray: the three order statistics (p01, median,
    p99) then come out of a single sort instead of three, which is roughly twice
    as fast per file. The results are identical to the xarray formulation:
      - mean/std are latitude-weighted over the valid points only
      - missing_fraction is weighted over all points
      - the outlier fractions count NaN as "not an outlier" and are weighted
        over all points
    """
    fname, nc_var, p001_by_month, p999_by_month = task

    if not os.path.exists(fname):
        return {"absent": fname}

    with xr.open_dataset(fname) as ds:
        da = ds[nc_var]
        if "time" in da.dims:
            da = da.isel(time=0)
        values = np.asarray(da.values, dtype="float64")
        lat = ds.lat.values
        timestamp = pd.Timestamp(ds.time.values[0])

    month = timestamp.month
    p001 = p001_by_month[month - 1]
    p999 = p999_by_month[month - 1]

    # Latitude weighting, broadcast over longitude
    weights = np.broadcast_to(
        np.cos(np.deg2rad(lat))[:, None], values.shape
    )

    valid = ~np.isnan(values)
    v = values[valid]
    w = weights[valid]
    w_total = weights.sum()
    w_valid = w.sum()

    if v.size == 0:
        # Completely empty field (a day with no observations at all). The xarray
        # reductions returned NaN for every statistic here, so reproduce that
        # rather than dividing by a zero weight or asking numpy for the
        # percentile of an empty array.
        return {
            "time": timestamp,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "minimum": np.nan,
            "maximum": np.nan,
            "p01": np.nan,
            "p99": np.nan,
            "missing": int(values.size),
            "missing_fraction": 1.0,
            "negative_outliers_fraction": 0.0,
            "positive_outliers_fraction": 0.0,
        }

    mean = (v * w).sum() / w_valid
    std = np.sqrt((w * (v - mean) ** 2).sum() / w_valid)
    p01, median, p99 = np.percentile(v, [1, 50, 99])

    return {
        "time": timestamp,
        # Distribution
        "mean": float(mean),
        "median": float(median),
        "std": float(std),
        "minimum": float(v.min()),
        "maximum": float(v.max()),
        "p01": float(p01),
        "p99": float(p99),
        # Missing data
        "missing": int((~valid).sum()),
        "missing_fraction": float(weights[~valid].sum() / w_total),
        # Outliers relative to the CDR climatology. The thresholds are derived
        # from the same frequency as the data being compared (see the threshold
        # filename, which carries the frequency): a threshold taken from monthly
        # means and applied to daily values flagged 7-24x more than the ~0.1% a
        # P99.9 threshold implies, because daily values spread much wider.
        "negative_outliers_fraction": float(w[v < p001].sum() / w_total),
        "positive_outliers_fraction": float(w[v > p999].sum() / w_total),
    }

def need_to_update_existing_file(stats_dir, filestring, frequency, dstart, dend,climatology_start):    
    existing_files = glob.glob(os.path.join(stats_dir, f"{filestring}_{frequency}_*.nc"))
    # print(existing_files)
    # If multiple summary files exist, pick the most recent one by modification time
    old_stats_path = max(existing_files, key=os.path.getmtime) if existing_files else None
    # Deleting an existing file is deferred until the requested range has been
    # validated, so that a rejected invocation cannot destroy it on the way out.
    path_to_remove = None
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
            path_to_remove = old_stats_path
            update_existing_file = False
        elif dstart < old_end_str:
            print(f"Existing summary file covers the date range {old_start_str} to {old_end_str}.")
            print(f"Requested date range is {dstart} to {dend}.")
            print("The requested start date is before the end date of the existing summary file. This indicates a recalculation rather than an extension. File will be deleted and recalculated.")
            path_to_remove = old_stats_path
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
                            f"the start date to be within the climatology period (2002-01-01 to 2022-12-31) for spatial consistency computation."
                        )

    else:
        
        old_end_dt = pd.to_datetime(old_end_str, format="%Y%m%d")
        old_start_dt = pd.to_datetime(old_start_str, format="%Y%m%d")
        new_start_dt = old_start_dt
        new_end_dt = pd.to_datetime(dend)

        # 3. GAP CHECK: verify that the new range directly follows the old one,
        # i.e. that no expected time step falls between the two. Looking at the
        # actual time steps of this frequency, rather than at a fixed number of
        # days, lets a monthly extension start on the first of the new month: the
        # end date in the filename is the requested end, not the last month
        # start, so a fixed one-day tolerance rejected the natural call.
        resolution = "D" if frequency.startswith("daily") else "MS"
        skipped = pd.date_range(
            old_end_dt + pd.Timedelta(days=1),
            pd.to_datetime(dstart) - pd.Timedelta(days=1),
            freq=resolution,
        )
        if len(skipped) > 0:
            listed = ", ".join(d.strftime("%Y-%m-%d") for d in skipped[:5])
            raise ValueError(
                f"Gap detected! Previous stats data ended on {old_end_dt:%Y-%m-%d}, "
                f"but new data starts on {pd.to_datetime(dstart):%Y-%m-%d}, which skips "
                f"{len(skipped)} {frequency} time step(s): {listed}"
                f"{' ...' if len(skipped) > 5 else ''}. "
                f"Please ensure that the new data is continuous with the existing data or adjust the date range accordingly."
            )
    if path_to_remove:
        os.remove(path_to_remove)

    print(new_start_dt, new_end_dt)
    return update_existing_file, old_stats_path, new_start_dt, new_end_dt


"""
Compute statistics of the dataset offline.

This script computes various statistics such as spatial consistency,
missing values, and climatology thresholds for outlier detection.

Accepts as arguments the frequency of the data (daily or monthly), start date, and end date.

Hardcoded variables:
- var: The variable to analyze (currently set to "precipitation").
- path patterns
- output file names for statistics and climatology thresholds.
- climatological period for threshold computation.

no need to change code below end of configuration section, unless you know what you are doing.

"""

# parse command line arguments
parser = argparse.ArgumentParser(description="Compute statistics for a dataset")
parser.add_argument("--frequency", type=str, default="daily", help="Frequency of the data (daily or monthly)")
parser.add_argument("--dstart", type=str, default="2002-01-01", help="Start date for the data")
parser.add_argument("--dend", type=str, default="2026-06-30", help="End date for the data")
parser.add_argument("--variable", type=str, default="precipitation", help="Variable to analyze")
parser.add_argument("--do-spatial", action="store_true", help="Compute the spatial consistency maps")
parser.add_argument("--do-p999", action="store_true", help="Compute the climatological P0.1/P99.9 thresholds")
parser.add_argument("--nprocs", type=int, default=None, help="Worker processes for the time series (default: SLURM_CPUS_PER_TASK, else all cores)")
parser.add_argument("--no-tseries", action="store_true", help="Skip the time series (e.g. to update only the spatial consistency)")
parser.add_argument("--clim-stride", type=int, default=1, help="Use every Nth file when pooling the climatology (keeps a daily climatology within memory)")
args = parser.parse_args()
frequency = args.frequency
# Normalise the dates to the compact form used in the aux filenames. The update
# logic compares them as strings against the dates parsed out of an existing
# filename, so "2026-06-30" and "20260630" must not behave differently: the
# dashed form used to compare as "before" every 8-digit date and silently
# triggered a full recalculation, deleting the previous file.
dstart = pd.to_datetime(args.dstart).strftime("%Y%m%d")
dend = pd.to_datetime(args.dend).strftime("%Y%m%d")
variable = args.variable

#  --- script options - we may skip certain parts if needed ----
# ideally do_compute_p999 should only be used in case of a new CDR
do_compute_spatial_consistency = False or args.do_spatial
do_compute_p999 = False or args.do_p999  # compute p99.9. this only needs to be done once, then the thresholds are saved to a file and can be reused. no need to change if more recent files become available
do_compute_stats_tseries = not args.no_tseries  # compute stats timeseries
climatology_start = pd.Timestamp("2002-01-01")
climatology_end   = pd.Timestamp("2022-12-31")  # end of climatology period for threshold computation
cdr_end           = pd.Timestamp("2022-12-31")  # last date of the CDR; later dates are ICDR files

DATASET = "satellite-precipitation"
PRODUCT = "GIRAFE"
ECV = "Precipitation"

variables = ["precipitation"]

# The aux filenames and the variable inside the file use the full variable name;
# the dataset filenames use a short prefix instead.
var_strings = {"precipitation": "precipitation"}
file_prefixes = {"precipitation": "PRE"}
nc_variables = {"precipitation": "precipitation"}
freq_strings = {"daily": "dm", "monthly": "mm"}

var_string = var_strings[variable]
file_prefix = file_prefixes[variable]
nc_var = nc_variables[variable]
freq_string = freq_strings[frequency]

dir_dataset = Path(f"../../../datasets/{PRODUCT}/{frequency}")


def dataset_file(date, freq):
    """Path of the dataset file holding one date at one frequency.

    This is the only place that knows how the dataset lays out its files, so
    both the main file list and the climatology block below go through it.
    GIRAFE keeps the daily files in {year}/{month}/ but the monthly ones
    directly in {year}/, and switches from the CDR to the ICDR naming after
    cdr_end.
    """
    folder = f"{date.year}/{date.month:02d}/" if freq == "daily" else f"{date.year}/"
    suffix = "IMPGS01GL.nc" if date <= cdr_end else "IMPGSI1GL.nc"
    return (f"../../../datasets/{PRODUCT}/{freq}/{folder}"
            f"{file_prefix}{freq_strings[freq]}{date:%Y%m%d}000000120{suffix}")


dates = pd.date_range(dstart, dend, freq="D" if frequency.startswith("daily") else "MS")

fnames = [dataset_file(date, frequency) for date in dates]

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
    
    # The filename carries the variable, so the lookup has to as well: with a
    # bare 'spatial_consistency' the glob never matched an existing file, which
    # made every run a full recalculation and blocked incremental extension.
    filestring = f'spatial_consistency_{var_string}'
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

    new_missing    = ds[var_string].isnull().sum("time").rename("Missing_values")
    new_num_values = ds[var_string].notnull().sum("time").rename("Number_of_values")
    new_max_values = ds[var_string].max("time").rename("Max_value")
    new_min_values = ds[var_string].min("time").rename("Min_value")

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
    outfile = os.path.join(stats_dir, f"{filestring}_{frequency}_{new_start_dt:%Y%m%d}_{new_end_dt:%Y%m%d}.nc")
    print(f"Writing spatial stats to {outfile}")
    summary_ds.to_netcdf(outfile)

    # Release the input files: the time series below forks worker processes, and
    # forking with netCDF/HDF5 handles still open in the parent can deadlock.
    summary_ds.close()
    ds.close()
 


#### compute climatology thresholds for outlier detection
# The thresholds are derived from, and named after, the frequency they will be
# compared against: a threshold pooled from monthly means and applied to daily
# values flags far more than the ~0.1% a P99.9 implies.
climatology_file = Path(stats_dir,f"{var_string}_p999_{frequency}_{climatology_start:%Y%m%d}_{climatology_end:%Y%m%d}.nc")
if do_compute_p999:
    print(f"Recomputing the climatology thresholds in {climatology_file} as requested.")
elif os.path.exists(climatology_file):
    with xr.open_dataset(climatology_file) as _thr:
        complete = {f"{var_string}_p001", f"{var_string}_p999"} <= set(_thr.data_vars)
    if complete:
        print(f"Climatology thresholds file {climatology_file} already exists. It will be reused.")
    else:
        # Files written before the low-tail threshold was introduced only hold
        # the p99.9 values, which would fail the time series below.
        print(f"Climatology thresholds file {climatology_file} has no low-tail threshold. Recomputing.")
        do_compute_p999 = True
elif do_compute_stats_tseries:
    # The time series statistics need the thresholds, so compute them on the fly.
    print(f"Climatology thresholds file {climatology_file} does not exist. It will be computed.")
    do_compute_p999 = True

if do_compute_p999:
    print(f"Computing per-calendar-month P0.1 and P99.9 thresholds for "
          f"{climatology_start:%Y}-{climatology_end:%Y} from the {frequency} files...")

    # Same frequency as the data this threshold will be compared against.
    dates_d = pd.date_range(climatology_start, climatology_end,
                            freq="D" if frequency.startswith("daily") else "MS")

    # Dates and paths are kept paired. Deriving the dates back out of the
    # filenames with a regex only invited a length mismatch as soon as one file
    # was missing, and it hard-coded the filename layout a second time.
    clim_files = [(d, dataset_file(d, frequency)) for d in dates_d]
    clim_files = [(d, f) for d, f in clim_files if Path(f).exists()]
    if args.clim_stride > 1:
        # Pooling every day of a 40-year daily climatology over a 1440x720 grid
        # is tens of GB per calendar month; a stride still leaves hundreds of
        # millions of samples per month, which is ample for a P99.9.
        clim_files = clim_files[::args.clim_stride]
        print(f"Using every {args.clim_stride}th climatology file")
    print(f"Climatology files found: {len(clim_files)} of {len(dates_d)}")

    # --------------------------------------------------------
    # Calculate threshold for each calendar month
    # --------------------------------------------------------

    monthly_p001 = []
    monthly_p999 = []

    for month in range(1, 13):

        month_files = [f for d, f in clim_files if d.month == month]

        print(f"\nProcessing month {month:02d}... {len(month_files)} files")

        if not month_files:
            raise ValueError(f"No climatology files found for calendar month {month:02d}")

        # Open the files lazily with Dask
        ds_clim = xr.open_mfdataset(
            month_files,
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

    # The thresholds are read into plain lists (indexed by month - 1) and the
    # file is closed again: the worker processes are forked further down, and
    # forking with an HDF5/netCDF file still open in the parent is asking for
    # trouble.
    with xr.open_dataset(climatology_file) as threshold_ds:
        monthly_p001 = [float(v) for v in threshold_ds[f"{var_string}_p001"].values]
        monthly_p999 = [float(v) for v in threshold_ds[f"{var_string}_p999"].values]

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

    # ------------------------------------------------------------
    # One file per worker process. The files are independent, so this is
    # embarrassingly parallel; executor.map keeps the results in the order of
    # the input, so the records stay in date order.
    # ------------------------------------------------------------
    nprocs = args.nprocs or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or os.cpu_count()
    # For SRB the variable inside the file is named after the file itself.
    tasks = [(f, var_string, monthly_p001, monthly_p999) for f in fnames]

    print(f"Computing statistics for {len(tasks)} files using {nprocs} processes...")

    records = []
    absent = []
    t_start = time.time()
    report_every = max(1, len(tasks) // 20)

    with ProcessPoolExecutor(max_workers=nprocs) as executor:
        for i, result in enumerate(executor.map(compute_file_stats, tasks, chunksize=16), 1):
            if "absent" in result:
                absent.append(result["absent"])
            else:
                records.append(result)
            if i % report_every == 0 or i == len(tasks):
                rate = i / max(time.time() - t_start, 1e-9)
                eta = (len(tasks) - i) / rate
                print(f"  {i}/{len(tasks)} files ({rate:.1f} files/s, ETA {eta/60:.1f} min)")

    elapsed = time.time() - t_start
    print(f"Processed {len(records)} files in {elapsed/60:.1f} min "
          f"({len(tasks)/max(elapsed, 1e-9):.1f} files/s)")

    for f in absent:
        print(f"File {f} does not exist, skipping.")

    # The files that could not be read should be exactly the ones that were
    # already missing when the run started. Anything else means files became
    # unreadable while the run was in progress (a truncated time series would
    # otherwise be written without any visible error).
    if len(absent) != len(missing_fnames):
        print(f"WARNING: {len(absent)} files were unreadable, but {len(missing_fnames)} "
              f"were missing at startup. The time series is likely incomplete!")

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