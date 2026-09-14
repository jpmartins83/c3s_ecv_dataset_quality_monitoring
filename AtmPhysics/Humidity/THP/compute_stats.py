"""QC statistics for the Tropospheric Humidity Profiles record.

The dataset is monthly and zonally averaged: one file holds a single month on a
36-band latitude by 251-level altitude grid, of which only 0-12 km carries
humidity. The whole record is therefore ~100 MB, which changes the shape of this
script compared with its siblings for the gridded products: there is no need to
stream files past a pool of workers, no need to extend an existing aux file
rather than recompute it, and no need to read the outer tails of a distribution
without holding it. Everything is read into memory in one pass and every
statistic is reduced from that.

What it writes into aux_files/, per release, source and monitored variable:

    zonal_record_<VAR>_<SOURCE>_<REL>_<dstart>_<dend>.nc
        The record itself as one (time, altitude, latitude) cube, cropped to the
        troposphere the product delivers. This is what the gallery plots: a
        latitude-height panel is a time slice of it, a time-altitude Hovmoller
        is its latitude mean, and both the observation-minus-reanalysis and the
        version-to-version differences are one cube minus another. Keeping it
        costs a few MB per variable and saves the gallery re-reading 230 files.

    tseries_stats_<VAR>_<SOURCE>_<REL>_<dstart>_<dend>.nc
        One record per month: the distribution of the latitude-height field,
        its missing fraction, and the fraction of it outside the climatological
        P0.1/P99.9. Same variable names as the gridded galleries use, so the
        plotting code is shared.

    zonal_consistency_<VAR>_<SOURCE>_<REL>_<dstart>_<dend>.nc
        Per grid cell over the whole record: missing count, valid count,
        maximum, minimum. Computed over the full altitude range rather than the
        cropped one, so that the levels the product never fills stay visible.

    <VAR>_p999_<SOURCE>_<REL>_<climstart>_<climend>.nc
        The climatological P0.1/P99.9 per calendar month, pooled over cells and
        years of the reference period, which the outlier fractions above are
        measured against.

Usage:
    python compute_stats.py                          # everything, every release
    python compute_stats.py --release v1 --source RO
    python compute_stats.py --variable Q,Q_stdev --do-p999
"""

from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd
import xarray as xr

# =============================================================================
# Configuration
# =============================================================================

PRODUCT = "THP"
ECV = "Humidity"

DATADIR = Path(f"../../../datasets/{ECV}/{PRODUCT}")
AUX_DIR = Path("aux_files")

# Only 0-12 km carries humidity; the grid runs to 50 km and the rest is empty by
# construction. The cube is cropped to this so that the plots and the statistics
# are about the product rather than about its padding. The consistency maps are
# deliberately not cropped -- see below.
ALT_TOP = 12000.0

# The two delivered product types, and the observation-minus-reanalysis
# difference derived from them. 'tokens' are the source names the delivered
# filenames carry: the a priori model changes with the record, ERA-Interim
# behind the CDR and ERA5 behind the ICDR, so both are matched.
SOURCES = {
    "RO": dict(tokens=("metop",), label="Radio occultation",
               derived=None, operation=None, units=None),
    "MOD": dict(tokens=("erai@metop", "era5@metop"),
                label="Collocated reanalysis",
                derived=None, operation=None, units=None),
    # Observation minus reanalysis, absolutely and relatively. The single most
    # useful QC series for this record: a retrieval change, a sensor change or a
    # reanalysis change shows up here long before it is visible in either
    # product on its own.
    #
    # Both are kept because humidity falls by three orders of magnitude between
    # the surface and 12 km. The absolute difference is dominated by the lowest
    # kilometre and says nothing about the upper troposphere; the relative one
    # weighs every level alike, at the price of amplifying the noise where the
    # reanalysis itself is near zero.
    "DIFF": dict(tokens=(), label="RO minus reanalysis",
                 derived=("RO", "MOD"), operation="difference", units=None),
    "RDIFF": dict(tokens=(), label="RO minus reanalysis, relative",
                  derived=("RO", "MOD"), operation="relative", units="percent"),
}

# One entry per release of the dataset; see download_data.py, which owns the
# same table and the CDS request keys that go with it. Dates come from the
# dataset constraints, read on 2026-09-11.
RELEASES = {
    "v1": dict(start="2006-12-01", end="2025-12-01", label="Version 1",
               product_versions=("1.0", "1.1", "1.2", "1.3")),
    # "v2": dict(start="2006-12-01", end=None, label="Version 2",
    #            product_versions=("2.0",)),
}

# The monitored quantities, keyed as they are in the files.
#
# 'diff' marks the ones whose observation-minus-reanalysis difference means
# something. It is False for three of them by construction, which is worth
# knowing before reading a flat-zero series as a success: the reanalysis file
# repeats the observations' Q_num and Q_samperr unchanged (it is sampled at the
# same occultation locations), reports no measurement uncertainty at all
# (Q_obssig is identically zero), and is by definition 100 % a priori (WQ). The
# gallery cross-checks the first two instead of differencing them.
VARIABLES = {
    "Q":         dict(units="g/kg", diff=True,
                      label="Monthly mean humidity (sampling error corrected)"),
    "Q_stdev":   dict(units="g/kg", diff=True,
                      label="Monthly standard deviation of humidity"),
    "Q_obssig":  dict(units="g/kg", diff=False,
                      label="Measurement uncertainty of the mean"),
    "Q_samperr": dict(units="g/kg", diff=False,
                      label="Sampling error of the mean"),
    "Q_num":     dict(units="1", diff=False,
                      label="Monthly data number"),
    "WQ":        dict(units="percent", diff=False,
                      label="A priori fraction"),
}

# Reference period for the outlier thresholds: the reprocessed part of the
# record (ROM SAF product GRM-29-R1, December 2006 to December 2016). The
# interim record that follows it is what the thresholds are meant to judge, so
# it is kept out of them, exactly as the gridded galleries derive their
# thresholds from the TCDR and not the ICDR.
CLIMATOLOGY_START = pd.Timestamp("2006-12-01")
CLIMATOLOGY_END = pd.Timestamp("2016-12-01")

# Which release's climatology the thresholds come from. Every release is judged
# against this one, so that a change in the outlier fraction from one release to
# the next means the data moved and not the yardstick -- which is the whole
# point of keeping the releases side by side. It also covers the case of a
# release that is published as a short record and so has no reference period of
# its own to pool.
#
# Set to None to give every release its own thresholds instead, which is what
# the gridded galleries in this repository do because they have one release.
THRESHOLD_RELEASE = "v1"

# zgrid_<r|i>hgmet_<source>_<YYYYMM>_<R|I>_<processing>_<version>.nc
FILE_PATTERN = re.compile(
    r"zgrid_(?P<grid>[ri]hgmet)_(?P<source>[a-z0-9@]+)_(?P<date>\d{6})_"
    r"(?P<record>[RI])_(?P<processing>\d{4})_(?P<version>\d{4})\.nc"
)

# =============================================================================


def source_units(variable, source):
    """The units of a variable as one source reports it.

    A derived source can change them: the relative difference is a percentage
    whatever the variable it was formed from.
    """
    return SOURCES[source]["units"] or VARIABLES[variable]["units"]


def collect_files(release, source):
    """Index the files of one release and source by month."""
    root = DATADIR / release / source
    records = []
    for path in sorted(root.rglob("*.nc")) if root.is_dir() else []:
        match = FILE_PATTERN.match(path.name)
        if match is None:
            continue
        records.append({
            "file_path": str(path),
            "time": pd.Timestamp(f"{match.group('date')}01"),
            "record": match.group("record"),
            "version_token": match.group("version"),
        })
    df = pd.DataFrame(records, columns=["file_path", "time", "record", "version_token"])
    return df.sort_values("time").reset_index(drop=True)


def read_cube(files, variables):
    """The listed files as one (time, alt, lat) cube per variable.

    The files are opened one at a time rather than with open_mfdataset: the time
    coordinate is a cftime object in some months and a datetime64 in others, and
    concatenating those through xarray raises. The month is taken from the
    filename, which is unambiguous, and the file's own time stamp is kept only
    as a check (see verify_timestamps below).
    """
    per_time = {name: [] for name in variables}
    times, stamps, alt, lat = [], [], None, None

    for row in files.itertuples():
        with xr.open_dataset(row.file_path, decode_times=False) as ds:
            if alt is None:
                alt, lat = ds["alt"].values, ds["lat"].values
            elif not (np.array_equal(alt, ds["alt"].values)
                      and np.array_equal(lat, ds["lat"].values)):
                raise SystemExit(f"{row.file_path} is on a different grid than "
                                 f"the files before it")
            for name in variables:
                values = ds[name].values
                # (time, alt, lat, lon) with a single time and a single dummy
                # longitude -- the product is a zonal mean, and the lon axis
                # carries one band covering the globe.
                per_time[name].append(np.asarray(values, dtype="float64").reshape(
                    values.shape[1], values.shape[2]))
            stamps.append((float(ds["year"].values[0]), float(ds["month"].values[0])))
        times.append(row.time)

    keep = alt <= ALT_TOP
    cube = xr.Dataset(
        {name: (("time", "alt", "lat"), np.stack(per_time[name])[:, keep, :])
         for name in variables},
        coords={"time": pd.DatetimeIndex(times), "alt": alt[keep], "lat": lat},
    )
    full = xr.Dataset(
        {name: (("time", "alt", "lat"), np.stack(per_time[name]))
         for name in variables},
        coords={"time": pd.DatetimeIndex(times), "alt": alt, "lat": lat},
    )
    return cube, full, stamps


def verify_timestamps(files, stamps):
    """Report any file whose own year/month disagrees with its filename.

    The filename is what the whole workflow keys on, so a mismatch would make
    every statistic land in the wrong month without anything else noticing.
    """
    wrong = []
    for row, (year, month) in zip(files.itertuples(), stamps):
        if (year, month) != (row.time.year, row.time.month):
            wrong.append(f"{Path(row.file_path).name}: file says "
                         f"{int(year)}-{int(month):02d}")
    return wrong


def latitude_weights(shape, lat):
    """cos(latitude) weights for an (alt, lat) field.

    The bands are equal in degrees, so they are not equal in area; every
    statistic below is weighted, as in the gridded galleries. Altitude levels
    are equally spaced and are weighted equally.
    """
    return np.broadcast_to(np.cos(np.deg2rad(lat))[None, :], shape)


def distribution_stats(values, weights, w_total, p001, p999):
    """The statistics of one latitude-height field.

    The variable names are the ones the gridded QC galleries write, so the
    gallery's time-series figure is the same code for every dataset.
    """
    valid = ~np.isnan(values)
    v = values[valid]
    w = weights[valid]

    if v.size == 0:
        return {
            "mean": np.nan, "median": np.nan, "std": np.nan,
            "minimum": np.nan, "maximum": np.nan, "p01": np.nan, "p99": np.nan,
            "missing": int(values.size), "missing_fraction": 1.0,
            "negative_outliers_fraction": 0.0, "positive_outliers_fraction": 0.0,
        }

    w_valid = w.sum()
    mean = (v * w).sum() / w_valid
    std = np.sqrt((w * (v - mean) ** 2).sum() / w_valid)
    p01, median, p99 = np.percentile(v, [1, 50, 99])

    return {
        "mean": float(mean),
        "median": float(median),
        "std": float(std),
        "minimum": float(v.min()),
        "maximum": float(v.max()),
        "p01": float(p01),
        "p99": float(p99),
        "missing": int((~valid).sum()),
        "missing_fraction": float(weights[~valid].sum() / w_total),
        "negative_outliers_fraction": float(w[v < p001].sum() / w_total),
        "positive_outliers_fraction": float(w[v > p999].sum() / w_total),
    }


def aux_path(kind, variable, source, release, start, end):
    return AUX_DIR / (f"{kind}_{variable}_{source}_{release}_"
                      f"{start:%Y%m%d}_{end:%Y%m%d}.nc")


def threshold_path(variable, source, release):
    return AUX_DIR / (f"{variable}_p999_{source}_{release}_"
                      f"{CLIMATOLOGY_START:%Y%m%d}_{CLIMATOLOGY_END:%Y%m%d}.nc")


def threshold_release(release):
    """The release whose climatology this one is judged against."""
    return THRESHOLD_RELEASE or release


def drop_superseded(kind, variable, source, release, keep):
    """Delete aux files of the same kind whose date range this run replaces.

    The filename carries the range it covers, so a shorter record left behind
    from an earlier run would still match the gallery's glob and could be picked
    up instead of this one.
    """
    for path in AUX_DIR.glob(f"{kind}_{variable}_{source}_{release}_*.nc"):
        if path != keep:
            path.unlink()


def compute_thresholds(cube, variable, source, release):
    """Climatological P0.1/P99.9 per calendar month, pooled over the reference period.

    Pooled over cells and years the way the gridded galleries do it, so the
    fraction beyond the threshold is ~0.1 % per tail by construction and any
    month well above that is the signal.
    """
    reference = cube[variable].sel(time=slice(CLIMATOLOGY_START, CLIMATOLOGY_END))
    if reference.sizes["time"] == 0:
        raise SystemExit(
            f"{variable} {source} {release}: no files inside the threshold "
            f"reference period {CLIMATOLOGY_START:%Y-%m} to "
            f"{CLIMATOLOGY_END:%Y-%m}; download it, or move the period.")

    p001, p999 = [], []
    for month in range(1, 13):
        pool = reference.sel(time=reference["time.month"] == month).values.ravel()
        pool = pool[np.isfinite(pool)]
        if pool.size == 0:
            p001.append(np.nan)
            p999.append(np.nan)
            continue
        low, high = np.percentile(pool, [0.1, 99.9])
        p001.append(float(low))
        p999.append(float(high))

    out = xr.Dataset(
        {f"{variable}_p001": (["month"], p001), f"{variable}_p999": (["month"], p999)},
        coords={"month": np.arange(1, 13)},
    )
    out[f"{variable}_p001"].attrs = {
        "long_name": "Monthly climatological 0.1th percentile", "quantile": 0.001,
        "reference_period": f"{CLIMATOLOGY_START:%Y-%m} to {CLIMATOLOGY_END:%Y-%m}",
        "units": source_units(variable, source),
    }
    out[f"{variable}_p999"].attrs = {
        "long_name": "Monthly climatological 99.9th percentile", "quantile": 0.999,
        "reference_period": f"{CLIMATOLOGY_START:%Y-%m} to {CLIMATOLOGY_END:%Y-%m}",
        "units": source_units(variable, source),
    }
    path = threshold_path(variable, source, release)
    out.to_netcdf(path)
    print(f"  thresholds -> {path}")
    return p001, p999


def load_thresholds(variable, source, release):
    path = threshold_path(variable, source, release)
    if not path.exists():
        return None
    with xr.open_dataset(path) as ds:
        return ([float(v) for v in ds[f"{variable}_p001"].values],
                [float(v) for v in ds[f"{variable}_p999"].values])


def write_record(cube, full, variable, source, release, start, end):
    path = aux_path("zonal_record", variable, source, release, start, end)
    out = cube[[variable]].copy()
    out[variable].attrs = {
        "long_name": VARIABLES[variable]["label"],
        "units": source_units(variable, source),
        "altitude_range": f"0 to {ALT_TOP:.0f} m",
        "source": SOURCES[source]["label"],
        "release": release,
        "altitude_levels_delivered": int(full.sizes["alt"]),
    }
    # Written back at the precision it was delivered in, and compressed: the
    # cube is a convenience copy of the record, and at float64 the aux files
    # would come to more than the dataset they are derived from.
    out.to_netcdf(path, encoding={variable: {"dtype": "float32",
                                             "zlib": True, "complevel": 4}})
    drop_superseded("zonal_record", variable, source, release, path)
    print(f"  record     -> {path}")


def write_consistency(full, variable, source, release, start, end):
    """Per-cell aggregates over the whole record, on the full altitude grid.

    Not cropped to ALT_TOP on purpose: this is the panel that has to show that
    the product delivers nothing above 12 km, and a cropped version would hide
    the one thing it is there to report.
    """
    da = full[variable]
    out = xr.Dataset({
        "Missing_values": da.isnull().sum("time"),
        "Number_of_values": da.notnull().sum("time"),
        "Max_value": da.max("time", skipna=True),
        "Min_value": da.min("time", skipna=True),
    })
    for name in out.data_vars:
        out[name].attrs = {"units": "1" if "values" in name.lower()
                           else source_units(variable, source)}
    out.attrs = {"records": int(da.sizes["time"]),
                 "source": SOURCES[source]["label"], "release": release}
    path = aux_path("zonal_consistency", variable, source, release, start, end)
    out.to_netcdf(path)
    drop_superseded("zonal_consistency", variable, source, release, path)
    print(f"  consistency-> {path}")


def write_tseries(cube, variable, source, release, start, end, thresholds):
    p001, p999 = thresholds
    lat = cube["lat"].values
    records = []
    for stamp in cube["time"].values:
        field = cube[variable].sel(time=stamp).values
        weights = latitude_weights(field.shape, lat)
        month = pd.Timestamp(stamp).month
        records.append({
            "time": pd.Timestamp(stamp),
            **distribution_stats(field, weights, weights.sum(),
                                 p001[month - 1], p999[month - 1]),
        })

    stats = pd.DataFrame(records).set_index("time")
    out = xr.Dataset.from_dataframe(stats)
    out.attrs = {"units": source_units(variable, source),
                 "source": SOURCES[source]["label"], "release": release,
                 "threshold_reference_period":
                     f"{CLIMATOLOGY_START:%Y-%m} to {CLIMATOLOGY_END:%Y-%m}"}
    path = aux_path("tseries_stats", variable, source, release, start, end)
    out.to_netcdf(path)
    drop_superseded("tseries_stats", variable, source, release, path)
    print(f"  tseries    -> {path}")


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


def process(release, source, variables, dstart, dend, do_p999):
    """Every aux file for one release and one source."""
    derived = SOURCES[source]["derived"]

    if derived is None:
        files = collect_files(release, source)
        if files.empty:
            print(f"{release} {source}: no files found under "
                  f"{DATADIR / release / source}, skipping.")
            return
        files = files[(files["time"] >= dstart) & (files["time"] <= dend)]
        print(f"{release} {source}: {len(files)} files, "
              f"{files['time'].min():%Y-%m} to {files['time'].max():%Y-%m}")
        cube, full, stamps = read_cube(files, variables)
        for message in verify_timestamps(files, stamps):
            print(f"  WARNING: timestamp mismatch, {message}")
    else:
        # The difference is built from the two cubes just written, aligned on
        # the months both sources deliver. A month present in only one of them
        # cannot be differenced and is dropped rather than carried as NaN, so
        # that the series and the integrity table disagree visibly instead of
        # quietly.
        left, right = derived
        cubes, fulls = [], []
        for other in (left, right):
            other_files = collect_files(release, other)
            if other_files.empty:
                print(f"{release} {source}: {other} has no files, skipping.")
                return
            other_files = other_files[(other_files["time"] >= dstart)
                                      & (other_files["time"] <= dend)]
            one, one_full, _ = read_cube(other_files, variables)
            cubes.append(one)
            fulls.append(one_full)
        cube_left, cube_right = xr.align(*cubes, join="inner")
        full_left, full_right = xr.align(*fulls, join="inner")
        if SOURCES[source]["operation"] == "relative":
            # Per cent of the reanalysis value. Where the reanalysis is itself
            # near zero this is large and noisy rather than wrong, so it is left
            # in and the panels use robust colour limits; only an exact zero,
            # which would be an infinity, is dropped.
            denominator = cube_right.where(cube_right != 0)
            cube = 100.0 * (cube_left - cube_right) / denominator
            full = 100.0 * (full_left - full_right) / full_right.where(full_right != 0)
        else:
            cube, full = cube_left - cube_right, full_left - full_right
        dropped = cubes[0].sizes["time"] - cube.sizes["time"]
        print(f"{release} {source}: {cube.sizes['time']} months differenced"
              + (f", {dropped} month(s) present in only one source" if dropped else ""))

    if cube.sizes["time"] == 0:
        print(f"{release} {source}: nothing inside {dstart:%Y-%m} to {dend:%Y-%m}.")
        return

    start = pd.Timestamp(cube["time"].values.min())
    end = pd.Timestamp(cube["time"].values.max())

    for variable in variables:
        print(f" {variable}:")
        write_record(cube, full, variable, source, release, start, end)
        write_consistency(full, variable, source, release, start, end)

        reference = threshold_release(release)
        thresholds = (None if (do_p999 and reference == release)
                      else load_thresholds(variable, source, reference))
        if thresholds is None:
            if reference != release:
                raise SystemExit(
                    f"{release} {source} {variable}: the thresholds come from "
                    f"release '{reference}' (see THRESHOLD_RELEASE) and it has "
                    f"none yet. Run compute_stats.py --release {reference} "
                    f"first, or set THRESHOLD_RELEASE to None to give every "
                    f"release its own.")
            thresholds = compute_thresholds(cube, variable, source, release)
        elif reference != release:
            print(f"  thresholds <- {threshold_path(variable, source, reference).name}")
        write_tseries(cube, variable, source, release, start, end, thresholds)


def main():
    parser = argparse.ArgumentParser(
        description="QC statistics for the Tropospheric Humidity Profiles record")
    parser.add_argument("--release", action="append", default=[],
                        help="Release (default: every release with files on disk)")
    parser.add_argument("--source", action="append", default=[],
                        help="RO, MOD, DIFF (default: all three)")
    parser.add_argument("--variable", action="append", default=[],
                        help="Monitored variable (default: all of them)")
    parser.add_argument("--dstart", default=None, help="First month, YYYY-MM-DD")
    parser.add_argument("--dend", default=None, help="Last month, YYYY-MM-DD")
    parser.add_argument("--do-p999", action="store_true",
                        help="Re-derive the climatological P0.1/P99.9 thresholds "
                             "even if the file already exists. Needed when the "
                             "reference period or a release's retrieval changes, "
                             "not when months are appended.")
    args = parser.parse_args()

    releases = resolve(args.release, RELEASES, "release")
    sources = resolve(args.source, SOURCES, "source")
    variables = resolve(args.variable, VARIABLES, "variable")

    AUX_DIR.mkdir(parents=True, exist_ok=True)

    for release in releases:
        config = RELEASES[release]
        dstart = pd.Timestamp(args.dstart or config["start"])
        dend = pd.Timestamp(args.dend or config["end"] or "2100-01-01")
        for source in sources:
            wanted = variables
            if SOURCES[source]["derived"] is not None:
                wanted = [v for v in variables if VARIABLES[v]["diff"]]
                skipped = [v for v in variables if v not in wanted]
                if skipped:
                    print(f"{release} {source}: {', '.join(skipped)} not "
                          f"differenced (see VARIABLES).")
                if not wanted:
                    continue
            process(release, source, wanted, dstart, dend, args.do_p999)


if __name__ == "__main__":
    main()
