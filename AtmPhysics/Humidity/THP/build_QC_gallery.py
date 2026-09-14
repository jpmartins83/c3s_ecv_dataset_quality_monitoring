"""Build the Tropospheric Humidity Profiles QC gallery.

Same inspections as the gridded QC galleries in this repository -- integrity,
metadata, a time series per monitored quantity with climatological outliers, a
per-month browser and a record-wide consistency panel -- with two differences
that come from the data rather than from taste:

* The product is a zonal mean, so there is nothing to draw on a map. Every panel
  that is a map elsewhere is a **latitude-height** section here: latitude across,
  altitude up, one panel per month. The record-wide consistency panel is the same
  section with the whole record reduced into it, and a third panel type, the
  time-altitude Hovmoller, replaces nothing in the gridded galleries but is the
  natural way to see a 19-year record of profiles at a glance.

* The dataset ships the radio-occultation retrievals and the reanalysis
  collocated at the same occultation points, so obs-minus-model is available as
  a source in its own right, absolutely and in per cent.

Releases are the gallery's primary axis. Everything -- data directories, aux
files, panels, the tab row at the top of the page -- is keyed by release, so
version 2 is a folder next to version 1 and a row in the RELEASES table, and the
release-difference panel picks it up with nothing else to change.
"""

from functools import lru_cache
from pathlib import Path
import argparse
import io
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, to_hex
import numpy as np
import pandas as pd
import xarray as xr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import earthkit.data as ekd

from sites.sdk.sites import Site, Authenticator


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DOMAIN = "AtmPhysics"
ECV = "Humidity"
PRODUCT = "THP"
DATASET_TITLE = "Tropospheric Humidity Profiles"

DATADIR = Path(f"../../../datasets/{ECV}/{PRODUCT}")
AUX_DIR = Path("aux_files")

GALLERY_DIR = Path(f"{PRODUCT}_QC_gallery")
PROFILES_DIR = GALLERY_DIR / "profiles"
CONSISTENCY_DIR = GALLERY_DIR / "consistency"
HOVMOLLER_DIR = GALLERY_DIR / "hovmoller"
RELEASE_DIFF_DIR = GALLERY_DIR / "release_diff"
HTML_NAME = "QC_timeseries.html"
HTML_PATH = GALLERY_DIR / HTML_NAME
TOP_LEVEL_HTML = Path(HTML_NAME)

UPLOAD_TARGET = f"dataset_qc/{DOMAIN}/{ECV}/{PRODUCT}/"
UPLOAD_SPACE = "cxjo"
UPLOAD_SITE = "ecv-info"

# One entry per release of the dataset, and the primary axis of the whole
# gallery. compute_stats.py and download_data.py carry the same table; this one
# adds only what the page needs. A release with no files on disk is dropped from
# the page rather than shown empty, so uncommenting v2 before downloading it
# changes nothing.
RELEASES = {
    "v1": dict(label="Version 1", start="2006-12-01", end="2025-12-01",
               product_versions=("1.0", "1.1", "1.2", "1.3")),
    # "v2": dict(label="Version 2", start="2006-12-01", end=None,
    #            product_versions=("2.0",)),
}

# Which release pairs get a difference panel. Auto-derived: every consecutive
# pair of releases present on disk, newer minus older.
SOURCES = {
    "RO": dict(label="Radio occultation", derived=None, units=None),
    "MOD": dict(label="Collocated reanalysis", derived=None, units=None),
    "DIFF": dict(label="RO − reanalysis", derived=("RO", "MOD"), units=None),
    "RDIFF": dict(label="RO − reanalysis (relative)", derived=("RO", "MOD"),
                  units="percent"),
}

VARIABLES = {
    "Q":         dict(units="g/kg", diff=True,
                      label="Monthly mean humidity (sampling error corrected)"),
    "Q_stdev":   dict(units="g/kg", diff=True,
                      label="Monthly standard deviation of humidity"),
    "Q_obssig":  dict(units="g/kg", diff=False,
                      label="Measurement uncertainty of the mean"),
    "Q_samperr": dict(units="g/kg", diff=False,
                      label="Sampling error of the mean"),
    "Q_num":     dict(units="1", diff=False, label="Monthly data number"),
    "WQ":        dict(units="percent", diff=False, label="A priori fraction"),
}

# The browser covers the last N months of each release.
N_GALLERY_MONTHS = 6

# zgrid_<r|i>hgmet_<source>_<YYYYMM>_<R|I>_<processing>_<version>.nc
FILE_PATTERN = re.compile(
    r"zgrid_(?P<grid>[ri]hgmet)_(?P<source>[a-z0-9@]+)_(?P<date>\d{6})_"
    r"(?P<record>[RI])_(?P<processing>\d{4})_(?P<version>\d{4})\.nc"
)

# Cross-checks between the two product types. The reanalysis file is sampled at
# the occultation points, so three of its fields are not independent data at
# all, and one is a constant. Checking that is worth more than differencing
# them: if any of these stops holding, the pairing of the two product types has
# changed and every obs-minus-model number on this page changes meaning.
CROSS_CHECKS = [
    dict(variable="Q_num", test="equal",
         detail="The reanalysis is sampled at the occultation locations, so its "
                "observation count must repeat the RO one exactly."),
    dict(variable="Q_samperr", test="equal",
         detail="Sampling error follows from the sampling pattern, which the two "
                "product types share, so it must be identical."),
    dict(variable="Q_obssig", test="zero", side="MOD",
         detail="The reanalysis carries no measurement uncertainty; the field is "
                "expected to be identically zero."),
    dict(variable="WQ", test="constant", side="MOD", value=100.0,
         detail="The reanalysis is by definition entirely a priori, so its "
                "a priori fraction is expected to be 100 % everywhere."),
]

# -----------------------------------------------------------------------------
# Colour, as in the sibling galleries: white means "no data" and nothing else.
# Here it matters more than on a map -- the product delivers nothing above 12 km
# and nothing where too few occultations fell, so a panel is part empty by
# construction and a near-white lowest class would read as dry air.
# -----------------------------------------------------------------------------
NAN_COLOR = "white"
WHITE_DISTANCE = 0.40


@lru_cache(maxsize=None)
def qc_colormap(name):
    """`name` with its near-white ends removed and NaN painted white."""
    rgba = plt.get_cmap(name)(np.linspace(0, 1, 256))
    distance = np.sqrt(((1.0 - rgba[:, :3]) ** 2).sum(axis=1))
    keep = np.flatnonzero(distance >= WHITE_DISTANCE)
    if keep.size == 0:
        raise ValueError(f"colormap '{name}' is white throughout")
    first, last = keep[0], keep[-1]
    if (distance[first:last + 1] < WHITE_DISTANCE).any():
        raise ValueError(f"colormap '{name}' passes through white in the middle; "
                         f"trimming cannot fix that, pick another one")
    cmap = ListedColormap(rgba[first:last + 1], name=f"{name}_nowhite")
    cmap.set_bad(NAN_COLOR)
    return cmap


# Humidity falls by three orders of magnitude between the surface and 12 km, so
# the moist variables are drawn on a geometric ladder of levels: a linear scale
# resolves the lowest kilometre and leaves the rest of the troposphere in one
# flat class. 'levels=None' derives a robust linear ladder from the data.
#
# A diverging quantity (the differences, and the sampling error, which is signed)
# gets a dark-centred palette: every diverging ColorBrewer ramp is white in the
# middle, which is the one colour reserved for missing data here.
PANEL_STYLES = {
    "Q": dict(cmap="viridis",
              levels=[0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 4, 8, 12, 20]),
    "Q_stdev": dict(cmap="cividis",
                    levels=[0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 4]),
    "Q_obssig": dict(cmap="magma",
                     levels=[1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1]),
    "Q_samperr": dict(cmap="managua_r", levels=None, symmetric=True),
    "Q_num": dict(cmap="turbo", levels=[0, 1, 10, 50, 100, 250, 500, 1000, 1500, 2500]),
    "WQ": dict(cmap="inferno", levels=np.arange(0, 101, 10)),
}

# Overrides for a derived source: the difference of a variable is not on the
# scale of the variable.
DERIVED_STYLES = {
    "DIFF": dict(cmap="managua_r", levels=None, symmetric=True),
    "RDIFF": dict(cmap="managua_r", levels=None, symmetric=True),
}

# The record-wide consistency statistics, as compute_stats.py writes them.
CONSISTENCY_STATS = {
    "Missing_values": dict(
        title="Number of Months Missing", label="# of months missing",
        cmap="magma_r", levels=None,
        # A fixed ladder at the low end: one or two missing months matter as
        # much as fifty, and a single wide class would hide them.
        low_levels=[0, 1, 2, 3, 5, 10, 20, 50, 100],
    ),
    "Number_of_values": dict(title="Number of Months Present",
                             label="# of months present", cmap="viridis", levels=None),
    "Max_value": dict(title="Maximum Value", label="Maximum [{units}]",
                      cmap="inferno", levels=None),
    "Min_value": dict(title="Minimum Value", label="Minimum [{units}]",
                      cmap="viridis", levels=None),
}


# -----------------------------------------------------------------------------
# Reading what is on disk
# -----------------------------------------------------------------------------
def collect_files(release, source):
    """Index the delivered files of one release and source by month."""
    root = DATADIR / release / source
    records = []
    for path in sorted(root.rglob("*.nc")) if root.is_dir() else []:
        match = FILE_PATTERN.match(path.name)
        if match is None:
            continue
        records.append({
            "file_path": str(path),
            "file_date": match.group("date"),
            "time": pd.Timestamp(f"{match.group('date')}01"),
            "record": "CDR" if match.group("record") == "R" else "ICDR",
            "processing": match.group("processing"),
            "version_token": match.group("version"),
        })
    return pd.DataFrame(records, columns=["file_path", "file_date", "time",
                                          "record", "processing", "version_token"])


def releases_on_disk():
    """The configured releases that actually have files, in configured order."""
    present = []
    for release in RELEASES:
        if any(not collect_files(release, s).empty
               for s in SOURCES if SOURCES[s]["derived"] is None):
            present.append(release)
    return present


def delivered_sources(release):
    return [s for s in SOURCES
            if SOURCES[s]["derived"] is None and not collect_files(release, s).empty]


def variables_for(source):
    """The monitored variables a source carries."""
    if SOURCES[source]["derived"] is None:
        return list(VARIABLES)
    return [v for v in VARIABLES if VARIABLES[v]["diff"]]


def find_aux(pattern):
    """Most recent aux_files/ file matching pattern, if any."""
    matches = sorted(AUX_DIR.glob(pattern))
    return matches[-1] if matches else None


def aux_record(variable, source, release):
    return find_aux(f"zonal_record_{variable}_{source}_{release}_*.nc")


def aux_tseries(variable, source, release):
    return find_aux(f"tseries_stats_{variable}_{source}_{release}_*.nc")


def aux_consistency(variable, source, release):
    return find_aux(f"zonal_consistency_{variable}_{source}_{release}_*.nc")


def aux_thresholds(variable, source, release):
    return find_aux(f"{variable}_p999_{source}_{release}_*.nc")


@lru_cache(maxsize=None)
def load_record(variable, source, release):
    """The (time, alt, lat) cube of one variable, or None if it is not there."""
    path = aux_record(variable, source, release)
    if path is None:
        return None
    with xr.open_dataset(path) as ds:
        return ds[variable].load()


def source_units(variable, source):
    return SOURCES[source]["units"] or VARIABLES[variable]["units"]


# -----------------------------------------------------------------------------
# Tables
# -----------------------------------------------------------------------------
def integrity_report(release, source, df):
    """Expected against delivered months for one release and source."""
    config = RELEASES[release]
    if df.empty:
        return {"files_found": 0, "expected": 0, "existing": 0, "missing": 0,
                "missing_dates": [], "period": ""}

    start = pd.Timestamp(config["start"])
    end = pd.Timestamp(config["end"]) if config["end"] else df["time"].max()
    expected = pd.date_range(start, end, freq="MS")
    expected_keys = {d.strftime("%Y%m") for d in expected}
    present = set(df["file_date"])
    missing = sorted(expected_keys - present)
    return {
        "files_found": len(df),
        "expected": len(expected_keys),
        "existing": len(expected_keys & present),
        "missing": len(missing),
        # The record is monthly, so even a badly incomplete release lists short.
        "missing_dates": [f"{m[:4]}-{m[4:]}" for m in missing[:120]],
        "period": f"{start:%Y-%m} – {end:%Y-%m}",
        "extra": sorted(present - expected_keys)[:24],
    }


def version_inventory(df):
    """What product versions the delivered files carry, and over which months.

    The point of the panel, and the reason the workflow is built around a
    release axis: the CDS release is one thing, the product_version inside the
    files is another, and this record already mixes several of the latter -- the
    reprocessed part is one version and the interim part has been re-issued
    more than once. A new release shows up here first.
    """
    if df.empty:
        return []
    rows = []
    grouped = df.groupby(["record", "processing", "version_token"], sort=True)
    for (record, processing, token), block in grouped:
        # The version is read out of a file of the group rather than parsed off
        # its name: the trailing filename token tracks the product version but
        # the mapping between them is the provider's business, and guessing it
        # would quietly mislabel a release that numbers itself differently.
        first_file = block.sort_values("time").iloc[0]["file_path"]
        with xr.open_dataset(first_file, decode_times=False) as ds:
            version = str(ds.attrs.get("product_version", "unknown"))
        rows.append({
            "record": record,
            "processing": processing,
            "version": version,
            "token": token,
            "count": len(block),
            "first": f"{block['time'].min():%Y-%m}",
            "last": f"{block['time'].max():%Y-%m}",
        })
    return sorted(rows, key=lambda r: r["first"])


def version_boundaries(release, source):
    """Where the delivered product version changes, as (month, label).

    The record is reprocessed in chunks -- the interim part of this one has been
    re-issued four times so far -- and a chunk boundary is the first thing to
    suspect when a series steps. Marking them on the panels turns "there is a
    jump in 2021" into "there is a jump where the processing changed".
    """
    underlying = SOURCES[source]["derived"][0] if SOURCES[source]["derived"] else source
    rows = version_inventory(collect_files(release, underlying))
    boundaries = []
    for previous, current in zip(rows, rows[1:]):
        boundaries.append({
            "time": f"{current['first']}-01",
            "label": f"{current['record']} v{current['version']}",
            "from": f"{previous['record']} v{previous['version']}",
        })
    return boundaries


def attribute_report(path):
    """The global attributes that identify a file's provenance."""
    wanted = ["product_id", "product_acronym", "product_version", "software_name",
              "software_version", "processing_software", "processing_center",
              "processing_date", "institution", "history"]
    with xr.open_dataset(path, decode_times=False) as ds:
        return {key: str(ds.attrs[key]) for key in wanted if key in ds.attrs}


def crosscheck_report(release):
    """Run the between-product-type checks over the whole record."""
    results = []
    for check in CROSS_CHECKS:
        variable, test = check["variable"], check["test"]
        row = {"variable": variable, "test": test, "detail": check["detail"]}
        if test == "equal":
            left = load_record(variable, "RO", release)
            right = load_record(variable, "MOD", release)
            if left is None or right is None:
                row.update(verdict="not checked",
                           note="one of the two product types has no aux record")
            else:
                left, right = xr.align(left, right, join="inner")
                delta = np.abs(left.values - right.values)
                worst = float(np.nanmax(delta)) if np.isfinite(delta).any() else 0.0
                differing = int(np.nansum(delta > 0))
                row.update(
                    verdict="pass" if differing == 0 else "FAIL",
                    note=(f"identical in all {left.size} cells of {left.sizes['time']} months"
                          if differing == 0 else
                          f"{differing} cells differ, worst |difference| {worst:.6g}"),
                )
        else:
            side = check["side"]
            values = load_record(variable, side, release)
            if values is None:
                row.update(verdict="not checked", note=f"{side} has no aux record")
            else:
                target = 0.0 if test == "zero" else float(check["value"])
                finite = np.isfinite(values.values)
                off = int(np.sum(finite & (values.values != target)))
                row.update(
                    verdict="pass" if off == 0 else "FAIL",
                    note=(f"{side} is {target:g} in all {int(finite.sum())} valid cells"
                          if off == 0 else
                          f"{off} of {int(finite.sum())} valid cells are not {target:g} "
                          f"(range {np.nanmin(values.values):.6g} to "
                          f"{np.nanmax(values.values):.6g})"),
                )
        results.append(row)
    return results


def metadata_report(release, source, df):
    """Fieldlist, xarray summary and provenance attributes of the newest file."""
    if df.empty:
        return f"No {source} files found for {release}."
    fname = df.sort_values("time").iloc[-1]["file_path"]
    result = {"file": fname, "attributes": attribute_report(fname)}
    try:
        fieldlist = ekd.from_source("file", fname).to_fieldlist()
        result["fieldlist"] = summarise_fieldlist(fieldlist.ls())
    except Exception as exception:
        with xr.open_dataset(fname, decode_times=False) as ds:
            result["fieldlist"] = (f"earthkit could not index this file "
                                   f"({type(exception).__name__}: {exception}).\n"
                                   f"Variables: " + ", ".join(ds.data_vars))
    buf = io.StringIO()
    with xr.open_dataset(fname, decode_times=False) as ds:
        ds.info(buf=buf)
    result["xarray_info"] = buf.getvalue()
    return result


def summarise_fieldlist(fls):
    """Field listing with identical rows collapsed into one, counted.

    earthkit builds one field per 2-D slice, so a variable with an extra
    dimension is listed once per slice; the count carries what the repetition
    did. Listings whose rows are already distinct are left untouched.
    """
    if fls.empty:
        return fls.to_string(index=False)
    rows = fls.astype(str)
    if not rows.duplicated().any():
        return fls.to_string(index=False)
    return rows.value_counts(sort=False).reset_index(name="fields").to_string(index=False)


# -----------------------------------------------------------------------------
# Panels
# -----------------------------------------------------------------------------
def robust_levels(values, n=11, symmetric=False):
    """A linear ladder of levels from the data, or None if there is no data."""
    finite = np.asarray(values, dtype="float64")
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    if symmetric:
        # Centred on zero, so that the sign of a difference is readable off the
        # colour rather than off the colour bar.
        span = float(np.percentile(np.abs(finite), 99))
        if span <= 0:
            return None
        step = nice_step(2 * span / (n - 1))
        half = step * ((n - 1) // 2)
        return np.arange(-half, half + step / 2, step)
    lo, hi = np.percentile(finite, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    step = nice_step((hi - lo) / (n - 1))
    start = np.floor(lo / step) * step
    return np.arange(start, start + step * n + step / 2, step)


def nice_step(step):
    if not np.isfinite(step) or step <= 0:
        return 1.0
    magnitude = 10.0 ** np.floor(np.log10(step))
    for nice in (1, 2, 2.5, 5, 10):
        if step <= nice * magnitude:
            return nice * magnitude
    return 10 * magnitude


def panel_style(variable, source):
    style = dict(PANEL_STYLES.get(variable, dict(cmap="viridis", levels=None)))
    style.update(DERIVED_STYLES.get(source, {}))
    return style


def draw_section(ax, x, y, values, style, values_for_levels=None):
    """One latitude-height (or time-height) section, with white for missing.

    Returns the mesh and the class boundaries it was drawn with, so the colour
    bar can tick every boundary: several of these scales are geometric, and a
    colour bar left to choose its own round numbers prints the 0.003, 0.01,
    0.03 end of the ladder as three zeroes.
    """
    levels = style.get("levels")
    if levels is None:
        levels = robust_levels(values if values_for_levels is None else values_for_levels,
                               symmetric=style.get("symmetric", False))
    cmap = qc_colormap(style.get("cmap", "viridis"))
    ax.set_facecolor(NAN_COLOR)
    if levels is None or len(levels) < 2:
        return ax.pcolormesh(x, y, values, cmap=cmap, shading="nearest"), None
    norm = BoundaryNorm(np.asarray(levels, dtype="float64"), cmap.N, extend="both")
    mesh = ax.pcolormesh(x, y, values, cmap=cmap, norm=norm, shading="nearest")
    return mesh, [float(v) for v in levels]


def add_colorbar(fig, ax, mesh, levels, label):
    """A colour bar ticked at the class boundaries, printed as written."""
    bar = fig.colorbar(mesh, ax=ax, label=label, extend="both", pad=0.02,
                       **({"ticks": levels} if levels else {}))
    if levels:
        bar.ax.set_yticklabels([f"{v:g}" for v in levels])
    return bar


def save_profile_panel(cube, stamp, output_path, title, units, style):
    """Latitude-height section of one month."""
    field = cube.sel(time=stamp)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    mesh, levels = draw_section(ax, cube["lat"].values, cube["alt"].values / 1000.0,
                                field.values, style, values_for_levels=cube.values)
    ax.set_xlabel("Latitude [degrees north]")
    ax.set_ylabel("Altitude [km]")
    ax.set_xlim(-90, 90)
    ax.set_xticks(np.arange(-90, 91, 30))
    ax.grid(linewidth=0.3, alpha=0.4)
    ax.set_title(title)
    add_colorbar(fig, ax, mesh, levels, units)
    fig.savefig(output_path, dpi=140, bbox_inches="tight", facecolor=NAN_COLOR)
    plt.close(fig)


def save_consistency_panel(nc_path, statistic, output_path, title, units):
    """Latitude-height section of one record-wide statistic."""
    cfg = CONSISTENCY_STATS[statistic]
    with xr.open_dataset(nc_path) as ds:
        if statistic not in ds.data_vars:
            return False
        da = ds[statistic].load()

    values = da.values
    levels = cfg.get("levels")
    if levels is None:
        levels = robust_levels(values)
        if levels is not None:
            levels = [float(x) for x in levels]
    low_levels = cfg.get("low_levels")
    if low_levels is not None:
        upper = [float(x) for x in (levels or []) if x > low_levels[-1]]
        levels = [float(x) for x in low_levels] + upper

    style = dict(cmap=cfg["cmap"], levels=levels)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    mesh, levels = draw_section(ax, da["lat"].values, da["alt"].values / 1000.0,
                                values, style)
    ax.set_xlabel("Latitude [degrees north]")
    ax.set_ylabel("Altitude [km]")
    ax.set_xlim(-90, 90)
    ax.set_xticks(np.arange(-90, 91, 30))
    ax.grid(linewidth=0.3, alpha=0.4)
    ax.set_title(title)
    add_colorbar(fig, ax, mesh, levels, cfg["label"].format(units=units))
    fig.savefig(output_path, dpi=140, bbox_inches="tight", facecolor=NAN_COLOR)
    plt.close(fig)
    return True


def mark_boundaries(ax, boundaries):
    """Vertical markers at the processing-version changes of a time axis."""
    for boundary in boundaries:
        when = pd.Timestamp(boundary["time"])
        ax.axvline(when, color="white", linewidth=1.6, alpha=0.9)
        ax.axvline(when, color="black", linewidth=0.8, linestyle="--", alpha=0.9)
        ax.annotate(boundary["label"], xy=(when, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(3, -11), textcoords="offset points",
                    fontsize=7, rotation=90, va="top", ha="left", color="black")


def save_hovmoller_panel(cube, output_path, title, units, style, boundaries=()):
    """Time-altitude section of the latitude mean.

    Latitude-weighted, because the 5-degree bands are equal in degrees and not
    in area. This is the panel that shows a 19-year record in one image: a
    processing change, a satellite change or a drift appears as a horizontal
    seam across the whole troposphere.
    """
    weights = np.cos(np.deg2rad(cube["lat"].values))
    profile = cube.weighted(xr.DataArray(weights, dims="lat")).mean("lat")

    fig, ax = plt.subplots(figsize=(11, 4.6))
    mesh, levels = draw_section(ax, pd.to_datetime(cube["time"].values),
                                cube["alt"].values / 1000.0, profile.values.T, style)
    ax.set_xlabel("Date")
    ax.set_ylabel("Altitude [km]")
    ax.grid(linewidth=0.3, alpha=0.4)
    mark_boundaries(ax, boundaries)
    ax.set_title(title)
    add_colorbar(fig, ax, mesh, levels, units)
    fig.savefig(output_path, dpi=140, bbox_inches="tight", facecolor=NAN_COLOR)
    plt.close(fig)


def save_release_diff_panel(newer, older, output_path, title, units):
    """Latitude-height section of one release minus another, time-averaged.

    Averaged over the months the two releases share: a release difference is
    about what reprocessing changed, which a single month would confuse with
    weather.
    """
    left, right = xr.align(newer, older, join="inner")
    if left.sizes["time"] == 0:
        return False
    delta = (left - right).mean("time", skipna=True)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    mesh, levels = draw_section(ax, delta["lat"].values, delta["alt"].values / 1000.0,
                                delta.values,
                                dict(cmap="managua_r", levels=None, symmetric=True))
    ax.set_xlabel("Latitude [degrees north]")
    ax.set_ylabel("Altitude [km]")
    ax.set_xlim(-90, 90)
    ax.set_xticks(np.arange(-90, 91, 30))
    ax.grid(linewidth=0.3, alpha=0.4)
    ax.set_title(title)
    add_colorbar(fig, ax, mesh, levels, units)
    fig.savefig(output_path, dpi=140, bbox_inches="tight", facecolor=NAN_COLOR)
    plt.close(fig)
    return int(left.sizes["time"])


# -----------------------------------------------------------------------------
# Time-series figure -- the same five rows as the gridded galleries
# -----------------------------------------------------------------------------
def percentile_label(thresholds_path, variable, tail):
    default = "P0.1" if tail == "low" else "P99.9"
    name = f"{variable}_p001" if tail == "low" else f"{variable}_p999"
    if thresholds_path is None:
        return default
    try:
        with xr.open_dataset(thresholds_path) as ds:
            quantile = float(ds[name].attrs["quantile"])
    except (KeyError, OSError, TypeError, ValueError):
        return default
    return f"P{100 * quantile:g}"


def create_qc_figure(stats_path, title, units, thresholds_path=None, variable=None,
                     boundaries=()):
    with xr.open_dataset(stats_path) as ds_stats:
        stats = ds_stats.to_dataframe().reset_index()
    stats["time"] = pd.to_datetime(stats["time"])

    low_label = percentile_label(thresholds_path, variable, "low")
    high_label = percentile_label(thresholds_path, variable, "high")

    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=[
            "Mean — temporal stability",
            "Distribution — P01, median and P99",
            "Variability — standard deviation",
            "Extremes — minimum and maximum",
            "Data completeness and climatological outliers",
        ],
    )

    fig.add_trace(go.Scatter(x=stats["time"], y=stats["mean"], mode="lines",
                             name="Mean", line=dict(width=2)), row=1, col=1)

    # P01-P99 band as one closed polygon per contiguous run of valid data:
    # fill="tonexty" pairs segments up wrongly across a gap.
    times = stats["time"].to_numpy()
    p01 = stats["p01"].to_numpy(dtype="float64")
    p99 = stats["p99"].to_numpy(dtype="float64")
    valid = np.flatnonzero(np.isfinite(p01) & np.isfinite(p99))
    first_segment = True
    if valid.size:
        for run in np.split(valid, np.flatnonzero(np.diff(valid) > 1) + 1):
            if run.size < 2:
                continue
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([times[run], times[run][::-1]]),
                    y=np.concatenate([p99[run], p01[run][::-1]]),
                    mode="lines", line=dict(width=0), fill="toself",
                    fillcolor="rgba(100,100,100,0.15)", hoverinfo="skip",
                    name="P01–P99", legendgroup="band", showlegend=first_segment,
                ), row=2, col=1,
            )
            first_segment = False

    fig.add_trace(go.Scatter(x=stats["time"], y=stats["p99"], mode="lines",
                             name="P99", line=dict(width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=stats["time"], y=stats["median"], mode="lines",
                             name="Median", line=dict(width=2, dash="dash")), row=2, col=1)
    fig.add_trace(go.Scatter(x=stats["time"], y=stats["p01"], mode="lines",
                             name="P01", line=dict(width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=stats["time"], y=stats["std"], mode="lines",
                             name="Std", line=dict(width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=stats["time"], y=stats["maximum"], mode="lines",
                             name="Maximum", line=dict(width=1.5)), row=4, col=1)
    fig.add_trace(go.Scatter(x=stats["time"], y=stats["minimum"], mode="lines",
                             name="Minimum", line=dict(width=1.5)), row=4, col=1)
    fig.add_trace(
        go.Scatter(x=stats["time"], y=stats["missing_fraction"], mode="lines",
                   name="Missing fraction", customdata=stats[["missing"]].values,
                   hovertemplate="Missing fraction: %{y:.4%}<br>"
                                 "Missing cells: %{customdata[0]:.0f}<extra></extra>"),
        row=5, col=1)
    fig.add_trace(
        go.Scatter(x=stats["time"], y=stats["negative_outliers_fraction"], mode="lines",
                   name=f"Low-tail outliers (< clim. {low_label})", line=dict(dash="dash")),
        row=5, col=1)
    fig.add_trace(
        go.Scatter(x=stats["time"], y=stats["positive_outliers_fraction"], mode="lines",
                   name=f"High-tail outliers (> clim. {high_label})", line=dict(dash="dot")),
        row=5, col=1)

    # The processing-version changes, on every row: a step in a series that sits
    # on one of these lines is a reprocessing artefact until shown otherwise.
    for boundary in boundaries:
        fig.add_vline(x=pd.Timestamp(boundary["time"]).to_pydatetime(),
                      line=dict(color="rgba(120,120,120,0.9)", width=1, dash="dot"),
                      row="all", col=1)
    if boundaries:
        for boundary in boundaries:
            fig.add_annotation(x=pd.Timestamp(boundary["time"]).to_pydatetime(),
                               y=1.0, yref="y domain", row=1, col=1,
                               text=boundary["label"], showarrow=False,
                               textangle=-90, xanchor="left", yanchor="top",
                               font=dict(size=9, color="#666"))

    for r, label in [(1, units), (2, units), (3, units), (4, units), (5, "Fraction")]:
        fig.update_yaxes(title_text=label, row=r, col=1)
    fig.update_xaxes(title_text="Date", row=5, col=1)
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        height=1300, hovermode="x unified", template="plotly_white",
        margin=dict(l=80, r=40, t=110, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5),
    )
    return fig


# -----------------------------------------------------------------------------
# Manifests
# -----------------------------------------------------------------------------
def gallery_months(cube):
    """The last N_GALLERY_MONTHS months of a cube."""
    times = pd.to_datetime(cube["time"].values)
    return times[-N_GALLERY_MONTHS:]


def build_profile_manifest(releases, skip_render=False):
    manifest, missing = [], []
    for release in releases:
        for source in SOURCES:
            for variable in variables_for(source):
                cube = load_record(variable, source, release)
                if cube is None:
                    missing.append({"release": release, "source": source,
                                    "variable": variable, "reason": "no aux record"})
                    continue
                units = source_units(variable, source)
                style = panel_style(variable, source)
                out_dir = PROFILES_DIR / release / source / variable
                out_dir.mkdir(parents=True, exist_ok=True)
                for i, stamp in enumerate(gallery_months(cube), 1):
                    out = out_dir / f"panel{i:02d}.png"
                    label = f"{stamp:%B %Y}"
                    title = (f"{PRODUCT} {variable} — {SOURCES[source]['label']} — "
                             f"{RELEASES[release]['label']} — {label}")
                    if not skip_render:
                        save_profile_panel(cube, stamp, out, title, units, style)
                    if out.exists():
                        manifest.append({
                            "release": release, "source": source, "variable": variable,
                            "key": f"{source}/{variable}",
                            "date": f"{stamp:%Y-%m}", "label": label,
                            "image": str(out.relative_to(GALLERY_DIR)).replace(os.sep, "/"),
                        })
    return manifest, missing


def build_consistency_manifest(releases, skip_render=False):
    manifest = []
    for release in releases:
        for source in SOURCES:
            for variable in variables_for(source):
                nc_path = aux_consistency(variable, source, release)
                if nc_path is None:
                    continue
                period = " – ".join(
                    pd.to_datetime(d, format="%Y%m%d").strftime("%Y-%m")
                    for d in re.findall(r"\d{8}", nc_path.stem))
                units = source_units(variable, source)
                out_dir = CONSISTENCY_DIR / release / source / variable
                out_dir.mkdir(parents=True, exist_ok=True)
                for statistic, cfg in CONSISTENCY_STATS.items():
                    out = out_dir / f"{statistic}.png"
                    title = (f"{variable} — {SOURCES[source]['label']} — "
                             f"{cfg['title']}")
                    drawn = (out.exists() if skip_render else
                             save_consistency_panel(nc_path, statistic, out, title, units))
                    if drawn:
                        manifest.append({
                            "release": release, "source": source, "variable": variable,
                            "key": f"{source}/{variable}", "statistic": statistic,
                            "label": cfg["title"], "period": period,
                            "source_file": nc_path.name,
                            "image": str(out.relative_to(GALLERY_DIR)).replace(os.sep, "/"),
                        })
    return manifest


def build_hovmoller_manifest(releases, skip_render=False):
    manifest = []
    for release in releases:
        for source in SOURCES:
            for variable in variables_for(source):
                cube = load_record(variable, source, release)
                if cube is None:
                    continue
                out_dir = HOVMOLLER_DIR / release / source
                out_dir.mkdir(parents=True, exist_ok=True)
                out = out_dir / f"{variable}.png"
                times = pd.to_datetime(cube["time"].values)
                title = (f"{PRODUCT} {variable} — {SOURCES[source]['label']} — "
                         f"latitude-weighted mean profile, "
                         f"{times.min():%Y-%m} to {times.max():%Y-%m}")
                if not skip_render:
                    save_hovmoller_panel(cube, out, title,
                                         source_units(variable, source),
                                         panel_style(variable, source),
                                         version_boundaries(release, source))
                if out.exists():
                    manifest.append({
                        "release": release, "source": source, "variable": variable,
                        "key": f"{source}/{variable}",
                        "period": f"{times.min():%Y-%m} – {times.max():%Y-%m}",
                        "image": str(out.relative_to(GALLERY_DIR)).replace(os.sep, "/"),
                    })
    return manifest


def build_release_diff_manifest(releases, skip_render=False):
    """One panel per consecutive pair of releases, newer minus older.

    Empty while only one release is on disk, which is what the page says.
    """
    manifest = []
    for older, newer in zip(releases, releases[1:]):
        for source in SOURCES:
            for variable in variables_for(source):
                new_cube = load_record(variable, source, newer)
                old_cube = load_record(variable, source, older)
                if new_cube is None or old_cube is None:
                    continue
                out_dir = RELEASE_DIFF_DIR / f"{newer}_minus_{older}" / source
                out_dir.mkdir(parents=True, exist_ok=True)
                out = out_dir / f"{variable}.png"
                units = source_units(variable, source)
                title = (f"{variable} — {SOURCES[source]['label']} — "
                         f"{RELEASES[newer]['label']} minus "
                         f"{RELEASES[older]['label']}, mean over shared months")
                months = (out.exists() if skip_render else
                          save_release_diff_panel(new_cube, old_cube, out, title, units))
                if months:
                    manifest.append({
                        "pair": f"{newer}_minus_{older}",
                        "label": f"{RELEASES[newer]['label']} − {RELEASES[older]['label']}",
                        "source": source, "variable": variable,
                        "key": f"{source}/{variable}",
                        "months": months if months is not True else None,
                        "image": str(out.relative_to(GALLERY_DIR)).replace(os.sep, "/"),
                    })
    return manifest


def prune_stale_pngs(*roots_and_manifests):
    """Delete PNGs under a gallery directory that its manifest does not claim.

    An empty manifest prunes nothing: a run that rendered nothing found nothing
    to render (an unmounted DATADIR, missing aux files) rather than making
    everything on disk stale, and --skip-* is meant to reuse those PNGs.
    """
    removed = 0
    for root, manifest in roots_and_manifests:
        if not manifest or not root.exists():
            continue
        keep = {(GALLERY_DIR / item["image"]).resolve() for item in manifest}
        for png in sorted(root.rglob("*.png")):
            if png.resolve() not in keep:
                png.unlink()
                removed += 1
        for d in sorted((d for d in root.rglob("*") if d.is_dir()),
                        key=lambda d: len(d.parts), reverse=True):
            if not any(d.iterdir()):
                d.rmdir()
    return removed


def upload_qc_gallery(gallery_dir, token, target_path=UPLOAD_TARGET,
                      space=UPLOAD_SPACE, site_name=UPLOAD_SITE):
    """Upload the static gallery to an ECMWF Site."""
    gallery_dir = Path(gallery_dir).expanduser().resolve()
    if not gallery_dir.is_dir():
        raise FileNotFoundError(f"Gallery directory does not exist: {gallery_dir}")
    if not (gallery_dir / HTML_NAME).is_file():
        raise FileNotFoundError(f"Expected HTML file not found: {gallery_dir / HTML_NAME}")
    if not (gallery_dir / "profiles").is_dir():
        raise FileNotFoundError(f"Expected profiles directory not found: "
                                f"{gallery_dir / 'profiles'}")

    site = Site(space=space, name=site_name)
    content_manager = site.get_content_manager(
        authenticator=Authenticator.from_token(token=token))
    content_manager.upload(local_path=str(gallery_dir), remote_path=target_path,
                           recursive=True)
    print(f"{PRODUCT} QC gallery uploaded successfully.")
    print(f"https://sites.ecmwf.int/{space}/{site_name}/"
          f"{target_path.rstrip('/')}/{HTML_NAME}")


HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__ Quality Control</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body { font-family: Arial, sans-serif; margin: 0; padding: 18px; color: #222; }
h1 { margin: 0 0 6px; } h2 { margin-top: 28px; }
.panel { border: 1px solid #ddd; border-radius: 6px; padding: 12px; margin-bottom: 18px; }
.tabs, .period-buttons { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 14px; }
button, select { font-size: 15px; padding: 7px 10px; border: 1px solid #bbb; border-radius: 4px; background: #f7f7f7; cursor: pointer; }
button.active { font-weight: 700; background: #e5e5e5; }
pre { white-space: pre-wrap; background: #f7f7f7; padding: 10px; border-radius: 4px; overflow-x: auto; }
table { border-collapse: collapse; width: 100%; } th, td { border: 1px solid #ddd; padding: 6px 8px; text-align: left; vertical-align: top; } th { background: #f5f5f5; }
img.panel-img { display:block; max-width:100%; width:900px; height:auto; margin:8px auto; }
img.wide { width:1100px; }
.nav { display:flex; justify-content:center; gap:8px; margin:8px 0; } .center{text-align:center;} .small{color:#555;font-size:.92em;}
.releasebar { background:#eef4ff; border:1px solid #cfe0ff; border-radius:6px; padding:10px 12px; margin-bottom:14px; }
.pass { color:#0a7d28; font-weight:700; } .fail { color:#b00020; font-weight:700; } .skip { color:#777; }
.controls { display:flex; flex-wrap:wrap; gap:14px; align-items:center; }
</style>
</head>
<body>
<h1>__TITLE__ Quality Control</h1>
<p class="small">Monthly, zonally averaged profiles. Every panel is a latitude-height section: latitude across, altitude up. White is missing data — no colour scale used here goes near white, so a blank area is a gap in the product, not a low value. The product delivers humidity between 0 and 12 km only.</p>

<div class="releasebar">
<b>Release:</b>
<div class="tabs" id="releaseTabs"></div>
<div class="small" id="releaseSummary"></div>
</div>

<div class="panel"><h2>Dataset integrity</h2><div id="integrity"></div></div>
<div class="panel"><h2>Delivered versions</h2>
<p class="small">What the delivered files declare, independently of the release they were requested as. The reprocessed record and the interim record carry different product versions, and the interim part has been re-issued more than once, so this is where a new release or a silent re-issue shows up first.</p>
<div id="inventory"></div></div>
<div class="panel"><h2>Product-type cross-checks</h2>
<p class="small">The reanalysis product is sampled at the occultation locations, so several of its fields are not independent data. These checks assert that; if one fails, the pairing of the two product types has changed and every obs-minus-model number on this page changes meaning.</p>
<div id="crosschecks"></div></div>
<div class="panel"><h2>Metadata</h2><div id="metadata"></div></div>

<div class="panel">
<h2>QC time series</h2>
<div class="controls">
<label>Source: <select id="seriesSourceSelector"></select></label>
<label>Variable: <select id="seriesVariableSelector"></select></label>
</div>
<div id="plots"></div>
</div>

<div class="panel">
<h2>Latitude–height sections</h2>
<div class="controls">
<label>Source: <select id="profileSourceSelector"></select></label>
<label>Variable: <select id="profileVariableSelector"></select></label>
</div>
<div class="period-buttons" id="profileButtons"></div>
<div class="center small" id="profileCounter"></div>
<div class="nav"><button id="prevButton">← Previous</button><button id="nextButton">Next →</button></div>
<img id="profileImg" class="panel-img" alt="Latitude-height section"><div class="center small" id="profileStatus"></div>
</div>

<div class="panel">
<h2>Time–altitude Hovmöller</h2>
<p class="small">The whole record in one image: the latitude-weighted mean profile against time. A reprocessing seam, a satellite change or a drift shows up as a horizontal discontinuity.</p>
<div class="controls">
<label>Source: <select id="hovSourceSelector"></select></label>
<label>Variable: <select id="hovVariableSelector"></select></label>
</div>
<img id="hovImg" class="panel-img wide" alt="Hovmöller"><div class="center small" id="hovStatus"></div>
</div>

<div class="panel">
<h2>Record-wide consistency</h2>
<p class="small">Per grid cell over the whole record, on the full 0–50 km grid the files are written on, so the levels the product never fills stay visible.</p>
<div class="controls">
<label>Source: <select id="consSourceSelector"></select></label>
<label>Variable: <select id="consVariableSelector"></select></label>
</div>
<div class="period-buttons" id="consButtons"></div>
<img id="consImg" class="panel-img" alt="Consistency section"><div class="center small" id="consStatus"></div>
</div>

<div class="panel">
<h2>Release differences</h2>
<p class="small">Newer release minus older, averaged over the months they share.</p>
<div id="releaseDiff"></div>
</div>

<script>
const DATA = __PAYLOAD__;
const PLOTS = __PLOTS__;
const PLOT_CONFIG = {responsive:true, displaylogo:false};
const RELEASES = DATA.releases;
let currentRelease = RELEASES[0] || null;
const rlabel = r => (DATA.release_labels[r] || r);
const slabel = s => (DATA.source_labels[s] || s);

/* Which (source, variable) pairs a manifest holds for the current release. */
function pairsIn(items, release){
  const seen = [];
  for(const it of items){
    if(release !== undefined && it.release !== release) continue;
    if(!seen.some(p => p.source === it.source && p.variable === it.variable))
      seen.push({source: it.source, variable: it.variable});
  }
  return seen;
}
function fillSelect(el, values, current, labeller){
  el.innerHTML = "";
  values.forEach(v => {const o = document.createElement("option"); o.value = v; o.textContent = labeller(v); el.appendChild(o);});
  if(!values.includes(current)) current = values[0];
  el.value = current;
  return current;
}
function sourcesOf(pairs){ return [...new Set(pairs.map(p => p.source))]; }
function variablesOf(pairs, source){ return pairs.filter(p => p.source === source).map(p => p.variable); }

/* ---- release tabs ---- */
function renderReleaseTabs(){
  const t = document.getElementById("releaseTabs"); t.innerHTML = "";
  RELEASES.forEach(r => {
    const b = document.createElement("button");
    b.textContent = rlabel(r); b.className = r === currentRelease ? "active" : "";
    b.onclick = () => {currentRelease = r; renderAll();};
    t.appendChild(b);
  });
  const s = document.getElementById("releaseSummary");
  s.textContent = RELEASES.length > 1
    ? `${RELEASES.length} releases on disk. Every panel below shows ${rlabel(currentRelease)}.`
    : `${rlabel(currentRelease)} is the only release on disk; a second one appears here as soon as it is downloaded.`;
}

/* ---- tables ---- */
function renderIntegrity(){
  const e = document.getElementById("integrity");
  let h = "<table><tr><th>Release</th><th>Source</th><th>Period</th><th>Files found</th><th>Expected</th><th>Existing</th><th>Missing</th></tr>";
  for(const r of RELEASES){ for(const s of DATA.delivered_sources[r]){
    const x = DATA.integrity[r][s];
    h += `<tr><td>${rlabel(r)}</td><td>${slabel(s)}</td><td>${x.period}</td><td>${x.files_found}</td><td>${x.expected}</td><td>${x.existing}</td><td>${x.missing}</td></tr>`;
    if(x.missing){const more = x.missing - x.missing_dates.length;
      h += `<tr><td colspan="7"><b>Missing:</b> ${x.missing_dates.join(", ")}${more>0?` … and ${more} more`:""}</td></tr>`;}
    if(x.extra && x.extra.length)
      h += `<tr><td colspan="7"><b>Outside the declared period:</b> ${x.extra.join(", ")}</td></tr>`;
  }}
  e.innerHTML = h + "</table>";
}
function renderInventory(){
  const e = document.getElementById("inventory");
  let h = "";
  for(const r of RELEASES){
    h += `<h3>${rlabel(r)}</h3>`;
    for(const s of DATA.delivered_sources[r]){
      const rows = DATA.inventory[r][s] || [];
      h += `<h4>${slabel(s)}</h4><table><tr><th>Record</th><th>Product version</th><th>Processing</th><th>Filename token</th><th>Months</th><th>First</th><th>Last</th></tr>`;
      for(const row of rows)
        h += `<tr><td>${row.record}</td><td>${row.version}</td><td>${row.processing}</td><td>${row.token}</td><td>${row.count}</td><td>${row.first}</td><td>${row.last}</td></tr>`;
      h += "</table>";
      const expected = DATA.expected_versions[r] || [];
      const off = rows.filter(row => !expected.includes(row.version)).map(row => row.version);
      if(off.length) h += `<p class="small fail">Product version(s) ${[...new Set(off)].join(", ")} are not among those declared for ${rlabel(r)} (${expected.join(", ")}).</p>`;
    }
  }
  e.innerHTML = h;
}
function renderCrosschecks(){
  const e = document.getElementById("crosschecks");
  let h = "<table><tr><th>Release</th><th>Variable</th><th>Check</th><th>Result</th><th>Detail</th></tr>";
  for(const r of RELEASES) for(const c of (DATA.crosschecks[r] || [])){
    const cls = c.verdict === "pass" ? "pass" : (c.verdict === "FAIL" ? "fail" : "skip");
    h += `<tr><td>${rlabel(r)}</td><td>${c.variable}</td><td>${c.test}</td><td class="${cls}">${c.verdict}</td><td>${c.note}<br><span class="small">${c.detail}</span></td></tr>`;
  }
  e.innerHTML = h + "</table>";
}
function renderMetadata(){
  const e = document.getElementById("metadata"); let h = "";
  for(const s of DATA.delivered_sources[currentRelease]){
    const m = DATA.metadata[currentRelease][s];
    h += `<h3>${slabel(s)}</h3>`;
    if(typeof m === "string"){h += `<pre>${m}</pre>`; continue;}
    h += `<div><b>Example file:</b> <code>${m.file}</code></div>`;
    h += "<table><tr><th>Attribute</th><th>Value</th></tr>";
    for(const [k, v] of Object.entries(m.attributes)) h += `<tr><td>${k}</td><td>${v}</td></tr>`;
    h += "</table>";
    h += `<pre>${m.fieldlist}</pre><pre>${m.xarray_info}</pre>`;
  }
  e.innerHTML = h;
}

/* ---- time series ---- */
let seriesSource = null, seriesVariable = null;
function renderSeries(){
  const pairs = DATA.series[currentRelease] || [];
  const sSel = document.getElementById("seriesSourceSelector");
  const vSel = document.getElementById("seriesVariableSelector");
  const el = document.getElementById("plots");
  if(!pairs.length){sSel.innerHTML = ""; vSel.innerHTML = ""; el.innerHTML = "<p>No pre-calculated statistics were found for this release.</p>"; return;}
  seriesSource = fillSelect(sSel, sourcesOf(pairs), seriesSource, slabel);
  seriesVariable = fillSelect(vSel, variablesOf(pairs, seriesSource), seriesVariable, v => v);
  const key = `${currentRelease}|${seriesSource}|${seriesVariable}`;
  if(!PLOTS[key]){el.innerHTML = "<p>No statistics file for this combination.</p>"; return;}
  el.innerHTML = '<div id="plotlyQC" style="width:100%;height:1300px;"></div>';
  const p = JSON.parse(PLOTS[key]);
  Plotly.newPlot("plotlyQC", p.data, p.layout, PLOT_CONFIG);
}

/* ---- latitude-height browser ---- */
let profileSource = null, profileVariable = null, profileIndex = 0;
function profileItems(){
  return DATA.profiles.filter(p => p.release === currentRelease
    && p.source === profileSource && p.variable === profileVariable);
}
function renderProfiles(){
  const pairs = pairsIn(DATA.profiles, currentRelease);
  const sSel = document.getElementById("profileSourceSelector");
  const vSel = document.getElementById("profileVariableSelector");
  if(!pairs.length){document.getElementById("profileImg").removeAttribute("src");
    document.getElementById("profileStatus").textContent = "No sections were rendered for this release."; return;}
  profileSource = fillSelect(sSel, sourcesOf(pairs), profileSource, slabel);
  profileVariable = fillSelect(vSel, variablesOf(pairs, profileSource), profileVariable, v => v);
  const items = profileItems();
  const c = document.getElementById("profileButtons"); c.innerHTML = "";
  if(!items.length){document.getElementById("profileImg").removeAttribute("src");
    document.getElementById("profileCounter").textContent = "No section available"; return;}
  profileIndex = Math.max(0, Math.min(profileIndex, items.length - 1));
  items.forEach((it, i) => {const b = document.createElement("button"); b.textContent = it.label;
    b.className = i === profileIndex ? "active" : ""; b.onclick = () => {profileIndex = i; renderProfiles();}; c.appendChild(b);});
  const item = items[profileIndex];
  document.getElementById("profileImg").src = item.image;
  document.getElementById("profileCounter").textContent = `${item.date} — ${profileIndex+1} / ${items.length}`;
  document.getElementById("profileStatus").textContent =
    `${item.variable} [${DATA.units[item.source+"|"+item.variable]}] — ${slabel(item.source)} — ${rlabel(currentRelease)}`;
}

/* ---- Hovmoller ---- */
let hovSource = null, hovVariable = null;
function renderHovmoller(){
  const pairs = pairsIn(DATA.hovmoller, currentRelease);
  const sSel = document.getElementById("hovSourceSelector");
  const vSel = document.getElementById("hovVariableSelector");
  const img = document.getElementById("hovImg"), st = document.getElementById("hovStatus");
  if(!pairs.length){img.removeAttribute("src"); st.textContent = "No Hovmöller panels for this release."; return;}
  hovSource = fillSelect(sSel, sourcesOf(pairs), hovSource, slabel);
  hovVariable = fillSelect(vSel, variablesOf(pairs, hovSource), hovVariable, v => v);
  const item = DATA.hovmoller.find(h => h.release === currentRelease && h.source === hovSource && h.variable === hovVariable);
  if(!item){img.removeAttribute("src"); st.textContent = "Not available."; return;}
  img.src = item.image;
  st.textContent = `${item.variable} [${DATA.units[item.source+"|"+item.variable]}] — ${slabel(item.source)} — ${item.period}`;
}

/* ---- record-wide consistency ---- */
let consSource = null, consVariable = null, consStat = null;
function renderConsistency(){
  const pairs = pairsIn(DATA.consistency, currentRelease);
  const sSel = document.getElementById("consSourceSelector");
  const vSel = document.getElementById("consVariableSelector");
  const img = document.getElementById("consImg"), st = document.getElementById("consStatus");
  const c = document.getElementById("consButtons"); c.innerHTML = "";
  if(!pairs.length){img.removeAttribute("src"); st.textContent = "No consistency panels for this release."; return;}
  consSource = fillSelect(sSel, sourcesOf(pairs), consSource, slabel);
  consVariable = fillSelect(vSel, variablesOf(pairs, consSource), consVariable, v => v);
  const items = DATA.consistency.filter(x => x.release === currentRelease && x.source === consSource && x.variable === consVariable);
  if(!items.length){img.removeAttribute("src"); st.textContent = "Not available."; return;}
  if(!items.some(x => x.statistic === consStat)) consStat = items[0].statistic;
  items.forEach(x => {const b = document.createElement("button"); b.textContent = x.label;
    b.className = x.statistic === consStat ? "active" : ""; b.onclick = () => {consStat = x.statistic; renderConsistency();}; c.appendChild(b);});
  const item = items.find(x => x.statistic === consStat);
  img.src = item.image;
  st.textContent = `${item.variable} — ${slabel(item.source)} — ${item.label} | Period: ${item.period} | Source: ${item.source_file}`;
}

/* ---- release differences ---- */
function renderReleaseDiff(){
  const e = document.getElementById("releaseDiff");
  if(!DATA.release_diff.length){
    e.innerHTML = `<p class="small">Only one release is on disk (${RELEASES.map(rlabel).join(", ")}), so there is nothing to difference yet. Download a second release and this panel fills itself in.</p>`;
    return;
  }
  const pairs = [...new Set(DATA.release_diff.map(d => d.pair))];
  let h = "";
  for(const pair of pairs){
    const items = DATA.release_diff.filter(d => d.pair === pair);
    h += `<h3>${items[0].label}</h3>`;
    for(const it of items)
      h += `<div><b>${it.variable} — ${slabel(it.source)}</b>${it.months?` <span class="small">(${it.months} shared months)</span>`:""}</div>
            <img class="panel-img" src="${it.image}" alt="${it.variable} release difference">`;
  }
  e.innerHTML = h;
}

document.getElementById("seriesSourceSelector").addEventListener("change", e => {seriesSource = e.target.value; seriesVariable = null; renderSeries();});
document.getElementById("seriesVariableSelector").addEventListener("change", e => {seriesVariable = e.target.value; renderSeries();});
document.getElementById("profileSourceSelector").addEventListener("change", e => {profileSource = e.target.value; profileVariable = null; profileIndex = 0; renderProfiles();});
document.getElementById("profileVariableSelector").addEventListener("change", e => {profileVariable = e.target.value; profileIndex = 0; renderProfiles();});
document.getElementById("hovSourceSelector").addEventListener("change", e => {hovSource = e.target.value; hovVariable = null; renderHovmoller();});
document.getElementById("hovVariableSelector").addEventListener("change", e => {hovVariable = e.target.value; renderHovmoller();});
document.getElementById("consSourceSelector").addEventListener("change", e => {consSource = e.target.value; consVariable = null; consStat = null; renderConsistency();});
document.getElementById("consVariableSelector").addEventListener("change", e => {consVariable = e.target.value; consStat = null; renderConsistency();});
document.getElementById("prevButton").onclick = () => {const n = profileItems().length; if(n){profileIndex = (profileIndex - 1 + n) % n; renderProfiles();}};
document.getElementById("nextButton").onclick = () => {const n = profileItems().length; if(n){profileIndex = (profileIndex + 1) % n; renderProfiles();}};
document.addEventListener("keydown", e => {if(e.key === "ArrowLeft") document.getElementById("prevButton").click(); if(e.key === "ArrowRight") document.getElementById("nextButton").click();});

function renderAll(){
  renderReleaseTabs(); renderIntegrity(); renderInventory(); renderCrosschecks();
  renderMetadata(); renderSeries(); renderProfiles(); renderHovmoller();
  renderConsistency(); renderReleaseDiff();
}
renderAll();
</script>
</body></html>'''


def main():
    parser = argparse.ArgumentParser(description=f"Build the {DATASET_TITLE} QC gallery")
    parser.add_argument("--skip-panels", action="store_true",
                        help="Reuse the PNGs already on disk instead of re-rendering "
                             "them (for changes that only affect the HTML)")
    parser.add_argument("--no-prune", action="store_true",
                        help="Keep PNGs this run did not render")
    parser.add_argument("--no-upload", action="store_true",
                        help="Write the gallery locally without uploading it")
    args = parser.parse_args()

    # Fail here rather than half-way through a render if a palette cannot be
    # made white-free.
    for style in list(PANEL_STYLES.values()) + list(DERIVED_STYLES.values()) \
            + list(CONSISTENCY_STATS.values()):
        qc_colormap(style.get("cmap", "viridis"))

    GALLERY_DIR.mkdir(parents=True, exist_ok=True)

    releases = releases_on_disk()
    if not releases:
        raise SystemExit(f"No release has files under {DATADIR}. Run "
                         f"download_data.py first.")

    integrity, inventory, metadata, crosschecks = {}, {}, {}, {}
    delivered, expected_versions = {}, {}
    for release in releases:
        delivered[release] = delivered_sources(release)
        expected_versions[release] = list(RELEASES[release]["product_versions"])
        integrity[release], inventory[release], metadata[release] = {}, {}, {}
        for source in delivered[release]:
            df = collect_files(release, source)
            integrity[release][source] = integrity_report(release, source, df)
            inventory[release][source] = version_inventory(df)
            metadata[release][source] = metadata_report(release, source, df)
        crosschecks[release] = crosscheck_report(release)

    profiles, missing_panels = build_profile_manifest(releases, args.skip_panels)
    hovmoller = build_hovmoller_manifest(releases, args.skip_panels)
    consistency = build_consistency_manifest(releases, args.skip_panels)
    release_diff = build_release_diff_manifest(releases, args.skip_panels)

    pruned = 0 if args.no_prune else prune_stale_pngs(
        (PROFILES_DIR, profiles), (HOVMOLLER_DIR, hovmoller),
        (CONSISTENCY_DIR, consistency), (RELEASE_DIFF_DIR, release_diff))

    # One figure per (release, source, variable) that has a statistics file.
    plots, series = {}, {}
    for release in releases:
        series[release] = []
        for source in SOURCES:
            for variable in variables_for(source):
                stats_path = aux_tseries(variable, source, release)
                if stats_path is None:
                    continue
                units = source_units(variable, source)
                title = (f"{DATASET_TITLE} — {variable} "
                         f"({VARIABLES[variable]['label']}) — "
                         f"{SOURCES[source]['label']} — {RELEASES[release]['label']}")
                figure = create_qc_figure(stats_path, title, units,
                                          aux_thresholds(variable, source, release),
                                          variable,
                                          version_boundaries(release, source))
                plots[f"{release}|{source}|{variable}"] = figure.to_json()
                series[release].append({"source": source, "variable": variable})

    payload = {
        "title": DATASET_TITLE,
        "releases": releases,
        "release_labels": {r: RELEASES[r]["label"] for r in releases},
        "source_labels": {s: SOURCES[s]["label"] for s in SOURCES},
        "delivered_sources": delivered,
        "expected_versions": expected_versions,
        "units": {f"{s}|{v}": source_units(v, s) for s in SOURCES for v in VARIABLES},
        "integrity": integrity,
        "inventory": inventory,
        "crosschecks": crosschecks,
        "metadata": metadata,
        "series": series,
        "profiles": profiles,
        "hovmoller": hovmoller,
        "consistency": consistency,
        "release_diff": release_diff,
    }

    html_text = (HTML_TEMPLATE
                 .replace("__PAYLOAD__", json.dumps(payload, default=str).replace("</", "<\\/"))
                 .replace("__PLOTS__", json.dumps(plots).replace("</", "<\\/"))
                 .replace("__TITLE__", DATASET_TITLE))
    HTML_PATH.write_text(html_text, encoding="utf-8")
    TOP_LEVEL_HTML.write_text(html_text, encoding="utf-8")

    if args.no_upload:
        print("Upload skipped (--no-upload)")
    else:
        upload_qc_gallery(gallery_dir=f"./{GALLERY_DIR}",
                          token=os.environ["ECMWF_SITES_TOKEN"])

    print(f"Created {HTML_PATH}")
    print(f"Created {TOP_LEVEL_HTML}")
    print(f"Releases on disk: {', '.join(releases)}")
    print(f"Created {len(profiles)} latitude-height sections, "
          f"{len(hovmoller)} Hovmoller panels, "
          f"{len(consistency)} consistency panels, "
          f"{len(release_diff)} release-difference panels")
    print(f"Time series figures: {len(plots)}")
    if pruned:
        print(f"Pruned {pruned} stale PNGs no longer referenced by the gallery")
    if missing_panels:
        print(f"Missing panel combinations: {len(missing_panels)} "
              f"(run compute_stats.py for them)")
    for release in releases:
        failed = [c for c in crosschecks[release] if c["verdict"] == "FAIL"]
        if failed:
            print(f"CROSS-CHECK FAILURES in {release}: "
                  + "; ".join(f"{c['variable']} ({c['note']})" for c in failed))


if __name__ == "__main__":
    main()
