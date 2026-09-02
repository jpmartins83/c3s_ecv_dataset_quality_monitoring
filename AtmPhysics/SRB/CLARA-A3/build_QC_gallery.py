from pathlib import Path
import argparse
import io
import json
import os
import re

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import earthkit.data as ekd
import earthkit.plots as ekp

from sites.sdk.sites import Site, Authenticator




# -----------------------------------------------------------------------------
# Configuration — this is the only block that changes from one dataset to another.
# The dataset name only appears in FOLDER names, never in the names of the files
# that are produced, so the gallery layout stays the same for every dataset.
# -----------------------------------------------------------------------------
DOMAIN = "AtmPhysics"
ECV = "SRB"
PRODUCT = "CLARA-A3"
DATASET_TITLE = "CLARA-A3 Surface Radiation Budget"
UNITS = "W m-2"

DATADIR = Path(f"../../../datasets/{ECV}/{PRODUCT}")
AUX_DIR = Path("aux_files")

GALLERY_DIR = Path(f"{PRODUCT}_QC_gallery")
MAPS_DIR = GALLERY_DIR / "maps"
SPATIAL_DIR = GALLERY_DIR / "spatial"
HTML_NAME = "QC_timeseries.html"
HTML_PATH = GALLERY_DIR / HTML_NAME
TOP_LEVEL_HTML = Path(HTML_NAME)

UPLOAD_TARGET = f"dataset_qc/{DOMAIN}/{ECV}/{PRODUCT}/"
UPLOAD_SPACE = "cxjo"
UPLOAD_SITE = "ecv-info"

FREQUENCIES = ("monthly_mean", "daily_mean")
FREQ_LABELS = {"monthly_mean": "Monthly", "daily_mean": "Daily"}
FREQ_RESOLUTION = {"monthly_mean": "MS", "daily_mean": "D"}

# One NetCDF file per product variable. Only SIS is delivered as daily means.
PRODUCTS = {
    "monthly_mean": ["SIS", "SDL", "SNS", "SNL", "SRB"],
    "daily_mean": ["SIS"],
}

# Fields to map, grouped by the product file that holds them.
MAP_FIELDS = {
    "monthly_mean": {
        "SIS": ["SIS", "SISCLS", "SIS_stdv", "SIS_nobs", "SIS_dms", "SISCLS_dms"],
        "SDL": ["SDL"],
        "SNS": ["SNS"],
        "SNL": ["SNL"],
        "SRB": ["SRB"],
    },
    "daily_mean": {
        "SIS": ["SIS", "SISCLS", "SIS_stdv", "SIS_nobs"],
    },
}

# The dataset filenames use the product name itself as their prefix.
PRODUCT_PREFIX = {p: p for f in MAP_FIELDS for p in MAP_FIELDS[f]}

# Map gallery covers the last N months of the dataset, one map per month.
# For daily means the map is taken from GALLERY_DAY of each month.
N_GALLERY_MONTHS = 6
GALLERY_DAY = 1

COLLECTION_START = pd.Timestamp("1979-01-01")
# None -> auto-detect from the newest file found for each (frequency, variable).
COLLECTION_END = None

# Filename pattern: <VAR><dm|mm><YYYYMMDD>000000319AVPOS<01|I1>GL.nc
# (01GL for the CDR, I1GL for the ICDR — the suffix is irrelevant here).
FILE_PATTERN = re.compile(r"(?P<prefix>[A-Z]{3})(?:dm|mm)(?P<file_date>\d{8})")

MAP_STYLES = {
    "SIS": dict(cmap="inferno", levels=np.arange(0, 451, 25)),
    "SISCLS": dict(cmap="inferno", levels=np.arange(0, 451, 25)),
    "SDL": dict(cmap="magma", levels=np.arange(50, 501, 25)),
    "SNS": dict(cmap="inferno", levels=np.arange(0, 401, 25)),
    "SNL": dict(cmap="viridis", levels=np.arange(-220, 21, 20)),
    "SRB": dict(cmap="RdBu_r", levels=np.arange(-150, 351, 25)),
    "SIS_stdv": dict(cmap="cividis", levels=np.arange(0, 141, 10)),
    "SIS_nobs": dict(cmap="Greens", levels=None),  # range differs per frequency
    "SIS_dms": dict(cmap="Blues", levels=np.arange(0, 32, 2)),
    "SISCLS_dms": dict(cmap="Blues", levels=np.arange(0, 32, 2)),
}

# Spatial-consistency statistics aggregated over the whole collection, as written
# by compute_stats.py. levels=None means they are derived from the data itself.
SPATIAL_MAPS = {
    "Missing_values": dict(
        title="Total Number of Missing Values",
        label="# of Missing Values",
        cmap="magma_r",
        # The upper end is derived from the data, because a fixed ladder cannot
        # suit both a monthly aggregate (a handful of missing months) and a daily
        # one over thousands of files. The low end keeps a fixed non-linear
        # ladder: one or five missing values matter as much as a thousand, and a
        # single wide 0-50 class would hide them.
        levels=None,
        low_levels=[0, 1, 3, 5, 10, 20, 50, 100, 300],
    ),
    "Number_of_values": dict(
        title="Total Number of Valid Values",
        label="# of Valid Values",
        cmap="viridis",
        levels=None,
    ),
    "Max_value": dict(
        title="Maximum Value",
        label=f"Maximum value [{UNITS}]",
        cmap="inferno",
        levels=None,
    ),
    "Min_value": dict(
        title="Minimum Value",
        label=f"Minimum value [{UNITS}]",
        cmap="viridis",
        levels=None,
    ),
}


# The dataset filenames use a short prefix; everything else (aux files, labels,
# the payload) uses the product name.
PREFIX_TO_PRODUCT = {prefix: product for product, prefix in PRODUCT_PREFIX.items()}


def collect_files(frequency):
    """Index every NetCDF file of one frequency by product variable and date."""
    files = [p for p in (DATADIR / frequency).rglob("*.nc") if p.is_file()]
    records = []
    for p in files:
        match = FILE_PATTERN.match(p.name)
        records.append({
            "file_path": str(p),
            "variable": PREFIX_TO_PRODUCT.get(match.group("prefix")) if match else None,
            "file_date": match.group("file_date") if match else None,
        })
    df = pd.DataFrame(records, columns=["file_path", "variable", "file_date"])
    return df.sort_values(["variable", "file_date"], na_position="last").reset_index(drop=True)


def newest_date(df):
    """Newest file date in df as a Timestamp, or None if there is none."""
    dates = df["file_date"].dropna()
    if dates.empty:
        return None
    return pd.to_datetime(dates.max(), format="%Y%m%d")


def gallery_dates(frequency, df):
    """The last N_GALLERY_MONTHS months present in the data, one date per month."""
    last = newest_date(df)
    if last is None:
        return pd.DatetimeIndex([])
    months = pd.date_range(end=last.normalize().replace(day=1),
                           periods=N_GALLERY_MONTHS, freq="MS")
    if not frequency.startswith("daily"):
        return months
    return pd.DatetimeIndex([m.replace(day=GALLERY_DAY) for m in months])


def integrity_report(frequency, variable, df):
    sub = df[df["variable"] == variable]
    end = COLLECTION_END or newest_date(sub)
    if end is None:
        return {
            "frequency": frequency,
            "variable": variable,
            "files_found": 0,
            "expected": 0,
            "existing": 0,
            "missing": 0,
            "missing_dates": [],
        }

    expected = pd.date_range(COLLECTION_START, end, freq=FREQ_RESOLUTION[frequency])
    expected_keys = set(expected.strftime("%Y%m%d"))
    present = set(sub["file_date"].dropna())
    missing = sorted(expected_keys - present)
    existing = sorted(expected_keys & present)
    return {
        "frequency": frequency,
        "variable": variable,
        "files_found": len(sub),
        "expected": len(expected_keys),
        "existing": len(existing),
        "missing": len(missing),
        "missing_dates": missing,
        "period": f"{COLLECTION_START:%Y-%m-%d} – {end:%Y-%m-%d}",
    }


def metadata_report(variable, df):
    valid = df[(df["variable"] == variable)].dropna(subset=["file_date"])
    if valid.empty:
        return f"No valid {variable} files found."

    fname = valid.iloc[-1]["file_path"]
    result = {"file": fname}

    if ekd is not None:
        fieldlist = ekd.from_source("file", fname).to_fieldlist()
        fls = fieldlist.ls()
        result["fieldlist"] = fls.to_string(index=False)
        ds_xr = ekd.from_source("file", fname).to_xarray()
        buf = io.StringIO()
        ds_xr.info(buf=buf)
        result["xarray_info"] = buf.getvalue()
    else:
        # Fallback if earthkit is unavailable.
        with xr.open_dataset(fname) as ds:
            result["fieldlist"] = "Variables: " + ", ".join(ds.data_vars)
            result["xarray_info"] = str(ds)

    return result


def auto_levels(da, n=11, robust=True):
    """Contour levels derived from the range of the data, or None.

    robust=True clips to the 1st–99th percentiles (better for single fields with
    a few extreme pixels); robust=False spans the full range, which is what the
    collection-wide minimum/maximum maps need so that no grid cell is left blank.
    """
    values = np.asarray(da.values, dtype="float64")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    lo, hi = np.nanpercentile(finite, [1, 99] if robust else [0, 100])
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if hi <= lo:
        # Constant field, e.g. a product with no missing values anywhere. One bin
        # is still worth drawing; returning None would drop the map altogether.
        width = 1.0 if float(lo).is_integer() else max(abs(lo) * 0.01, 1e-9)
        return np.array([lo, lo + width])
    step = (hi - lo) / (n - 1)
    # Round the step up to a "nice" value so the colour bar reads cleanly.
    magnitude = 10.0 ** np.floor(np.log10(step))
    for nice in (1, 2, 2.5, 5, 10):
        if step <= nice * magnitude:
            step = nice * magnitude
            break
    # Counts are whole numbers, so fractional levels would be meaningless: a
    # field holding only 570 and 571 should get one level per value, not ten
    # sub-divisions of a single count.
    if np.all(finite == np.floor(finite)):
        step = max(1.0, np.round(step))
    start = np.floor(lo / step) * step
    # The top boundary must sit STRICTLY above the maximum. If it coincides with
    # the maximum, every pixel holding that value falls out of range and is drawn
    # as missing - which blanks almost the whole map for a field whose maximum is
    # also its most common value, such as a complete-coverage count.
    n_levels = max(2, int(np.floor((hi - start) / step)) + 2)
    return start + step * np.arange(n_levels)


def percentile_label(thresholds_path, variable, tail):
    """'P0.1' / 'P99.9' read from the climatology file attrs, with a fallback."""
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


def create_qc_figure(stats_path, title, units, thresholds_path=None, variable=None):
    with xr.open_dataset(stats_path) as ds_stats:
        stats = ds_stats.to_dataframe().reset_index()

    stats["time"] = pd.to_datetime(stats["time"])

    low_label = percentile_label(thresholds_path, variable, "low")
    high_label = percentile_label(thresholds_path, variable, "high")

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[
            "Mean — temporal stability",
            "Distribution — P01, median and P99",
            "Variability — standard deviation",
            "Extremes — minimum and maximum",
            "Data completeness and climatological outliers",
        ],
    )

    fig.add_trace(go.Scatter(x=stats["time"], y=stats["mean"], mode="lines", name="Mean", line=dict(width=2)), row=1, col=1)
    # P01-P99 band, drawn as one closed polygon per contiguous run of valid data.
    # fill="tonexty" cannot be used here: across a NaN gap it pairs the segments
    # up wrongly and closes the polygon with a chord instead of following the
    # lower bound. It also anchors to whichever trace precedes it in the list,
    # which made the band span median..P01 rather than P01..P99.
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

    fig.add_trace(go.Scatter(x=stats["time"], y=stats["p99"], mode="lines", name="P99", line=dict(width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=stats["time"], y=stats["median"], mode="lines", name="Median", line=dict(width=2, dash="dash")), row=2, col=1)
    fig.add_trace(go.Scatter(x=stats["time"], y=stats["p01"], mode="lines", name="P01", line=dict(width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=stats["time"], y=stats["std"], mode="lines", name="Std", line=dict(width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=stats["time"], y=stats["maximum"], mode="lines", name="Maximum", line=dict(width=1.5)), row=4, col=1)
    fig.add_trace(go.Scatter(x=stats["time"], y=stats["minimum"], mode="lines", name="Minimum", line=dict(width=1.5)), row=4, col=1)

    fig.add_trace(
        go.Scatter(
            x=stats["time"], y=stats["missing_fraction"], mode="lines", name="Missing fraction",
            customdata=stats[["missing"]].values,
            hovertemplate="Missing fraction: %{y:.4%}<br>Missing pixels: %{customdata[0]:.0f}<extra></extra>",
        ), row=5, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=stats["time"], y=stats["negative_outliers_fraction"], mode="lines",
            name=f"Low-tail outliers (< clim. {low_label})", line=dict(dash="dash"),
        ),
        row=5, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=stats["time"], y=stats["positive_outliers_fraction"], mode="lines",
            name=f"High-tail outliers (> clim. {high_label})", line=dict(dash="dot"),
        ),
        row=5, col=1,
    )

    for r, label in [(1, units), (2, units), (3, units), (4, units), (5, "Fraction")]:
        fig.update_yaxes(title_text=label, row=r, col=1)
    fig.update_xaxes(title_text="Date", row=5, col=1)

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        height=1300,
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=80, r=40, t=110, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5),
    )
    return fig


def find_file(df, frequency, variable, date):
    key = date.strftime("%Y%m%d")
    matches = df.loc[(df["variable"] == variable) & (df["file_date"] == key), "file_path"]
    return matches.iloc[0] if not matches.empty else None


def save_map(fname, field, output_path, title):
    with xr.open_dataset(fname) as ds:
        if field not in ds.data_vars:
            return False
        da = ds[field]
        if "time" in da.dims:
            da = da.isel(time=0)
        da = da.squeeze(drop=True).load()

    fig = plt.figure(figsize=(13, 5.7))
    ax = plt.axes(projection=ccrs.PlateCarree())

    style = MAP_STYLES.get(field, {})
    levels = style.get("levels")
    if levels is None:
        levels = auto_levels(da)

    kwargs = {
        "ax": ax,
        "transform": ccrs.PlateCarree(),
        "cmap": style.get("cmap", "viridis"),
        "add_colorbar": True,
    }
    if levels is not None:
        kwargs["levels"] = levels

    try:
        da.plot.pcolormesh(**kwargs)
    except (TypeError, ValueError):
        kwargs.pop("levels", None)
        da.plot.pcolormesh(**kwargs)

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    ax.set_title(title)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def find_aux_file(pattern):
    """Most recent aux_files/ file matching pattern, if any."""
    matches = sorted(AUX_DIR.glob(pattern))
    return matches[-1] if matches else None


def save_spatial_map(nc_path, statistic, output_path, title):
    style_cfg = SPATIAL_MAPS[statistic]

    with xr.open_dataset(nc_path) as ds:
        if statistic not in ds.data_vars:
            return False
        da = ds[statistic].squeeze(drop=True).load()

    levels = style_cfg.get("levels")
    if levels is None:
        # Full range: these are collection-wide aggregates, so nothing should be
        # clipped out of the colour scale.
        levels = auto_levels(da, robust=False)

    low_levels = style_cfg.get("low_levels")
    if low_levels is not None:
        # Fixed non-linear ladder at the low end, data-derived levels above it.
        upper = [float(x) for x in (levels if levels is not None else [])
                 if x > low_levels[-1]]
        levels = [float(x) for x in low_levels] + upper

    if levels is None or len(levels) < 2:
        return False

    style = ekp.styles.Style(
        colors=style_cfg.get("cmap", "viridis"),
        levels=list(levels),
        ticks=list(levels),
    )

    chart = ekp.Map()
    chart.pcolormesh(da, style=style)
    chart.title(title)
    chart.coastlines()
    chart.gridlines()
    chart.legend(label=style_cfg["label"])
    chart.save(str(output_path))
    plt.close("all")
    return True


def build_spatial_manifest(skip_render=False):
    """Render the collection-wide spatial-consistency maps per frequency and variable.

    skip_render reuses the PNGs already on disk, for when only the HTML changed.
    """
    manifest = []
    for f in FREQUENCIES:
        for variable in PRODUCTS[f]:
            nc_path = find_aux_file(f"spatial_consistency_{variable}_{f}_*.nc")
            if nc_path is None:
                continue
            # The aggregation period is the pair of dates in the file name.
            period = " – ".join(
                pd.to_datetime(d, format="%Y%m%d").strftime("%Y-%m-%d")
                for d in re.findall(r"\d{8}", nc_path.stem)
            )
            out_dir = SPATIAL_DIR / f / variable
            out_dir.mkdir(parents=True, exist_ok=True)
            for statistic, cfg in SPATIAL_MAPS.items():
                out = out_dir / f"{statistic}.png"
                title = f"{variable} — {cfg['title']}"
                drawn = out.exists() if skip_render else save_spatial_map(nc_path, statistic, out, title)
                if drawn:
                    manifest.append({
                        "frequency": f,
                        "variable": variable,
                        "statistic": statistic,
                        "label": cfg["title"],
                        "period": period,
                        "source": nc_path.name,
                        "image": str(out.relative_to(GALLERY_DIR)).replace(os.sep, "/"),
                    })
    return manifest


def upload_qc_gallery(
    gallery_dir,
    token,
    target_path=UPLOAD_TARGET,
    space=UPLOAD_SPACE,
    site_name=UPLOAD_SITE,
    ):
    """
    Upload the QC static gallery to an ECMWF Site.

    Parameters
    ----------
    gallery_dir : str or pathlib.Path
        Local directory containing QC_timeseries.html and maps/.
        For example: "./<PRODUCT>_QC_gallery"

    token : str
        ECMWF Sites master token.

    target_path : str
        Path inside the ECMWF Site where the gallery will be uploaded.

    space : str
        ECMWF Sites space.

    site_name : str
        ECMWF Site name.

    Returns
    -------
    None
    """

    gallery_dir = Path(gallery_dir).expanduser().resolve()

    if not gallery_dir.is_dir():
        raise FileNotFoundError(
            f"Gallery directory does not exist: {gallery_dir}"
        )

    html_file = gallery_dir / HTML_NAME
    maps_dir = gallery_dir / "maps"

    if not html_file.is_file():
        raise FileNotFoundError(
            f"Expected HTML file not found: {html_file}"
        )

    if not maps_dir.is_dir():
        raise FileNotFoundError(
            f"Expected maps directory not found: {maps_dir}"
        )

    # Connect to the ECMWF Site
    site = Site(
        space=space,
        name=site_name,
    )

    content_manager = site.get_content_manager(
        authenticator=Authenticator.from_token(
            token=token
        )
    )

    # print(inspect.signature(content_manager.upload))
    # print(inspect.getsource(content_manager.upload))
    # help(content_manager.upload)
    # raise SystemExit

    # Upload the complete gallery
    content_manager.upload(
        local_path=str(gallery_dir),
        remote_path=target_path,
        recursive=True,
    )

    print(f"{PRODUCT} QC gallery uploaded successfully.")
    print(
        f"https://sites.ecmwf.int/{space}/{site_name}/"
        f"{target_path.rstrip('/')}/{HTML_NAME}"
    )


HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__ Quality Control</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body { font-family: Arial, sans-serif; margin: 0; padding: 18px; color: #222; }
h1 { margin: 0 0 14px; } h2 { margin-top: 28px; }
.panel { border: 1px solid #ddd; border-radius: 6px; padding: 12px; margin-bottom: 18px; }
.tabs, .period-buttons { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 14px; }
button, select { font-size: 15px; padding: 7px 10px; border: 1px solid #bbb; border-radius: 4px; background: #f7f7f7; cursor: pointer; }
button.active { font-weight: 700; background: #e5e5e5; }
pre { white-space: pre-wrap; background: #f7f7f7; padding: 10px; border-radius: 4px; overflow-x: auto; }
table { border-collapse: collapse; width: 100%; } th, td { border: 1px solid #ddd; padding: 6px 8px; text-align: left; vertical-align: top; } th { background: #f5f5f5; }
#map, #spatialMap { display:block; max-width:100%; width:1000px; height:auto; margin:8px auto; }
.nav { display:flex; justify-content:center; gap:8px; margin:8px 0; } .center{text-align:center;} .small{color:#555;font-size:.92em;}
</style>
</head>
<body>
<h1>__TITLE__ Quality Control</h1>
<div class="panel"><h2>Dataset integrity</h2><div id="integrity"></div></div>
<div class="panel"><h2>Metadata</h2><div id="metadata"></div></div>
<div class="panel">
<h2>QC time series</h2>
<div class="tabs" id="frequencyTabs"></div>
<label>Variable: <select id="plotVariableSelector"></select></label>
<div id="plots"></div>
</div>
<div class="panel">
<h2>Spatial inspection</h2>
<div class="tabs" id="mapFrequencyTabs"></div>
<label>Variable: <select id="variableSelector"></select></label>
<div class="period-buttons" id="periodButtons"></div>
<div class="center small" id="mapCounter"></div>
<div class="nav"><button id="prevButton">← Previous</button><button id="nextButton">Next →</button></div>
<img id="map" alt="QC map"><div class="center small" id="mapStatus"></div>
</div>
<div class="panel">
<h2>Spatial consistency</h2>
<p class="small">Statistics aggregated over the whole collection, computed per grid cell.</p>
<div class="tabs" id="spatialFrequencyTabs"></div>
<label>Variable: <select id="spatialVariableSelector"></select></label>
<div class="period-buttons" id="spatialStatButtons"></div>
<img id="spatialMap" alt="Spatial consistency map"><div class="center small" id="spatialStatus"></div>
</div>
<script>
const DATA = __PAYLOAD__;
const PLOTS = __PLOTS__;
const PLOT_CONFIG = {responsive:true, displaylogo:false};
const FREQS = DATA.frequencies;
const flabel = f => DATA.freq_labels[f] || f;
let currentPlotFrequency=FREQS[0], currentPlotVariable=null;
let currentMapFrequency=FREQS[0], currentField=null, currentIndex=0;
let currentSpatialFrequency=FREQS[0], currentSpatialVariable=null, currentSpatialStat=null;
function fieldsFor(f){const m=DATA.map_fields[f]||{};return DATA.variables[f].flatMap(v=>m[v]||[]);}
function renderIntegrity(){const e=document.getElementById("integrity");let h="<table><tr><th>Frequency</th><th>Variable</th><th>Period</th><th>Files found</th><th>Expected</th><th>Existing</th><th>Missing</th></tr>";for(const f of FREQS){for(const v of DATA.variables[f]){const r=DATA.integrity[f][v];h+=`<tr><td>${flabel(f)}</td><td>${v}</td><td>${r.period||""}</td><td>${r.files_found}</td><td>${r.expected}</td><td>${r.existing}</td><td>${r.missing}</td></tr>`;if(r.missing)h+=`<tr><td colspan="7"><b>Missing:</b> ${r.missing_dates.join(", ")}</td></tr>`;}}e.innerHTML=h+"</table>";}
function renderMetadata(){const e=document.getElementById("metadata");let h="";for(const f of FREQS){h+=`<h3>${flabel(f)}</h3>`;for(const v of DATA.variables[f]){const r=DATA.metadata[f][v];h+=`<h4>${v}</h4>`;if(typeof r==="string"){h+=`<pre>${r}</pre>`;continue;}h+=`<div><b>Example file:</b> <code>${r.file}</code></div><pre>${r.fieldlist}</pre><pre>${r.xarray_info}</pre>`;}}e.innerHTML=h;}
function showPlot(){const el=document.getElementById("plots"),key=currentPlotFrequency+"|"+currentPlotVariable;if(!PLOTS[key]){el.innerHTML=`<p>No pre-calculated ${flabel(currentPlotFrequency)} statistics file was found for ${currentPlotVariable}.</p>`;return;}el.innerHTML='<div id="plotlyQC" style="width:100%;height:1300px;"></div>';const p=JSON.parse(PLOTS[key]);Plotly.newPlot("plotlyQC",p.data,p.layout,PLOT_CONFIG);}
function updatePlotVariableOptions(){const s=document.getElementById("plotVariableSelector"),a=DATA.variables[currentPlotFrequency];s.innerHTML="";a.forEach(v=>{const o=document.createElement("option");o.value=v;o.textContent=v;s.appendChild(o);});if(!a.includes(currentPlotVariable))currentPlotVariable=a[0];s.value=currentPlotVariable;}
function renderPlotFrequencyTabs(){const t=document.getElementById("frequencyTabs");t.innerHTML="";FREQS.forEach(f=>{const b=document.createElement("button");b.textContent=flabel(f);b.className=f===currentPlotFrequency?"active":"";b.onclick=()=>{currentPlotFrequency=f;renderPlotFrequencyTabs();updatePlotVariableOptions();showPlot();};t.appendChild(b);});}
document.getElementById("plotVariableSelector").addEventListener("change",e=>{currentPlotVariable=e.target.value;showPlot();});
function mapsFor(f,v){return DATA.maps.filter(m=>m.frequency===f&&m.field===v);}
function renderMapFrequencyTabs(){const t=document.getElementById("mapFrequencyTabs");t.innerHTML="";FREQS.forEach(f=>{const b=document.createElement("button");b.textContent=flabel(f);b.className=f===currentMapFrequency?"active":"";b.onclick=()=>{currentMapFrequency=f;currentIndex=0;updateFieldOptions();updateMap();renderMapFrequencyTabs();};t.appendChild(b);});}
function updateFieldOptions(){const s=document.getElementById("variableSelector"),a=fieldsFor(currentMapFrequency).filter(v=>mapsFor(currentMapFrequency,v).length);s.innerHTML="";a.forEach(v=>{const o=document.createElement("option");o.value=v;o.textContent=v;s.appendChild(o);});if(!a.includes(currentField))currentField=a[0]||fieldsFor(currentMapFrequency)[0];s.value=currentField;}
function renderPeriodButtons(items){const c=document.getElementById("periodButtons");c.innerHTML="";items.forEach((item,i)=>{const b=document.createElement("button");b.textContent=item.label;b.className=i===currentIndex?"active":"";b.onclick=()=>{currentIndex=i;updateMap();};c.appendChild(b);});}
function updateMap(){const items=mapsFor(currentMapFrequency,currentField),img=document.getElementById("map"),status=document.getElementById("mapStatus");if(!items.length){img.removeAttribute("src");document.getElementById("mapCounter").textContent="No map available";status.textContent="";renderPeriodButtons([]);return;}currentIndex=Math.max(0,Math.min(currentIndex,items.length-1));const item=items[currentIndex];img.src=item.image;document.getElementById("mapCounter").textContent=`${item.date} — ${currentIndex+1} / ${items.length}`;status.textContent=`Variable: ${currentField} | Frequency: ${flabel(currentMapFrequency)}`;renderPeriodButtons(items);}
document.getElementById("variableSelector").addEventListener("change",e=>{currentField=e.target.value;currentIndex=0;updateMap();});
document.getElementById("prevButton").onclick=()=>{const n=mapsFor(currentMapFrequency,currentField).length;if(n){currentIndex=(currentIndex-1+n)%n;updateMap();}};
document.getElementById("nextButton").onclick=()=>{const n=mapsFor(currentMapFrequency,currentField).length;if(n){currentIndex=(currentIndex+1)%n;updateMap();}};
document.addEventListener("keydown",e=>{if(e.key==="ArrowLeft")document.getElementById("prevButton").click();if(e.key==="ArrowRight")document.getElementById("nextButton").click();});
function spatialFor(f,v){return DATA.spatial.filter(m=>m.frequency===f&&(v===undefined||m.variable===v));}
function renderSpatialFrequencyTabs(){const t=document.getElementById("spatialFrequencyTabs");t.innerHTML="";FREQS.forEach(f=>{if(!spatialFor(f).length)return;const b=document.createElement("button");b.textContent=flabel(f);b.className=f===currentSpatialFrequency?"active":"";b.onclick=()=>{currentSpatialFrequency=f;currentSpatialVariable=null;currentSpatialStat=null;renderSpatialFrequencyTabs();updateSpatialMap();};t.appendChild(b);});}
function updateSpatialMap(){const img=document.getElementById("spatialMap"),status=document.getElementById("spatialStatus"),sel=document.getElementById("spatialVariableSelector"),c=document.getElementById("spatialStatButtons");const all=spatialFor(currentSpatialFrequency);sel.innerHTML="";c.innerHTML="";if(!all.length){img.removeAttribute("src");status.textContent="No spatial consistency file was found.";return;}const vars=[...new Set(all.map(m=>m.variable))];if(!vars.includes(currentSpatialVariable))currentSpatialVariable=vars[0];vars.forEach(v=>{const o=document.createElement("option");o.value=v;o.textContent=v;sel.appendChild(o);});sel.value=currentSpatialVariable;const items=spatialFor(currentSpatialFrequency,currentSpatialVariable);if(!items.some(m=>m.statistic===currentSpatialStat))currentSpatialStat=items[0].statistic;items.forEach(m=>{const b=document.createElement("button");b.textContent=m.label;b.className=m.statistic===currentSpatialStat?"active":"";b.onclick=()=>{currentSpatialStat=m.statistic;updateSpatialMap();};c.appendChild(b);});const item=items.find(m=>m.statistic===currentSpatialStat);img.src=item.image;status.textContent=`${currentSpatialVariable} — ${item.label} | Frequency: ${flabel(currentSpatialFrequency)} | Period: ${item.period}`;}
document.getElementById("spatialVariableSelector").addEventListener("change",e=>{currentSpatialVariable=e.target.value;currentSpatialStat=null;updateSpatialMap();});
renderIntegrity();renderMetadata();
renderPlotFrequencyTabs();updatePlotVariableOptions();showPlot();
updateFieldOptions();renderMapFrequencyTabs();updateMap();
if(!spatialFor(currentSpatialFrequency).length&&DATA.spatial.length)currentSpatialFrequency=DATA.spatial[0].frequency;
renderSpatialFrequencyTabs();updateSpatialMap();
</script>
</body></html>'''


def main():
    parser = argparse.ArgumentParser(description=f"Build the {DATASET_TITLE} QC gallery")
    parser.add_argument("--skip-maps", action="store_true",
                        help="Reuse the per-date map PNGs already on disk instead of "
                             "re-rendering them (for changes that only affect the HTML, "
                             "the plots or the spatial-consistency maps)")
    parser.add_argument("--skip-spatial", action="store_true",
                        help="Reuse the spatial-consistency PNGs already on disk")
    parser.add_argument("--no-upload", action="store_true",
                        help="Write the gallery locally without uploading it to the ECMWF Site")
    args = parser.parse_args()

    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    MAPS_DIR.mkdir(parents=True, exist_ok=True)

    integrity = {}
    metadata = {}
    tables = {}
    for f in FREQUENCIES:
        df = collect_files(f)
        tables[f] = df
        integrity[f] = {v: integrity_report(f, v, df) for v in PRODUCTS[f]}
        metadata[f] = {v: metadata_report(v, df) for v in PRODUCTS[f]}

    map_manifest = []
    missing_maps = []
    for f in FREQUENCIES:
        df = tables[f]
        dates = gallery_dates(f, df)
        for variable, fields in MAP_FIELDS[f].items():
            for field in fields:
                var_dir = MAPS_DIR / f / field
                var_dir.mkdir(parents=True, exist_ok=True)
                for i, date in enumerate(dates, 1):
                    fname = find_file(df, f, variable, date)
                    out = var_dir / f"map{i:02d}.png"
                    if not fname:
                        missing_maps.append({"frequency": f, "variable": variable,
                                             "field": field, "date": date.strftime("%Y-%m-%d")})
                        continue
                    label = date.strftime("%B %Y") + (
                        f" (day {GALLERY_DAY})" if f.startswith("daily") else "")
                    title = f"{PRODUCT} {field} — {FREQ_LABELS[f]} — {label}"
                    drawn = out.exists() if args.skip_maps else save_map(fname, field, out, title)
                    if drawn:
                        map_manifest.append({
                            "frequency": f,
                            "variable": variable,
                            "field": field,
                            "date": date.strftime("%Y-%m-%d"),
                            "label": label,
                            "image": str(out.relative_to(GALLERY_DIR)).replace(os.sep, "/"),
                        })

    spatial_manifest = build_spatial_manifest(skip_render=args.skip_spatial)

    plots = {}
    for f in FREQUENCIES:
        for variable in PRODUCTS[f]:
            stats_path = find_aux_file(f"tseries_stats_{variable}_{f}_*.nc")
            if stats_path is None:
                print(f"No pre-calculated statistics found for {variable} {f}")
                continue
            # The thresholds carry the frequency they were derived from.
            thresholds_path = find_aux_file(f"{variable}_p999_{f}_*.nc")
            fig = create_qc_figure(
                stats_path,
                f"{PRODUCT} {variable} — {FREQ_LABELS[f]} Quality Control",
                UNITS,
                thresholds_path=thresholds_path,
                variable=variable,
            )
            plots[f"{f}|{variable}"] = fig.to_json()

    payload = {
        "title": DATASET_TITLE,
        "frequencies": list(FREQUENCIES),
        "freq_labels": FREQ_LABELS,
        "variables": {f: PRODUCTS[f] for f in FREQUENCIES},
        "map_fields": {f: MAP_FIELDS[f] for f in FREQUENCIES},
        "integrity": integrity,
        "metadata": metadata,
        "maps": map_manifest,
        "missing_maps": missing_maps,
        "spatial": spatial_manifest,
    }
    html_text = HTML_TEMPLATE.replace(
        "__PAYLOAD__", json.dumps(payload, default=str).replace("</", "<\\/")
    ).replace(
        "__PLOTS__", json.dumps(plots).replace("</", "<\\/")
    ).replace(
        "__TITLE__", DATASET_TITLE
    )

    HTML_PATH.write_text(html_text, encoding="utf-8")
    TOP_LEVEL_HTML.write_text(html_text, encoding="utf-8")

    if args.no_upload:
        print("Upload skipped (--no-upload)")
    else:
        token = os.environ["ECMWF_SITES_TOKEN"]

        upload_qc_gallery(
        gallery_dir=f"./{GALLERY_DIR}",
        token=token,
        )

    print(f"Created {HTML_PATH}")
    print(f"Created {TOP_LEVEL_HTML}")
    print(f"Created {len(map_manifest)} maps")
    print(f"Created {len(spatial_manifest)} spatial consistency maps")
    if missing_maps:
        print(f"Missing map combinations: {len(missing_maps)}")


if __name__ == "__main__":
    main()
