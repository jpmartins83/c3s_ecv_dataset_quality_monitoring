from pathlib import Path
import io
import json
import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
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
PRODUCT_PARAMETERS = {
    "delivery_start": pd.Timestamp("2026-01-01"),
    "delivery_end": pd.Timestamp("2026-06-01"),
    "collection_start": pd.Timestamp("2002-01-01"),
    "collection_end": pd.Timestamp("2026-06-01"),
}

DATADIR = Path("../../../datasets/GIRAFE")
FREQUENCIES = ("monthly", "daily")

MAP_VARIABLES = [
    "precipitation",
    "num_obs_fraction",
    "num_obs_rate",
    "num_days",
    "quality_flag",
    "num_days_snow",
]

GALLERY_DIR = Path("GIRAFE_QC_gallery")
MAPS_DIR = GALLERY_DIR / "maps"
HTML_PATH = GALLERY_DIR / "GIRAFE_QC_timeseries.html"
TOP_LEVEL_HTML = Path("GIRAFE_QC_timeseries.html")

GALLERY_MONTHS = pd.date_range(
    PRODUCT_PARAMETERS["delivery_start"],
    PRODUCT_PARAMETERS["delivery_end"],
    freq="MS",
)

MAP_STYLES = {
    "precipitation": dict(cmap="gist_earth_r", levels=[0, 0.5, 1, 2, 5, 10, 20, 50, 100]),
    "num_obs_fraction": dict(cmap="Greens", levels=np.linspace(0, 1, 11)),
    "num_obs_rate": dict(cmap="Purples", levels=np.linspace(0, 1, 11)),
    "num_days": dict(cmap="Blues", levels=np.arange(0, 32, 2)),
    "quality_flag": dict(cmap="tab10", levels=np.arange(-0.5, 6.5, 1)),
    "num_days_snow": dict(cmap="cool", levels=np.arange(0, 33, 2)),
}


def collect_files(frequency):
    files = [p for p in (DATADIR / frequency).rglob("*") if p.is_file()]
    df = pd.DataFrame({"file_path": [str(p) for p in files]})
    if frequency == "monthly":
        df["file_date"] = df["file_path"].str.extract(r"PREmm(\d{6})", expand=False)
    else:
        # GIRAFE daily pattern: PREdmYYYYMMDD...
        df["file_date"] = df["file_path"].str.extract(r"PREdm(\d{8})", expand=False)
    return df.sort_values("file_date", na_position="last").reset_index(drop=True)


def integrity_report(frequency, df):
    resolution = "MS" if frequency == "monthly" else "D"
    expected = pd.date_range(
        PRODUCT_PARAMETERS["collection_start"],
        PRODUCT_PARAMETERS["collection_end"],
        freq=resolution,
    )
    expected_keys = set(expected.strftime("%Y%m" if frequency == "monthly" else "%Y%m%d"))
    present = set(df["file_date"].dropna())
    missing = sorted(expected_keys - present)
    existing = sorted(expected_keys & present)
    return {
        "frequency": frequency,
        "files_found": len(df),
        "expected": len(expected_keys),
        "existing": len(existing),
        "missing": len(missing),
        "missing_dates": missing,
    }


def metadata_report(frequency, df):
    valid = df.dropna(subset=["file_date"])
    if valid.empty:
        return "No valid files found."

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


def create_qc_figure(stats_path, title):
    with xr.open_dataset(stats_path) as ds_stats:
        stats = ds_stats.to_dataframe().reset_index()

    stats["time"] = pd.to_datetime(stats["time"])

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
            "Data completeness and QC",
        ],
    )

    fig.add_trace(go.Scatter(x=stats["time"], y=stats["mean"], mode="lines", name="Mean", line=dict(width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=stats["time"], y=stats["p99"], mode="lines", name="P99", line=dict(width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=stats["time"], y=stats["median"], mode="lines", name="Median", line=dict(width=2, dash="dash")), row=2, col=1)
    fig.add_trace(go.Scatter(x=stats["time"], y=stats["p01"], mode="lines", name="P01", line=dict(width=1), fill="tonexty", fillcolor="rgba(100,100,100,0.15)"), row=2, col=1)
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
        go.Scatter(x=stats["time"], y=stats["negative_fraction"], mode="lines", name="Negative fraction", line=dict(dash="dash")),
        row=5, col=1,
    )
    fig.add_trace(
        go.Scatter(x=stats["time"], y=stats["outliers_fraction"], mode="lines", name="Outliers fraction", line=dict(dash="dot")),
        row=5, col=1,
    )

    for r, label in [(1, "Precipitation"), (2, "Precipitation"), (3, "Std"), (4, "Precipitation"), (5, "Fraction")]:
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


def find_file(df, frequency, date):
    key = date.strftime("%Y%m" if frequency == "monthly" else "%Y%m%d")
    matches = df.loc[df["file_date"] == key, "file_path"]
    return matches.iloc[0] if not matches.empty else None


def save_map(fname, variable, output_path, title):
    with xr.open_dataset(fname) as ds:
        if variable not in ds.data_vars:
            return False
        da = ds[variable]
        if "time" in da.dims:
            da = da.isel(time=0)
        da = da.squeeze(drop=True).load()

    fig = plt.figure(figsize=(10, 5.7))
    ax = plt.axes(projection=ccrs.PlateCarree())

    style = MAP_STYLES.get(variable, {})
    kwargs = {
        "ax": ax,
        "transform": ccrs.PlateCarree(),
        "cmap": style.get("cmap", "viridis"),
        "add_colorbar": True,
    }
    if "levels" in style:
        kwargs["levels"] = style["levels"]

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

def upload_girafe_qc_gallery(
    gallery_dir,
    token,
    target_path="dataset_qc/AtmPhysics/Precipitation/GIRAFE/",
    space="cxjo",
    site_name="ecv-info",
    ):
    """
    Upload the GIRAFE QC static gallery to an ECMWF Site.

    Parameters
    ----------
    gallery_dir : str or pathlib.Path
        Local directory containing GIRAFE_QC_timeseries.html and maps/.
        For example: "./GIRAFE_QC_gallery"

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

    html_file = gallery_dir / "GIRAFE_QC_timeseries.html"
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

    print("GIRAFE QC gallery uploaded successfully.")
    print(
        f"https://sites.ecmwf.int/{space}/{site_name}/"
        f"{target_path.rstrip('/')}/GIRAFE_QC_timeseries.html"
    )


HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GIRAFE Quality Control</title>
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
#map { display:block; max-width:100%; width:1000px; height:auto; margin:8px auto; }
.nav { display:flex; justify-content:center; gap:8px; margin:8px 0; } .center{text-align:center;} .small{color:#555;font-size:.92em;}
</style>
</head>
<body>
<h1>GIRAFE Quality Control</h1>
<div class="panel"><h2>Dataset integrity</h2><div id="integrity"></div></div>
<div class="panel"><h2>Metadata</h2><div id="metadata"></div></div>
<div class="panel"><h2>QC time series</h2><div class="tabs" id="frequencyTabs"></div><div id="plots"></div></div>
<div class="panel">
<h2>Spatial inspection</h2>
<div class="tabs" id="mapFrequencyTabs"></div>
<label>Variable: <select id="variableSelector"></select></label>
<div class="period-buttons" id="periodButtons"></div>
<div class="center small" id="mapCounter"></div>
<div class="nav"><button id="prevButton">← Previous</button><button id="nextButton">Next →</button></div>
<img id="map" alt="GIRAFE QC map"><div class="center small" id="mapStatus"></div>
</div>
<script>
const DATA = __PAYLOAD__;
const PLOTS = __PLOTS__;
const PLOT_CONFIG = {responsive:true, displaylogo:false};
let currentMapFrequency="monthly", currentVariable=DATA.map_variables[0], currentIndex=0;
function renderIntegrity(){const e=document.getElementById("integrity");let h="<table><tr><th>Frequency</th><th>Files found</th><th>Expected</th><th>Existing</th><th>Missing</th></tr>";for(const f of ["monthly","daily"]){const r=DATA.integrity[f];h+=`<tr><td>${f}</td><td>${r.files_found}</td><td>${r.expected}</td><td>${r.existing}</td><td>${r.missing}</td></tr>`;if(r.missing)h+=`<tr><td colspan="5"><b>Missing:</b> ${r.missing_dates.join(", ")}</td></tr>`;}e.innerHTML=h+"</table>";}
function renderMetadata(){const e=document.getElementById("metadata");let h="";for(const f of ["monthly","daily"]){const r=DATA.metadata[f];h+=`<h3>${f.toUpperCase()}</h3>`;if(typeof r==="string"){h+=`<pre>${r}</pre>`;continue;}h+=`<div><b>Example file:</b> <code>${r.file}</code></div><h4>Fields</h4><pre>${r.fieldlist}</pre><h4>xarray Dataset.info()</h4><pre>${r.xarray_info}</pre>`;}e.innerHTML=h;}
function showPlot(f){document.querySelectorAll("#frequencyTabs button").forEach(b=>b.classList.toggle("active",b.dataset.frequency===f));const el=document.getElementById("plots");if(!PLOTS[f]){el.innerHTML=`<p>No pre-calculated ${f} statistics file was found.</p>`;return;}el.innerHTML='<div id="plotlyQC" style="width:100%;height:1300px;"></div>';const p=JSON.parse(PLOTS[f]);Plotly.newPlot("plotlyQC",p.data,p.layout,PLOT_CONFIG);}
function renderPlots(){const t=document.getElementById("frequencyTabs");["monthly","daily"].forEach((f,i)=>{const b=document.createElement("button");b.textContent=f[0].toUpperCase()+f.slice(1);b.dataset.frequency=f;b.className=i===0?"active":"";b.onclick=()=>showPlot(f);t.appendChild(b);});showPlot("monthly");}
function mapsFor(f,v){return DATA.maps.filter(m=>m.frequency===f&&m.variable===v);}
function renderMapFrequencyTabs(){const t=document.getElementById("mapFrequencyTabs");t.innerHTML="";["monthly","daily"].forEach(f=>{const b=document.createElement("button");b.textContent=f[0].toUpperCase()+f.slice(1);b.className=f===currentMapFrequency?"active":"";b.onclick=()=>{currentMapFrequency=f;currentIndex=0;updateVariableOptions();updateMap();renderMapFrequencyTabs();};t.appendChild(b);});}
function updateVariableOptions(){const s=document.getElementById("variableSelector"),a=DATA.map_variables.filter(v=>mapsFor(currentMapFrequency,v).length);s.innerHTML="";a.forEach(v=>{const o=document.createElement("option");o.value=v;o.textContent=v;s.appendChild(o);});if(!a.includes(currentVariable))currentVariable=a[0]||DATA.map_variables[0];s.value=currentVariable;}
function renderPeriodButtons(items){const c=document.getElementById("periodButtons");c.innerHTML="";items.forEach((item,i)=>{const b=document.createElement("button");b.textContent=item.label;b.className=i===currentIndex?"active":"";b.onclick=()=>{currentIndex=i;updateMap();};c.appendChild(b);});}
function updateMap(){const items=mapsFor(currentMapFrequency,currentVariable),img=document.getElementById("map"),status=document.getElementById("mapStatus");if(!items.length){img.removeAttribute("src");document.getElementById("mapCounter").textContent="No map available";status.textContent="";renderPeriodButtons([]);return;}currentIndex=Math.max(0,Math.min(currentIndex,items.length-1));const item=items[currentIndex];img.src=item.image;document.getElementById("mapCounter").textContent=`${item.date} — ${currentIndex+1} / ${items.length}`;status.textContent=`Variable: ${currentVariable} | Frequency: ${currentMapFrequency}`;renderPeriodButtons(items);}
document.getElementById("variableSelector").addEventListener("change",e=>{currentVariable=e.target.value;currentIndex=0;updateMap();});
document.getElementById("prevButton").onclick=()=>{const n=mapsFor(currentMapFrequency,currentVariable).length;if(n){currentIndex=(currentIndex-1+n)%n;updateMap();}};
document.getElementById("nextButton").onclick=()=>{const n=mapsFor(currentMapFrequency,currentVariable).length;if(n){currentIndex=(currentIndex+1)%n;updateMap();}};
document.addEventListener("keydown",e=>{if(e.key==="ArrowLeft")document.getElementById("prevButton").click();if(e.key==="ArrowRight")document.getElementById("nextButton").click();});
renderIntegrity();renderMetadata();renderPlots();updateVariableOptions();renderMapFrequencyTabs();updateMap();
</script>
</body></html>'''


def main():
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    MAPS_DIR.mkdir(parents=True, exist_ok=True)

    integrity = {}
    metadata = {}
    tables = {}
    for f in FREQUENCIES:
        df = collect_files(f)
        tables[f] = df
        integrity[f] = integrity_report(f, df)
        metadata[f] = metadata_report(f, df)

    map_manifest = []
    missing_maps = []
    for f in FREQUENCIES:
        df = tables[f]
        for variable in MAP_VARIABLES:
            var_dir = MAPS_DIR / f / variable
            var_dir.mkdir(parents=True, exist_ok=True)
            for i, date in enumerate(GALLERY_MONTHS, 1):
                fname = find_file(df, f, date)
                out = var_dir / f"map{i:02d}.png"
                if not fname:
                    missing_maps.append({"frequency":f,"variable":variable,"date":date.strftime("%Y-%m-%d")})
                    continue
                title = f"GIRAFE {variable} — {f} — {date:%B %Y}" + (" (day 1)" if f == "daily" else "")
                if save_map(fname, variable, out, title):
                    map_manifest.append({
                        "frequency": f,
                        "variable": variable,
                        "date": date.strftime("%Y-%m-%d"),
                        "label": date.strftime("%B %Y") + (" (day 1)" if f == "daily" else ""),
                        "image": str(out.relative_to(GALLERY_DIR)).replace(os.sep, "/"),
                    })

    plots = {}
    for f in FREQUENCIES:
        stats_path = Path(f"GIRAFE_{f}_stats.nc")
        if stats_path.exists():
            fig = create_qc_figure(stats_path, f"GIRAFE {f.capitalize()} Quality Control")
            plots[f] = fig.to_json()

    payload = {
        "integrity": integrity,
        "metadata": metadata,
        "maps": map_manifest,
        "missing_maps": missing_maps,
        "map_variables": MAP_VARIABLES,
    }
    html_text = HTML_TEMPLATE.replace(
        "__PAYLOAD__", json.dumps(payload, default=str).replace("</", "<\\/")
    ).replace(
        "__PLOTS__", json.dumps(plots).replace("</", "<\\/")
    )

    HTML_PATH.write_text(html_text, encoding="utf-8")
    TOP_LEVEL_HTML.write_text(html_text, encoding="utf-8")

    token = os.environ["ECMWF_SITES_TOKEN"]

    upload_girafe_qc_gallery(
    gallery_dir="./GIRAFE_QC_gallery",
    token=token,
    )

    print(f"Created {HTML_PATH}")
    print(f"Created {TOP_LEVEL_HTML}")
    print(f"Created {len(map_manifest)} maps")
    if missing_maps:
        print(f"Missing map combinations: {len(missing_maps)}")


if __name__ == "__main__":
    main()
