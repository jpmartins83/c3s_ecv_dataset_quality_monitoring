"""Build an interactive QC gallery for the UTH daily analysis outputs."""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xarray as xr


STATS_FILE = Path("UTH_daily_stats.nc")
SPATIAL_FILE = Path("UTH_daily_spatial_consistency.nc")
GALLERY_DIR = Path("UTH_QC_gallery")


def write_spatial_maps():
    if not SPATIAL_FILE.exists():
        return []

    pages = []
    with xr.open_dataset(SPATIAL_FILE) as dataset:
        for variable, title, colorscale in (
            ("Missing_values", "Total Missing Values", "Magma_r"),
            ("Number_of_values", "Total Valid Values", "Viridis"),
            ("Max_value", "Maximum UTH [%]", "Inferno"),
            ("Min_value", "Minimum UTH [%]", "Viridis"),
        ):
            if variable not in dataset:
                continue
            data = dataset[variable].squeeze(drop=True)
            figure = go.Figure(go.Heatmap(
                x=data.lon.values,
                y=data.lat.values,
                z=data.values,
                colorscale=colorscale,
                colorbar_title=title,
            ))
            figure.update_layout(title=title, xaxis_title="Longitude", yaxis_title="Latitude")
            filename = f"{variable}.html"
            figure.write_html(GALLERY_DIR / filename, include_plotlyjs="cdn")
            pages.append((title, filename))
    return pages


def main():
    if not STATS_FILE.exists():
        raise SystemExit(f"Missing statistics file: {STATS_FILE}")
    GALLERY_DIR.mkdir(exist_ok=True)

    with xr.open_dataset(STATS_FILE) as dataset:
        stats = dataset.to_dataframe().reset_index()

    figure = make_subplots(
        rows=3, cols=2,
        subplot_titles=("Mean UTH", "Standard Deviation", "P01 / P99", "Missing Fraction",
                        "Valid Values", "Minimum / Maximum"),
    )
    for name, row, col in (("mean", 1, 1), ("std", 1, 2), ("missing_fraction", 2, 2),
                           ("number_of_values", 3, 1), ("minimum", 3, 2), ("maximum", 3, 2)):
        if name in stats:
            figure.add_trace(go.Scatter(x=stats.time, y=stats[name], name=name, mode="lines"),
                             row=row, col=col)
    for name in ("p01", "p99"):
        if name in stats:
            figure.add_trace(go.Scatter(x=stats.time, y=stats[name], name=name, mode="lines"),
                             row=2, col=1)
    figure.update_layout(height=900, title="Upper Tropospheric Humidity QC", showlegend=True)
    figure.update_yaxes(title_text="%", row=1, col=1)
    figure.update_yaxes(title_text="%", row=1, col=2)
    figure.update_yaxes(title_text="fraction", row=2, col=2)
    figure.update_yaxes(title_text="count", row=3, col=1)
    figure.update_yaxes(title_text="%", row=3, col=2)

    spatial_pages = write_spatial_maps()
    links = "".join(f'<li><a href="{filename}">{title}</a></li>'
                    for title, filename in spatial_pages)
    figure.write_html(GALLERY_DIR / "QC_timeseries.html", include_plotlyjs="cdn")
    (GALLERY_DIR / "index.html").write_text(
        "<h1>Upper Tropospheric Humidity QC</h1>"
        '<p><a href="QC_timeseries.html">Time series</a></p>'
        f"<ul>{links}</ul>",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()