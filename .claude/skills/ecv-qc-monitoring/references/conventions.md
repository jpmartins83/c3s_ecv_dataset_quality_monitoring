# Conventions, invariants and known artefacts

## Layout

```
<Domain>/<ECV>/<PRODUCT>/
    aux_files/                      statistics, thresholds, spatial consistency
    <PRODUCT>_QC_gallery/           maps/<freq>/<product>/<field>/mapNN.png
                                    spatial/<freq>/<quantity>/*.png
                                    QC_timeseries.html
    QC_timeseries.html              a copy at the top level
datasets/<ECV>/<PRODUCT>/<freq>/<YYYY>/<MM>/*.nc
```

The aux filename carries everything needed to find it again, and the gallery looks files up by
glob (`find_aux_file(f"{variable}_p999_{frequency}_*.nc")`) rather than by constructed name.

## The four rules and what they cost when broken

**Thresholds per frequency.** Until 2026-08-31 one monthly-derived threshold file served both
frequencies. Daily values spread much wider than monthly means, so every daily high-tail outlier
fraction was inflated 7-24x above the ~0.1% a P99.9 implies. GIRAFE had the mirror error -
daily thresholds reused for monthly - which pinned its monthly outlier fraction at exactly
0.000000 for all 294 months. After the fix every series sits at ~0.001.

**White means NaN.** `qc_colormap()` trims any colour within `WHITE_DISTANCE` (0.40 in RGB) of
white off the ends of the palette and paints NaN white; a palette that passes *through* white
raises at start-up. The cloud physical properties are 50-80% empty on a daily map (daylight- and
phase-limited), so a near-white lowest class makes the map unreadable. Diverging fields need a
dark-centred palette: the swaps made were RdYlBu_r -> managua_r (CPH), twilight_shifted ->
cividis_r (SZA), RdBu_r -> berlin (SRB net radiation, UTH mean_rh_delta).

**Per-product map window.** Taking it from the newest file of the *frequency* left CLARA-A3 LWP
and IWP - which stop five months before the cloud mask - with one map out of six and 160 missing
map combinations. Per product it is six of six and zero.

**Availability from constraints.json.** See `new-dataset.md` step 1.

## Memory and cost

A daily CLARA climatology pools ~1300 files x 1M pixels per calendar month. Concatenating that
and calling `DataArray.quantile` needs ~10.8 GB before numpy's own copies during the sort, and it
memory-killed the 32 GB jobs. `compute_stats.py` in Clouds streams the files and keeps only the
outer 0.1% from each end (`pool_climatology_tails` / `quantile_from_tails`) - exact for a
P0.1/P99.9, peaks at ~4 GB. **So `stride=` and `--mem=64G` are not needed there**; striding only
shortens the read. The other datasets still use the concatenating form.

The time series uses one worker per reserved CPU. Throughput saturates at 4-8 workers because
the NFS read, not the arithmetic, is the limit.

## Artefacts that are by construction, not bugs

Check these before investigating a suspicious series.

- **A `%` variable saturates.** For CLARA-A3 CFC and CPH the climatological P99.9 comes out at
  100, so the high-tail outlier fraction is near-degenerate.
- **An observation count that steps** blows past any threshold derived before the step. THP's
  `Q_num` shows tens of per cent "outliers" in 2019-2021 because Metop-C joined the
  constellation in 2019-08.
- **A record's a priori can change mid-series.** THP: ERA-Interim backs the CDR, ERA5 the ICDR,
  so any obs-minus-model series changes meaning at 2017-01.
- **Processing-chunk seams** are visible in a Hovmoller and are real. Read the version from the
  file's `product_version` attribute, never from the filename token - the token mapping is the
  provider's convention, not a contract.
- **Fields that are not independent data.** THP's reanalysis product repeats the observations'
  `Q_num` and `Q_samperr` exactly, its `Q_obssig` is identically zero, its `WQ` identically
  100%. The gallery asserts them as cross-checks rather than differencing them.

## The five siblings are copies

`build_QC_gallery.py` is maintained as five near-identical files. A change to the shared parts
(`qc_colormap`, `qc_colors`, `NAN_COLOR`, `WHITE_DISTANCE`, `gallery_dates`, `prune_stale_pngs`,
`upload_qc_gallery`, the HTML template) belongs in all five, even where it is only defensive -
that is how the white-means-NaN rule and the per-product window were rolled out. After porting,
re-render and re-upload each one.
