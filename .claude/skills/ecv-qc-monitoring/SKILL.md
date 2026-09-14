---
name: ecv-qc-monitoring
description: Build, extend or refresh the QC inspection of a C3S satellite ECV dataset in this repo - download from the CDS, compute QC statistics, render the QC gallery and publish it to the ECMWF Site. Use whenever the task mentions monitoring or inspecting an ECV/CDR/ICDR dataset, a new CDS dataset to onboard, a new delivery to fold in, a QC gallery, QC time series, outlier thresholds, spatial consistency maps, or any of the per-dataset folders under AtmPhysics/, AtmComposition/, Land*/ or Ocean/.
---

# ECV dataset quality monitoring

Every monitored dataset in this repo is the same three scripts in a folder named after the
product, plus two slurm wrappers. Nothing here is a library: the five (and counting) folders
are **maintained as copies of each other**, so a fix or a new idea belongs in all of them,
and a new dataset starts by copying the closest sibling rather than from scratch.

```
<Domain>/<ECV>/<PRODUCT>/
    download_data.py      + download_data.slurm     CDS -> datasets/<ECV>/<PRODUCT>/<freq>/<YYYY>/<MM>/
    compute_stats.py      + compute_stats.slurm     files -> aux_files/*.nc
    build_QC_gallery.py                             aux_files + files -> QC_timeseries.html + PNGs -> ECMWF Site
```

Existing folders, nearest-sibling order:

| Folder | Shape |
|---|---|
| [AtmPhysics/Clouds/CLARA-A3](AtmPhysics/Clouds/CLARA-A3) | richest: many products per frequency, several quantities per file, per-product end dates, streaming climatology |
| [AtmPhysics/SRB/CLARA-A3](AtmPhysics/SRB/CLARA-A3), [AtmPhysics/ERB/CLARA-A3](AtmPhysics/ERB/CLARA-A3) | one quantity per file, two frequencies |
| [AtmPhysics/Precipitation/GIRAFE](AtmPhysics/Precipitation/GIRAFE) | single product, two frequencies |
| [AtmPhysics/UTH](AtmPhysics/UTH) | single frequency, small record, minimal compute_stats |
| [AtmPhysics/Humidity/THP](AtmPhysics/Humidity/THP) | **not gridded**: release axis instead of frequency, latitude-height sections instead of maps, whole record read in one pass |

## Which job is this

- **New dataset to monitor** -> `references/new-dataset.md`. Read the CDS catalogue first; do not
  guess variable names or availability.
- **New delivery / ICDR release on a dataset already monitored** -> `references/refresh.md`.
  Short: bump the end dates, re-run download and stats, rebuild the gallery.
- **Changing how the gallery looks or what it checks** -> make the change in the sibling that
  needs it, then port it to the other four. `references/conventions.md` lists what must stay true.
- **Answering a question about the record itself** (is this step real? why is this series flat?)
  -> `references/conventions.md` has the known by-construction artefacts; check there before
  chasing a bug.

## Non-negotiables

These were each paid for once. Full reasoning in `references/conventions.md`.

1. **Thresholds are per frequency.** The climatological P0.1/P99.9 must come from the same
   frequency it judges, and the filename carries it: `<VAR>_p999_<freq>_<start>_<end>.nc`.
2. **White means NaN, and nothing else.** Every palette is trimmed of its near-white end by
   `qc_colormap()`; a palette that is white in its *middle* raises at start-up instead of being
   used. Diverging fields use `managua`/`berlin`, never a ColorBrewer ramp.
3. **The map window is per product**, not per frequency: `gallery_dates()` takes the per-product
   slice of the file table, because products in one dataset stop at different dates.
4. **Availability comes from `constraints.json`**, is hard-coded with the date it was read, and
   appears in both `download_data.py` (`AVAILABILITY_END`) and `compute_stats.slurm`.
5. **The jobs are restartable and incremental.** Download skips blocks already on disk; stats
   extend an existing series for a later `--dend` rather than recomputing it.
