# Onboarding a new ECV dataset

Work in this order. Steps 1-2 are cheap and prevent most of the rework.

## 1. Read the CDS catalogue, do not guess

```bash
curl -sL https://cds-dev-cci2.copernicus-climate.eu/api/catalogue/v1/collections/<dataset>/form.json
curl -sL https://cds-dev-cci2.copernicus-climate.eu/api/catalogue/v1/collections/<dataset>/constraints.json
```

`-L` is required (307 without it). The `api/retrieve/v1/...` path is 404 and the processes route
is POST-only, so use the catalogue paths.

- **form.json** gives the exact machine values: variable names, `product_family`, `origin`,
  `time_aggregation`.
- **constraints.json** gives the valid combinations. Expand year x month x day per
  (time_aggregation, variable) to get the **first and last available date per variable** - the
  form's year list does not tell you this, and products within one dataset routinely stop at
  different dates. Requesting a date outside the constraints fails the whole request.

Record what you read and the date you read it, in a comment next to the hard-coded table.

## 2. Settle the scope with the user

Decisions the code cannot make and that are expensive to reverse:

- **Which quantities are monitored.** Not the same as which files are downloaded: one file can
  carry several geophysical quantities (CLARA-A3 `CTO` -> CTT, CTH, CTP).
- **Which frequencies.** Daily is one to two orders of magnitude more data (~1.5 TB for CLARA-A3
  clouds daily against ~60 GB monthly).
- **What gets integrity-and-metadata only.** A product whose shape does not fit 2-D statistics
  (CLARA-A3 `JCH`, a 6-D histogram on its own grid) still belongs in the integrity and metadata
  tables but has no time series and no maps.

## 3. Copy the nearest sibling

Pick from the table in SKILL.md, copy the whole folder, then work through the configuration
blocks. Each script has exactly one block to change; the rest is generic.

**download_data.py** - `DATASET`, `PRODUCT`, `ECV`, `PRODUCT_FAMILY`, `ORIGIN`, `VARIABLES`
(CDS name -> file token), `FREQUENCY_VARIABLES`, `AVAILABILITY_END`, `CDR_END_YEAR`,
`FREQ_TOKENS`/`FILE_TOKENS`, `FILE_PATTERN`.
Look up filenames rather than constructing them - the platform code in the tail changes through
the record, and a constructed name silently reports a present file as missing.

**compute_stats.py** - `DATASET`, `PRODUCT`, `ECV`, and `MONITORED`:
`{key: dict(prefix=<file token>, nc_var=<name inside the file>, units=<...>)}`. The three are
different names; the key is what appears in aux filenames and in the gallery.

**build_QC_gallery.py** - the block at the top: `DOMAIN`, `ECV`, `PRODUCT`, `DATASET_TITLE`,
`DATADIR`, `UPLOAD_TARGET`, `FREQUENCIES`, `PRODUCTS`, `MONITORED`, `QC_SERIES`, `_MAP_FIELDS`,
`MAP_STYLES`, `SPATIAL_MAPS`, `COLLECTION_START`, `FILE_PATTERN`. The dataset name appears in
folder names only, never in produced filenames, so the gallery layout is identical everywhere.

**both .slurm files** - the usage header is the real documentation of the dataset (variable list
with units and source file, the group expansions, the per-variable end dates, the
regenerate-everything loop). Rewrite it; do not leave the sibling's.

## 4. Download

One job per product, they are independent:

```bash
for P in <tokens>; do sbatch download_data.slurm monthly_mean $P; done
```

Resumable - a block whose files are all on disk is skipped, so resubmit a job that timed out.

## 5. Statistics

One job per (frequency, product file). Quantities sharing a file must go in one job (the file
read from NFS is the cost, not the arithmetic); quantities from different files cannot.

```bash
sbatch compute_stats.slurm daily_mean all_CTO spatial
```

`spatial` is **not** optional on a first run - without it no `spatial_consistency_*.nc` is
written and the gallery's spatial tab reports nothing found. Thresholds derive automatically
when absent; `p999` only forces a re-derivation (for a new CDR, not for appended files).

Each job writes into `aux_files/`:

```
tseries_stats_<var>_<freq>_<dstart>_<dend>.nc
<var>_p999_<freq>_<climstart>_<climend>.nc
spatial_consistency_<var>_<freq>_<dstart>_<dend>.nc
```

## 6. Gallery

```bash
python build_QC_gallery.py --no-upload     # check locally first
ECMWF_SITES_TOKEN=... python build_QC_gallery.py
```

Flags: `--skip-maps`, `--skip-spatial` (reuse PNGs on disk, for HTML-only changes), `--no-prune`,
`--no-upload`. Published to `https://sites.ecmwf.int/cxjo/ecv-info/dataset_qc/<DOMAIN>/<ECV>/<PRODUCT>/`.

## 7. Read the run's own report

The tail of the run is the acceptance test: `Missing map combinations` should be 0, and the
integrity table should report zero missing files. A non-zero count is usually a real
configuration error (a per-frequency window, a field that product does not carry), not noise.
