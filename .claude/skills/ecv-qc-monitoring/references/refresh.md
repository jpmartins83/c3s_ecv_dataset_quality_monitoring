# Folding in a new delivery

The routine update after a new ICDR release on a dataset already monitored.

1. **Re-read `constraints.json`** (see `new-dataset.md` step 1) for the new last date per
   variable.
2. **Bump the end dates in both places**: `AVAILABILITY_END` in `download_data.py` and the
   per-variable `case` block in `compute_stats.slurm`. Update the "read on <date>" comment.
3. **Download.** Only the new blocks are fetched; everything on disk is skipped.
4. **Re-run the stats jobs.** For a later `--dend` the run *extends* the existing series instead
   of recomputing it. Do not pass `p999`: the thresholds are climatological and must not move
   because new files arrived - a jump in the outlier fraction has to mean the data moved.
   Pass `spatial` if the spatial-consistency maps should cover the new files too (their filename
   carries the date range, so a stale one is visible in `aux_files/`).
5. **Rebuild the gallery.** The map window follows the newest file automatically, per product.
6. **Check the run's report**: zero missing files, zero missing map combinations.

To force a rebuild from scratch instead, delete that quantity's aux file first - re-running over
the same date range recalculates and deletes the superseded file once the replacement is planned,
so it never appends twice.
