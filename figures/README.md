# Figure pipeline

```
aspir-edge-data/results     ──scripts/aggregate_results.py──▶  measurements_<board>.csv
aspir-edge-data/results
aspir-edge-data/datasets    ──figures/derive.py─────────────▶  other tables, strip_*.npz, patterns.npz
tables                      ──figures/plots.py──────────────▶  figures/output/*.pdf, *.png
```

Only the aggregator and `derive.py` read reports; the plots read tables alone.
`aspir-edge-data/derived/` holds the tables as published, so every figure can be
drawn without re-deriving anything.

## Running

```bash
pip install -r envs/host/requirements.txt
export ASPIR_EDGE_DATA=/path/to/aspir-edge-data

# Draw from the published tables.
ASPIR_FIGURE_DATA=$ASPIR_EDGE_DATA/derived python figures/plots.py          # the article's figures
ASPIR_FIGURE_DATA=$ASPIR_EDGE_DATA/derived python figures/plots.py --list   # all of them
ASPIR_FIGURE_DATA=$ASPIR_EDGE_DATA/derived python figures/plots.py --only energy threads

# Or re-derive the tables from the reports first.
mkdir -p figures/data
python figures/derive.py
for b in jetson ultra96v2; do
    python scripts/aggregate_results.py $ASPIR_EDGE_DATA/results/$b --out figures/data/measurements_$b.csv
done
python figures/plots.py
```

`ASPIR_FIGURE_DATA` defaults to `figures/data/`; output goes to
`figures/output/`. Re-deriving from aspir-edge-data reproduces every table in its
`derived/` directory.

## Figures

| `--only` name | Output | In the article |
|---|---|---|
| `budget` | `fig_pipeline` | yes |
| `dataset_samples` | `fig_dataset_samples` | yes |
| `error_floor` | `fig_error_floor` | yes |
| `latency` | `fig_latency` | yes |
| `overview_thumbs` | `thumbs/*.png`, embedded in the overview diagram | yes |
| `qualitative` | `fig_qualitative` | yes |
| `quality` | `fig_quality` | yes |
| `energy` | `fig_energy` | no |
| `fidelity` | `fig_fidelity` | no |
| `power_trace` | `fig_power_trace` | no |
| `precision` | `fig_precision` | no |
| `qualitative_struct` | `fig_qualitative_struct` | no |
| `threads` | `fig_thread_scaling` | no |

## What needs CelebA frames

CelebA images may not be redistributed, so from the published data these are
skipped or partial:

- the structured-class row of `dataset_samples` and the face thumbnail of
  `overview_thumbs`;
- `qualitative_struct`;
- re-deriving the CelebA rows of `fp32_reference.csv` and
  `quality_reference.csv` (the published rows are carried over from the article's
  tables).

Producing them needs the CelebA subset rebuilt locally
(`aspir-edge-data/celeba/`) and the CelebA experiments re-run to obtain their
outputs.

## Files

| File | Role |
|---|---|
| `common.py` | Conventions shared by every table and figure: route and precision labels, filters, medians, bootstrap intervals, figure style |
| `derive.py` | Reports and datasets to tables: latency quantiles, per-frame quality with bootstrap intervals, energy per plane, engine composition, sensing configuration, image strips |
| `plots.py` | Tables to figures, one function per figure |
| `derive_clean_beams.py` | Regenerates the pre-speckle beam frames with ASPIR's own generator. Run it from an ASPIR v1.2.1 source tree: `python derive_clean_beams.py <irbeam test_images.npz> <output dir>` |

## Guards

Three mistakes are prevented in code, because each produces a plausible wrong
number:

- **Timing pass.** Some runs are labelled latency but timed only the
  ~100-frame quality loop, where a p99 is effectively the maximum.
  `common.is_timing_pass` keeps them out of every latency figure.
- **Clock state.** `common.load` drops runs whose clocks could move: a Jetson
  run must have its clocks pinned; an Ultra96-V2 run qualifies because the board
  has no frequency scaling (`cpu_dvfs: false`).
- **Pattern polarity.** `derive.py` reads polarity from the sign of the mask
  values and checks it against the configuration, since bipolar Hadamard
  patterns need twice as many projections as binary ones.
