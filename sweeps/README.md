# Sweep matrices

The configuration matrices the campaign ran, for `scripts/run_sweep.py`. Each
expands to one run per configuration and repetition; every run writes one
report.

```bash
python3 ~/aspir/scripts/run_sweep.py ~/aspir/sweeps/<matrix>.json --dry-run
python3 ~/aspir/scripts/run_sweep.py ~/aspir/sweeps/<matrix>.json
```

`--dry-run` prints the plan and says which configurations already have a
report. An interrupted sweep resumes by repeating the command. `--force`
re-measures everything, `--only <substring>` filters, `--keep-going` continues
past a failing run.

A configuration counts as already measured only if an existing report agrees on
everything that defines the measurement — precision, threads, timing-pass
length, energy durations, programmable-logic state and tag — so a quality run
never stands in for a timing run, nor a control for its pair.

## Ultra96-V2

| Matrix | What it measures |
|---|---|
| `e1_quality_grid.json` | Quality of the CPU FP32 and DPU INT8 outputs over all 14 experiments, one repetition, outputs saved. No timing pass: with fixed weights and inputs the output is deterministic |
| `e2_operating_point.json` | Latency with the 10,000-frame pass, tail quantiles, sustained throughput and energy at 8 % cake-cutting for both image classes, three repetitions |
| `e3_thread_scaling.json` | CPU latency at 1–4 threads and energy at 1, 2 and 4, with the DPU as reference |
| `e5_frame_budget.json` | Linear reconstruction time on the Arm cores, once per sensing family and ratio |
| `e7_controls_u96.json` | CPU INT8 latency and energy (data width versus architecture), and CPU FP32 energy with the bitstream loaded but idle (tag `pl_loaded`) |
| `e8_compute_plane_u96.json` | Energy at the operating point with both IRPS5401 controllers instrumented, for the DPU, the CPU and the CPU with the bitstream loaded (tags `dual_pmic`, `dual_pmic_pl_loaded`) |

## Jetson Orin NX

| Matrix | What it measures |
|---|---|
| `e1_quality_grid_jetson.json` | GPU INT8 quality across the grid, one repetition |
| `e2_operating_point_jetson.json` | Latency, tails, throughput and energy at the operating point, three repetitions, with the precision study |
| `e5_frame_budget.json` | Linear reconstruction time, once per sensing family and ratio |
| `e6_quality_strict_jetson.json` | GPU quality across the grid with `int8_strict` engines |
| `e7_controls_jetson.json` | CPU INT8 controls, GPU timing at the ends of the ratio sweep, power series, CPU FP32 outputs |
| `e8_int8_redo_jetson.json` | Every INT8 measurement repeated on `int8_strict` engines rebuilt from scratch |
| `e9_operating_point_clean.json` | The operating-point table re-measured from a clean working tree |

Build the engines a Jetson matrix needs before running it
(`scripts/build_trt_engine.py`), and pin the clocks.

## Fields

- `repetitions` — independent process launches; tables report medians across
  them.
- `warmup` — frames discarded before timing.
- `limit` — evaluation frames used for quality; 0 means all 100.
- `timing_frames` — length of the timing pass; the runners' default is 10,000.
- `dwell_s` — idle gap between runs, so a thermal transient does not carry into
  the next one.
- `tag` — marks a deliberate variant of an otherwise identical configuration.
