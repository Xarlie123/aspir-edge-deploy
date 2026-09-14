# Provenance

This repository was assembled from three working repositories. Paths were
adapted to the layout here; logic was changed only where listed below.

## Where each part comes from

| Part | Source | Commit |
|---|---|---|
| `src/aspir_unet/`, `scripts/`, `sweeps/` | Implementation repository, `app/` | `b95a1da5ef40`, clean tree |
| `models/*/phase_c_dpu/` | Implementation repository, `app/models/`: xmodels, fixed points and logs of the campaign build of 2026-08-24; quantiser provenance files added in `12354dc` and `40647c1` | `b95a1da5ef40` |
| `hardware/dpu_overlay/` | Implementation repository, `workspace/hw/phase_c_dpu_vivado/` and `workspace/scripts/fetch-dpu-trd.sh`; bitstream and XSA of 2026-05-10 | `b95a1da5ef40` |
| `envs/ultra96v2/petalinux/` | Implementation repository, PetaLinux project `project-spec/meta-user/` and `project-spec/configs/` | `b95a1da5ef40` |
| `figures/` | Manuscript repository, `figures/scripts/` | `68e0a02`, unchanged at `4f600d5` |
| `src/aspir_unet/model/unet.py`, `unet_res.py` | Byte-identical to ASPIR v1.2.1 | ASPIR `08f773c76f70` |

## Adaptations made here

- Layout: `app/{src,scripts,sweeps,models}` moved to the top level;
  `workspace/hw/phase_c_dpu_vivado/` became `hardware/dpu_overlay/`; the DPU IP is
  fetched into `hardware/dpu-trd/`.
- `scripts/link_data.sh` is new: it links per-experiment inputs from
  aspir-edge-data. `scripts/build_dpu_xmodels.sh` mounts that checkout in the
  container so the links resolve.
- `scripts/quantize_dpu_int8.py`: `--model-dir` is required; its default pointed
  at a pre-campaign experiment.
- `scripts/deploy.sh`: the board address is required, and the deployment stamp
  records the repository name instead of the absolute host path.
- `scripts/install-board-deps.sh`: also installs onnx 1.17.0, which the campaign
  installed by hand.
- `hardware/dpu_overlay/gen_dpu_overlay.sh` requires `XILINX_DIR`;
  `fetch-dpu-trd.sh` no longer carries a placeholder archive hash that could
  never match.
- `figures/`: input and output locations are configurable (`ASPIR_EDGE_DATA`,
  `ASPIR_FIGURE_DATA`) and outputs go to `figures/output/`; the plots skip the
  CelebA panels whose frames are not published; `derive_clean_beams.py` takes
  its paths as arguments.
- `src/aspir_unet/model/__init__.py` no longer exposes the quantisation-aware
  training mirrors; a comment about mirroring ASPIR was removed from `unet.py`,
  which makes the file byte-identical to ASPIR's.
- Code the article does not use was left out.

## Code that produced the published measurements

Each run report records its code in `run.code`, stamped by `deploy.sh` when the
code was deployed. Per-report detail: `provenance/runs.csv` in aspir-edge-data.

### Jetson Orin NX — as recorded

| Recorded commit | Working tree | Reports | Sweeps |
|---|---|---|---|
| `32d8f6d` | modified | 96 | e1 quality grid, e2 operating point, e5 frame budget |
| `8878fec` | modified | 41 | e2 operating point (repeat), e6 strict-INT8 quality, e7 controls, and runs outside a sweep |
| `402150f` | modified | 26 | e8 INT8 redo |
| `2e23b61` | clean | 84 | e9 clean-tree operating point, and runs outside a sweep |
| `911f4f6` | modified | 3 | reconstruction energy |

For the runs deployed from a modified tree, which files were modified was not
recorded.

### Ultra96-V2 — reconstructed

**Every Ultra96-V2 report records `c6bec1b` with a clean tree, and that is not
the code that ran.** During the campaign, fixes were copied to the board without
re-running `deploy.sh`, so the stamp kept describing the first deployment. The
code that actually ran is reconstructed from fingerprints each fix left in the
reports, from launch times, and from what was synchronised; for every file
executed on the board it matches the commit shown.

| Effective code | Reports | Sweeps | Evidence |
|---|---|---|---|
| `a87c0ea` | 54 | e1 quality grid (28), e2 operating point (20), first launch of e3 thread scaling (6) | Reports carry `clocks.cpu_dvfs` and `device.soc_compatible`, introduced in `008e20f` and `a87c0ea`; all launched before the next synchronisation |
| `92840dc` | 52 | e2 operating point (4), e3 thread scaling (9), e5 frame budget (21), e7 controls (18) | Launched from 08:25 UTC on 2026-08-30; CPU INT8 reports carry their precision in the file name, introduced in `92840dc`; e2 re-ran exactly the four configurations the previous resume logic had skipped |
| `f7028de` | 9 | e8 compute plane | Reports carry `irps5401_pl.*` and `irps5401_ps.*` rails, introduced in `f7028de` |

## Defects in the measurement code, and what they did to published data

1. **Board input on the nine e8 reports** (fixed in `b95a1da`). After the two
   IRPS5401 controllers were renamed, the board-input filter matched their
   front-end rails, so `board_input_w` and the derived energies of those reports
   are inflated by about 0.76 W. The per-rail averages are correct.
   `scripts/aggregate_results.py` recomputes board input from them and flags the
   corrected rows (`board_input_recomputed`); no other report changes.
2. **CPU INT8 report names** (fixed in `92840dc`). The CPU runner left the
   precision out of the file name, so 25 Jetson CPU INT8 reports were named like
   FP32 ones. aspir-edge-data publishes them under corrected names
   (`provenance/renamed_reports.csv`). Their contents were always correct, and
   aggregation reads the precision from the contents.
3. **Sweep resume** (fixed in `92840dc`). Resume matched reports by file name and
   skipped runs it should have measured. The skipped configurations were
   measured before publication; no published report is affected.
4. **Aggregation keys** (fixed in `c57829f`). Timing-pass length and tag were not
   grouping keys, so summaries could pool a quality run with a timing run, or a
   control with its pair. Affects summaries only, not reports.
