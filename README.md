# aspir-edge-deploy

Deployment and measurement code for the article

> C. Chabert-Ull, H. Tobón-Maya, S. I. Zapata-Valencia, E. Tajahuerce, G. León.
> *Quantised U-Net Denoising for Single-Pixel Imaging on Embedded GPU and
> Reconfigurable Logic.* Submitted to The Journal of Supercomputing.

The article deploys one U-Net denoiser for single-pixel imaging in INT8 on two
embedded platforms — a Vitis AI DPU on an Avnet Ultra96-V2 and TensorRT on an
NVIDIA Jetson Orin NX — each compared against its own Arm CPU, and measures
latency, energy and reconstruction quality over two image classes, three
sensing families and five sampling ratios.

This repository holds what is needed to rebuild the deployed models and the
accelerator overlay, re-run every measurement, and regenerate the article's
tables and figures.

| Companion | Contents | Licence |
|---|---|---|
| **aspir-edge-data v1.0.0**, [doi:10.5281/zenodo.22761771](https://doi.org/10.5281/zenodo.22761771), [GitHub](https://github.com/Xarlie123/aspir-edge-data) | Datasets, checkpoints, ONNX exports, sensing masks, reconstructions, model outputs and every measurement report | CC BY 4.0 |
| **ASPIR v1.2.1**, [doi:10.5281/zenodo.19864089](https://doi.org/10.5281/zenodo.19864089) | The simulator that generated the datasets and trained the networks | Apache-2.0 |

## Contents

| Path | What it holds |
|---|---|
| `src/aspir_unet/` | Measurement package: model definition, runners for ONNX Runtime, VART and TensorRT, latency and energy sampling, quality metrics, run provenance |
| `scripts/` | Entry points: DPU quantisation and compilation, TensorRT engine building, CPU INT8 quantisation, the measurement runners, reconstruction timing, the sweep driver, aggregation, deployment |
| `sweeps/` | The configuration matrices of the campaign, as run |
| `models/` | The 14 compiled DPU models the Ultra96-V2 executed, with fixed points and quantiser provenance |
| `hardware/dpu_overlay/` | Block design, project regeneration, constraints, bitstream and XSA of the DPU overlay |
| `envs/` | Pinned environments: host, Vitis AI container, Ultra96-V2 PetaLinux image, Jetson |
| `figures/` | The pipeline from the reports in aspir-edge-data to the article's tables and figures |
| `PROVENANCE.md` | Which commit each part comes from, which code produced each set of measurements, and the defects fixed since |

## Platforms and versions

| | Ultra96-V2 | Jetson Orin NX | Host |
|---|---|---|---|
| Accelerator | DPUCZDX8G v4.1, B512, 300 MHz, fingerprint `0x101000056010400`, on a Zynq UltraScale+ XCZU3EG | Orin NX GPU (Engineering Reference Developer Kit Super) | — |
| CPU | 4 × Cortex-A53, fixed at 1.2 GHz (the kernel has no cpufreq driver) | Cortex-A78AE; clocks pinned with `jetson_clocks` (CPU 1497.6 MHz, GPU 1173 MHz as recorded) | x86-64 |
| OS | PetaLinux 2022.2, kernel 5.15.36-xilinx-v2022.2 | JetPack 6.2 (L4T R36.4.3), Ubuntu 22.04.5, kernel 5.15.148-tegra | Ubuntu 24.04 |
| Python | 3.9.9 | 3.10.12 | 3.12.3 |
| Accelerator runtime | VART 3.0.0, vivado-flow (`/dev/dpu`) | TensorRT 10.3.0, cuda-python | — |
| CPU runtime | ONNX Runtime 1.17.3 | ONNX Runtime 1.17.3 | — |
| INT8 toolchain | Vitis AI 3.0 container on the host: `vai_q_pytorch` 3.0.0+a44284e, PyTorch 1.12.1, Python 3.7.12 | TensorRT calibration on the board | — |
| Hardware tools | Vivado 2022.2 | — | — |
| Power telemetry | PMBus: IR38060 (board input), two IRPS5401 (PL and PS rails) | INA3221: `VDD_IN`, `VDD_CPU_GPU_CV`, `VDD_SOC` | — |

The networks were trained with ASPIR v1.2.1 on PyTorch 2.6.0 (CUDA 12.4); the
PyTorch that quantises for the DPU is therefore not the one that trained the
model, and the quality of the FP32 reference computed under 2.6.0 (41.505 dB)
and measured on the board under ONNX Runtime (41.534 dB) agree to 0.03 dB.
Exact package pins are in `envs/`.

## Setting up

```bash
export ASPIR_EDGE_DATA=/path/to/aspir-edge-data
cd aspir-edge-deploy
scripts/link_data.sh        # models/<id>/source -> $ASPIR_EDGE_DATA/datasets/<id>/source
```

A runner resolves an experiment from `models/<id>/source/` (checkpoint, ONNX,
evaluation and calibration frames) and `models/<id>/phase_c_dpu/` (compiled DPU
model).

Experiment identifiers are `<class>_64_unet_<configuration>_8_16`: class
`irbeam` (analytic IR beam profiles) or `celebA` (CelebA-derived frames);
configuration `cch_<r>pct` (Hadamard cake-cutting at r % sampling, r in
1, 2, 4, 8, 16), `scatter` (random patterns) or
`sweep_4degrees_2_bar_width_1_stride` (bar sweep). `8_16` is the U-Net width.

**Only the seven beam-profile experiments run from the published data.** CelebA
images may not be redistributed, so aspir-edge-data ships the CelebA
experiments' checkpoints, ONNX exports and masks but not their frames; its
`celeba/` directory documents how to rebuild the subset from a local copy.

## Reproducing each kind of measurement

### DPU models — host

```bash
scripts/build_dpu_xmodels.sh --force irbeam_64_unet_cch_8pct_8_16
```

Post-training quantisation with `vai_q_pytorch` on the 200 calibration frames of
the training split, then compilation with `vai_c_xir` against
`hardware/dpu_overlay/arch.json`, inside the pinned container. **A rebuilt
xmodel is a different binary holding the same model** — the compiler embeds a
build identifier. Compare fixed points, not hashes: `models/README.md`.

### DPU overlay — host, Vivado

`hardware/dpu_overlay/README.md`. The DPU IP is downloaded from AMD and is not
part of this repository.

### Ultra96-V2 image

`envs/README.md`: the PetaLinux 2022.2 user layer and configuration, the VART
vivado-flow build, and the `aspir-load-dpu` helper that loads the overlay at
runtime.

### TensorRT engines — on the Jetson

```bash
python3 scripts/build_trt_engine.py --experiment <id> --precision int8_strict
```

| `--precision` | Builder configuration |
|---|---|
| `fp32` | FP32 only |
| `fp16` | FP16 allowed |
| `int8` | INT8 allowed with FP16 fallback; the builder chooses the precision of each layer |
| `int8_nofp16` | INT8 allowed, FP16 fallback disabled |
| `int8_strict` | INT8 pinned on every compute layer, precision constraints obeyed — the INT8 GPU configuration reported in the article |

INT8 engines calibrate on the same 200 training-split frames as the DPU
(`--calib-npz`, default the experiment's `calibration_images.npz`). A
precision flag is a builder setting, not a guarantee: every run records the
per-layer precision composition of the engine it used, and
`--require-int8-fraction` / `--require-forced-layers` discard an engine whose
composition falls short. Serialised engines are tied to the GPU and TensorRT
build that produced them and are not distributed.

### CPU INT8 models — on each board

```bash
python3 scripts/quantize_cpu_int8.py --experiment <id>
```

ONNX Runtime `quantize_static`: QDQ format, per-tensor, asymmetric uint8
activations, symmetric int8 weights, ONNX Runtime's default calibration method,
the same 200 calibration frames. Run it on the board that executes the model,
so the graph is written by the ONNX Runtime version that reads it.

### Measurements — on each board

```bash
scripts/deploy.sh petalinux@<board-address>        # code, sweeps, models -> ~/aspir
ssh petalinux@<board-address>
aspir-load-dpu                                     # Ultra96-V2, once per boot
python3 ~/aspir/scripts/probe_device.py            # environment check
python3 ~/aspir/scripts/run_sweep.py ~/aspir/sweeps/e2_operating_point.json --dry-run
python3 ~/aspir/scripts/run_sweep.py ~/aspir/sweeps/e2_operating_point.json
```

On the Jetson, pin the clocks first (`sudo jetson_clocks`). Single runs:
`run_cpu_ort.py`, `run_dpu_int8.py`, `run_gpu_trt.py` and `run_recon_timing.py`,
each with `--experiment <id>` and `--energy-mode` for an energy run. Reports
land in `~/aspir/results/<id>/`.

What the runners and matrices encode:

- **Latency**: batch 1, 20 warm-up frames, a 10,000-frame timing pass, three
  independent process launches per configuration. The timed boundary includes
  the host-side int8 conversion on the DPU and the host–device copies on the GPU.
- **Quality**: PSNR, SSIM and MAE per frame over the 100 evaluation frames,
  against ground truth and against the host FP32 output, with outputs clipped to
  [0, 1].
- **Energy**: a 10 s idle phase and a 30 s phase of sustained inference, every
  rail sampled throughout and the power series kept, together with the
  completion time of every frame of the load phase.
- **Isolation**: the CPU runner unloads the programmable logic before measuring,
  so the CPU baseline is not taken with the accelerator powered; the DPU runner
  reloads it.
- **Provenance**: each report records parameters, code identity, software
  versions, device, clock state, accelerator identity and the sha256 of every
  input.
- **Resume**: `run_sweep.py` accepts an existing report only if it measured the
  same thing — precision, threads, timing-pass length, energy durations,
  programmable-logic state and tag — not merely because a file of that name
  exists.

### Power rails

| Board | Rail | Meaning |
|---|---|---|
| Ultra96-V2 | `ir38060.pout1` | Board input (12 V front-end regulator) |
| | `irps5401_pl.pout1`…`pout5` | PL controller, I²C 0x13/0x43: VCCAUX 1.8 V, VCCO_HP 1.2 V, VCCO_PSDDR 1.1 V, **VCCINT** (`pout4`), DisplayPort LDO |
| | `irps5401_ps.pout1`…`pout5` | PS controller, I²C 0x14/0x44: VCCPSAUX 1.8 V, VCC_PSINTLP, VCCO 3.3 V, **VCC_PSINTFP** (`pout4`), PSPLL LDO |
| Jetson Orin NX | `ina3221.VDD_IN` | Module input |
| | `ina3221.VDD_CPU_GPU_CV` | CPU, GPU and accelerator domain |
| | `ina3221.VDD_SOC` | SoC domain |

The IRPS5401 `pin*` channels are the controllers' inputs. Controller roles are
from the Avnet Ultra96-V2 schematic (sheets 15 and 16). Reports whose rails are
named bare `irps5401.*` predate the two controllers being told apart and contain
the PS controller only.

### Uncertainty of the dynamic energy

Energy reports carry an `energy_mode.uncertainty` block that follows the
convention of ASPIR (`src/aspir_unet/measurement/uncertainty.py` holds the
derivation).

- The dynamic energy per frame is E_dyn = (P̄ − P_idle) · t̄, over 10 blocks of
  the load phase. Both phases drop their first 2.5 s, as the figures do.
- The idle term is σ_idle · t̄, with σ_idle the standard deviation of the 1 s
  means of the idle phase (ASPIR polls its idle baseline at 1 Hz). It is not
  the standard error; σ_idle/√n is stored as well.
- The block term propagates the block-to-block spread of P and t to first
  order.
- The two are also combined, and a 95 % bootstrap interval is given.
- A third term covers sensor resolution, Δ/√6 · t̄. It is decisive for
  `ir38060.pout1`, which reads in 1.25 W steps (5.00 or 6.25 W), so the DPU's
  dynamic power on the board-input plane equals one step. The IRPS5401 rails
  resolve about 31 mW and the Jetson's INA3221 about 15 mW.

The same fields can be recomputed from archived reports, which have the power
series but no frame times; those use the sustained frame rate for every block:

```bash
python3 scripts/energy_uncertainty.py $ASPIR_EDGE_DATA/results/ultra96v2 $ASPIR_EDGE_DATA/results/jetson --out uncertainty.csv
python3 tests/test_uncertainty.py
```

### Aggregation and figures

```bash
python3 scripts/aggregate_results.py $ASPIR_EDGE_DATA/results/ultra96v2 --out ultra96v2.csv
python3 scripts/aggregate_results.py $ASPIR_EDGE_DATA/results/ultra96v2 --out summary.csv --summary
```

The aggregator recomputes board input from the per-rail averages instead of
trusting a report's own total (see `PROVENANCE.md`). Figures and tables:
`figures/README.md`.

## Not included, and why

| Item | Reason |
|---|---|
| DPU IP (DPUCZDX8G_VAI_v3.0) | AMD distributes it under its own licence; `hardware/dpu_overlay/fetch-dpu-trd.sh` downloads it |
| Avnet `meta-avnet` layer and BSP hardware exports | Third-party; obtain them from Avnet's Ultra96-V2 PetaLinux 2022.2 BSP |
| TensorRT engines | Bound to the GPU and TensorRT build that serialised them |
| TensorRT calibration caches | Written on the Jetson next to the engines and never copied off the board; not available for release. Building an engine recalibrates from `calibration_images.npz` |
| CPU INT8 ONNX models | Generated on each board and not archived; `scripts/quantize_cpu_int8.py` regenerates them |
| Datasets, checkpoints, ONNX exports, reports | In aspir-edge-data |
| Training | In ASPIR |
| Code the article does not use | Quantisation-aware training for other FPGA flows, superseded runners, and the tooling that imported ASPIR's outputs into the working repository |

## Licence and citation

Apache License 2.0 (`LICENSE`). `hardware/dpu_overlay/constraints/u96v2_sbc_base.xdc`
derives from Avnet's Ultra96-V2 base design. Citation metadata: `CITATION.cff`.

Version 1.0.0 is archived on Zenodo as [doi:10.5281/zenodo.22761898](https://doi.org/10.5281/zenodo.22761898).
