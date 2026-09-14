# Compiled DPU models

One directory per experiment of the article's sensing grid (7 sensing
configurations × 2 image classes), each holding the model the Ultra96-V2
executed:

| File | What it is |
|---|---|
| `phase_c_dpu/<id>_b512_u96v2.xmodel` | Compiled model, target `DPUCZDX8G_ISA1_B512_0101000056010400`, a single DPU subgraph |
| `phase_c_dpu/meta.json` | Written by `vai_c_xir` and read by VART at load time |
| `phase_c_dpu/md5sum.txt` | Written by `vai_c_xir` |
| `phase_c_dpu/quantizer_scratch/quant_info.json` | Fixed-point position of every quantised tensor, from `vai_q_pytorch` |
| `phase_c_dpu/quantizer_provenance.json` | Toolchain and inputs of the quantisation: torch, Python, `vai_q_pytorch`, Vitis AI image tag, calibration frame count, sha256 of checkpoint and calibration set |
| `phase_c_dpu/build_dpu.log` | Quantiser and compiler log of the campaign build (13 of 14; see below) |

Every model compiles to one DPU subgraph with no CPU fallback partition: the
whole network — 13 convolutions, two max-pooling, two resize and two
concatenation operators — runs on the accelerator.

The checkpoints, ONNX exports and the evaluation and calibration frames these
models were built from are in **aspir-edge-data**. `scripts/link_data.sh` links
them in as `models/<id>/source/`.

## Identifying a model: by content, not by hash

`vai_c_xir` embeds a build identifier in the xmodel, so **rebuilding the same
model gives a different binary**: three consecutive builds of one experiment
produced three different hashes from bit-identical quantisation. A hash change
therefore does not mean the model changed.

Compare `quant_info.json` tensor by tensor, or deserialise both xmodels and
compare their XIR graphs. Both checks were run when the provenance files were
added: across all 14 experiments the fixed points were identical (26 parameter
and 18 activation tensors each) and the graphs matched operator for operator,
including every attribute and the weight data. The binaries kept here are the
ones the published measurements were taken with.

## Rebuilding

```bash
ASPIR_EDGE_DATA=/path/to/aspir-edge-data scripts/link_data.sh
ASPIR_EDGE_DATA=/path/to/aspir-edge-data scripts/build_dpu_xmodels.sh --force <id>
```

This runs `vai_q_pytorch` (post-training quantisation, 200 calibration frames
from the training split, bias correction, no fast fine-tuning) and `vai_c_xir`
against `hardware/dpu_overlay/arch.json`, inside the pinned Vitis AI 3.0
container (`envs/vitis-ai/IMAGE`).

Only the beam-profile experiments can be rebuilt from aspir-edge-data alone:
the CelebA-derived calibration frames are not redistributable.

## Missing file

`irbeam_64_unet_cch_8pct_8_16/phase_c_dpu/build_dpu.log` is absent. That
experiment was the one rebuilt repeatedly to test build determinism, and its
campaign log was overwritten in the process. Its xmodel is the campaign binary.
