"""
Experiment registry — one identifier resolves every artefact of a run.

Before this module the canonical experiment was a constant inside `bench.py`,
which was fine while there was exactly one model and one dataset. The JoS
measurement campaign sweeps five sampling ratios x two network sizes x two
routes, so the experiment has to become a parameter.

An *experiment* is a directory holding everything a run needs:

    <models_root>/<experiment_id>/
    ├── source/
    │   ├── test_images.npz          evaluation set (required)
    │   ├── calibration_images.npz   PTQ calibration set, disjoint from the
    │   │                            evaluation set (optional, host-side only)
    │   ├── masks.npz                sensing patterns, for reconstruction
    │   │                            timing (optional)
    │   ├── test_config.json         aspir experiment config (optional)
    │   └── *.onnx                   FP32 export for the CPU route
    ├── phase_c_dpu/
    │   └── *.xmodel                 compiled INT8 model for the DPU route
    └── results/                     per-run JSON reports (host layout)

`<models_root>` is `<repo>/app/models` on the host and `<deploy>/models` on the
board; the same experiment id therefore names the same experiment on both
sides, which is what makes a sweep reproducible from either place.

Artefact resolution never guesses silently: when a directory holds more than
one candidate and no preference rule applies, resolution raises with the list
of candidates rather than picking one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

# Fallback deployment root, used only when the script location does not reveal
# the layout (e.g. a script executed from outside its directory on the board).
BOARD_ROOT = Path("/home/petalinux/aspir")

HOST_MODELS_SUB = Path("app") / "models"

SOURCE_DIRNAME = "source"
# Historical name first-come: new experiments may use either, and the tree of
# already-compiled models on the board uses the old one.
DPU_DIRNAMES = ("dpu_int8", "phase_c_dpu")
DPU_DIRNAME = DPU_DIRNAMES[-1]
# Serialised TensorRT engines, built on the Jetson that runs them.
TRT_DIRNAME = "jetson_trt"
RESULTS_DIRNAME = "results"

TEST_NPZ = "test_images.npz"
CALIBRATION_NPZ = "calibration_images.npz"
MASKS_NPZ = "masks.npz"
CONFIG_JSON = "test_config.json"

# Experiment used when no --experiment is given, preserving the behaviour the
# scripts had before the registry existed.
LEGACY_EXPERIMENT_ID = "beam_64_gaussian_unet_8pct_cch"


@dataclass(frozen=True)
class Layout:
    """Where models and results live on this machine."""

    models_root: Path
    repo_root: Optional[Path]      # None when running on the board
    deploy_root: Optional[Path]    # set on the board, None on the host

    @property
    def is_board(self) -> bool:
        return self.repo_root is None


@dataclass(frozen=True)
class Experiment:
    """Every artefact of one experiment, resolved to absolute paths."""

    experiment_id: str
    root: Path
    source_dir: Path
    results_dir: Path
    npz: Path
    onnx: Optional[Path]
    xmodel: Optional[Path]
    calibration_npz: Optional[Path]
    masks_npz: Optional[Path]
    config_json: Optional[Path]
    trt_dir: Path
    layout: Layout

    # -- accessors that fail loudly instead of returning None ---------------

    @property
    def onnx_int8(self) -> Optional[Path]:
        """The INT8 export, quantised on the board that will execute it."""
        if self.onnx is None:
            return None
        path = self.onnx.with_name(self.onnx.stem + "_int8.onnx")
        return path if path.is_file() else None

    def require_onnx_int8(self) -> Path:
        path = self.onnx_int8
        if path is None:
            raise FileNotFoundError(
                f"experiment '{self.experiment_id}': no INT8 export. Build it "
                f"with the same runtime that will execute it:\n"
                f"    python3 scripts/quantize_cpu_int8.py --experiment "
                f"{self.experiment_id}")
        return path

    def require_onnx(self) -> Path:
        if self.onnx is None:
            raise FileNotFoundError(
                f"experiment '{self.experiment_id}': no .onnx under "
                f"{self.source_dir}. The CPU route needs an FP32 export."
            )
        return self.onnx

    def require_xmodel(self) -> Path:
        if self.xmodel is None:
            raise FileNotFoundError(
                f"experiment '{self.experiment_id}': no .xmodel under "
                f"{self.root / DPU_DIRNAME}. Quantise and compile first "
                f"(see scripts/quantize_dpu_int8.py)."
            )
        return self.xmodel

    def trt_engine(self, precision: str) -> Path:
        """Where the engine for one precision lives, whether or not it exists.

        Engines are per device and per TensorRT build, so they are produced on
        the Jetson by `build_trt_engine.py` and never shipped from the host.
        """
        return self.trt_dir / f"{self.experiment_id}_{precision.lower()}.engine"

    def require_trt_engine(self, precision: str) -> Path:
        path = self.trt_engine(precision)
        if not path.is_file():
            raise FileNotFoundError(
                f"experiment '{self.experiment_id}': no {precision} engine at "
                f"{path}. Build it on the Jetson first:\n"
                f"    python3 scripts/build_trt_engine.py "
                f"--experiment {self.experiment_id} --precision {precision}"
            )
        return path

    def require_calibration(self) -> Path:
        if self.calibration_npz is None:
            raise FileNotFoundError(
                f"experiment '{self.experiment_id}': no calibration set at "
                f"{self.source_dir / CALIBRATION_NPZ}. INT8 must be calibrated "
                f"on training-split frames, never on the evaluation set."
            )
        return self.calibration_npz

    def require_masks(self) -> Path:
        if self.masks_npz is None:
            raise FileNotFoundError(
                f"experiment '{self.experiment_id}': no {MASKS_NPZ} under "
                f"{self.source_dir}. Reconstruction timing needs the sensing "
                f"patterns exported alongside the dataset."
            )
        return self.masks_npz

    def config(self) -> dict:
        """The aspir-side experiment config, or {} when absent."""
        if self.config_json is None:
            return {}
        try:
            return json.loads(self.config_json.read_text())
        except (OSError, ValueError):
            return {}

    def to_dict(self) -> dict:
        """Identity block for a run report."""
        return {
            "experiment_id": self.experiment_id,
            "root": rel_to(self.root, self.layout),
            "npz": rel_to(self.npz, self.layout),
            "onnx": rel_to(self.onnx, self.layout) if self.onnx else None,
            "onnx_int8": (rel_to(self.onnx_int8, self.layout)
                          if self.onnx_int8 else None),
            "xmodel": rel_to(self.xmodel, self.layout) if self.xmodel else None,
            "calibration_npz": (rel_to(self.calibration_npz, self.layout)
                                if self.calibration_npz else None),
            "masks_npz": (rel_to(self.masks_npz, self.layout)
                          if self.masks_npz else None),
            "trt_engines": sorted(p.name for p in self.trt_dir.glob("*.engine"))
                           if self.trt_dir.is_dir() else [],
            "config": self.config(),
        }


def rel_to(path: Path, layout: Layout) -> str:
    """Path relative to the repo root on the host, absolute on the board."""
    root = layout.repo_root
    if root is None:
        return str(path)
    try:
        return str(Path(path).resolve().relative_to(root))
    except ValueError:
        return str(path)


def discover_layout(script_file: str | Path) -> Layout:
    """Work out whether we are running from the host repo or a deployment.

    Args:
        script_file: pass `__file__` from the calling script.

    Host layout:  <repo>/app/scripts/<script>.py  with  <repo>/app/models/
    Board layout: <deploy>/scripts/<script>.py    with  <deploy>/models/
    """
    here = Path(script_file).resolve()

    if len(here.parents) >= 3:
        repo_root = here.parents[2]
        models = repo_root / HOST_MODELS_SUB
        if models.is_dir():
            return Layout(models_root=models, repo_root=repo_root,
                          deploy_root=None)

    if len(here.parents) >= 2:
        deploy_root = here.parents[1]
        models = deploy_root / "models"
        if models.is_dir():
            return Layout(models_root=models, repo_root=None,
                          deploy_root=deploy_root)

    return Layout(models_root=BOARD_ROOT / "models", repo_root=None,
                  deploy_root=BOARD_ROOT)


def list_experiments(layout: Layout) -> List[str]:
    """Every experiment id that has at least an evaluation set."""
    if not layout.models_root.is_dir():
        return []
    found = []
    for child in sorted(layout.models_root.iterdir()):
        if (child / SOURCE_DIRNAME / TEST_NPZ).is_file():
            found.append(child.name)
    return found


def _pick_one(
    directory: Path,
    pattern: str,
    prefer: Sequence[str] = (),
    exclude: Sequence[str] = (),
    what: str = "file",
) -> Optional[Path]:
    """Resolve exactly one file, or raise with the ambiguity spelled out.

    `prefer` holds lowercase substrings tried in order; the first substring
    matched by exactly one candidate wins. Anything still ambiguous is an
    error, because silently choosing the wrong model is the kind of mistake
    that only surfaces after the numbers are in a paper.
    """
    if not directory.is_dir():
        return None
    candidates = sorted(p for p in directory.glob(pattern)
                        if p.is_file()
                        and not any(tok in p.name for tok in exclude))
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    for token in prefer:
        hits = [p for p in candidates if token.lower() in p.name.lower()]
        if len(hits) == 1:
            return hits[0]
    names = ", ".join(p.name for p in candidates)
    raise ValueError(
        f"ambiguous {what} in {directory}: {names}. "
        f"Pass the path explicitly on the command line."
    )


def resolve(
    experiment_id: Optional[str] = None,
    script_file: str | Path = __file__,
    layout: Optional[Layout] = None,
) -> Experiment:
    """Resolve an experiment id to all of its artefacts.

    Args:
        experiment_id: directory name under the models root; when None, the
            legacy canonical experiment is used.
        script_file: `__file__` of the calling script, used to find the layout.
        layout: pre-computed layout, when the caller already has one.
    """
    layout = layout or discover_layout(script_file)
    exp_id = experiment_id or LEGACY_EXPERIMENT_ID
    root = layout.models_root / exp_id

    if not root.is_dir():
        known = list_experiments(layout)
        raise FileNotFoundError(
            f"experiment '{exp_id}' not found under {layout.models_root}. "
            f"Known experiments: {', '.join(known) if known else '(none)'}"
        )

    source = root / SOURCE_DIRNAME
    npz = source / TEST_NPZ
    if not npz.is_file():
        raise FileNotFoundError(f"experiment '{exp_id}': missing {npz}")

    # The INT8 export is a derived sibling of the FP32 one, not a second
    # candidate for it: the CPU route selects between them by precision.
    onnx = _pick_one(source, "*.onnx", prefer=(exp_id, "unet"),
                     exclude=("_int8",), what="onnx export")
    xmodel = None
    for dpu_dir in DPU_DIRNAMES:
        xmodel = _pick_one(root / dpu_dir, "*.xmodel",
                           prefer=("b512", exp_id), what="xmodel")
        if xmodel is not None:
            break

    calibration = source / CALIBRATION_NPZ
    masks = source / MASKS_NPZ
    config = source / CONFIG_JSON

    if layout.is_board:
        deploy_root = layout.deploy_root or BOARD_ROOT
        results_dir = deploy_root / RESULTS_DIRNAME / exp_id
    else:
        results_dir = root / RESULTS_DIRNAME

    return Experiment(
        experiment_id=exp_id,
        root=root,
        source_dir=source,
        results_dir=results_dir,
        npz=npz,
        onnx=onnx,
        xmodel=xmodel,
        calibration_npz=calibration if calibration.is_file() else None,
        masks_npz=masks if masks.is_file() else None,
        config_json=config if config.is_file() else None,
        trt_dir=root / TRT_DIRNAME,
        layout=layout,
    )
