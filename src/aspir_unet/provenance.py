"""
Run provenance — what a report needs to be evidence rather than a number.

Every run report carries a `run` block describing exactly how it was produced:
which code, which model file, which dataset file, which software versions,
which machine, in which thermal and power-governor state. The rule the paper
depends on is that any figure can be traced back to the run that produced it,
and any run can be re-executed from what its own report says.

On the host the code identity comes from git. On the board there is no git
checkout, so `deploy.sh` writes `scripts/deploy_info.json` at deployment time
and this module reads it back — the board therefore reports the host commit it
was deployed from, which is the honest answer to "which code ran here".

Nothing in this module imports a heavy dependency on purpose: versions of
onnxruntime / vart / xir are read from `sys.modules` when the caller has
already imported them, so collecting provenance never changes what a run loads.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

SCHEMA_VERSION = 2

# Environment variables that change measured behaviour and must be recorded.
_TRACKED_ENV = (
    "XLNX_ENABLE_FINGERPRINT_CHECK",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "GOMP_CPU_AFFINITY",
    "VART_DEBUG",
)

_CPUFREQ = Path("/sys/devices/system/cpu")


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def file_fingerprint(path: Optional[Path]) -> Optional[dict]:
    """sha256 + size + mtime of one artefact, or None when absent."""
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return {"path": str(p), "exists": False}
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    st = p.stat()
    return {
        "path": str(p),
        "name": p.name,
        "sha256": h.hexdigest(),
        "bytes": st.st_size,
        "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------------------
# Code identity
# ---------------------------------------------------------------------------

def _git(args: list, cwd: Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git"] + args, cwd=str(cwd), capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def _deploy_info(script_file: Optional[Path]) -> Optional[dict]:
    """Read the deployment stamp written by deploy.sh, if present."""
    if script_file is None:
        return None
    for candidate in (
        Path(script_file).resolve().parent / "deploy_info.json",
        Path(script_file).resolve().parent.parent / "deploy_info.json",
    ):
        if candidate.is_file():
            try:
                return json.loads(candidate.read_text())
            except (OSError, ValueError):
                return None
    return None


def code_identity(repo_root: Optional[Path],
                  script_file: Optional[Path] = None) -> dict:
    """Which code produced this run: git on the host, deploy stamp on the board."""
    if repo_root is not None:
        sha = _git(["rev-parse", "HEAD"], repo_root)
        if sha:
            status = _git(["status", "--porcelain"], repo_root)
            return {
                "source": "git",
                "git_sha": sha,
                "git_describe": _git(["describe", "--always", "--dirty"], repo_root),
                "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root),
                "dirty": bool(status),
                "dirty_files": [ln[3:] for ln in (status or "").splitlines()][:20],
            }
    info = _deploy_info(script_file)
    if info:
        return {"source": "deploy_info", **info}
    return {"source": "unknown", "git_sha": None, "dirty": None}


# ---------------------------------------------------------------------------
# Software and host state
# ---------------------------------------------------------------------------

def _module_version(name: str) -> Optional[str]:
    """Version of an already-imported module; never imports it itself."""
    mod = sys.modules.get(name)
    if mod is None:
        return None
    for attr in ("__version__", "VERSION", "version"):
        v = getattr(mod, attr, None)
        if isinstance(v, str):
            return v
    return "present"


def software_versions() -> dict:
    versions = {
        "python": platform.python_version(),
        "kernel": platform.release(),
        "platform": platform.platform(),
    }
    for name in ("numpy", "onnxruntime", "skimage", "vart", "xir", "torch"):
        versions[name] = _module_version(name)
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if line.startswith("PRETTY_NAME="):
                versions["os_release"] = line.split("=", 1)[1].strip().strip('"')
                break
    except OSError:
        pass
    return versions


def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def host_state() -> dict:
    """Machine identity plus the knobs that change measured performance."""
    governors = set()
    max_khz = set()
    online = 0
    if _CPUFREQ.is_dir():
        for cpu in sorted(_CPUFREQ.glob("cpu[0-9]*")):
            freq = cpu / "cpufreq"
            if not freq.is_dir():
                continue
            online += 1
            g = _read_text(freq / "scaling_governor")
            if g:
                governors.add(g)
            f = _read_text(freq / "scaling_max_freq")
            if f:
                max_khz.add(int(f))
    state: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "cpus_with_cpufreq": online,
        "cpu_governor": sorted(governors) or None,
        "cpu_max_freq_khz": sorted(max_khz) or None,
        "env": {k: os.environ[k] for k in _TRACKED_ENV if k in os.environ},
    }
    try:
        state["loadavg"] = [round(v, 2) for v in os.getloadavg()]
    except (OSError, AttributeError):
        pass
    return state


_GPU_FREQ_GLOBS = (
    "/sys/devices/platform/bus@0/*.gpu/devfreq/*/cur_freq",
    "/sys/devices/*.gpu/devfreq/*/cur_freq",
    "/sys/class/devfreq/*/cur_freq",
)


def _declared_opp_khz() -> list:
    """CPU operating points as the device tree declares them.

    On a board with no cpufreq driver the sysfs interface is absent entirely,
    so there is no governor to read and no frequency to compare against. The
    device tree still declares the OPP table, which is readable unprivileged
    and is enough to say what the silicon is allowed to run at.
    """
    import struct

    out = []
    for node in sorted(Path("/proc/device-tree").glob("opp-table-cpu/opp*")):
        raw = node / "opp-hz"
        try:
            blob = raw.read_bytes()
        except OSError:
            continue
        if len(blob) == 8:
            hz = struct.unpack(">Q", blob)[0]
        elif len(blob) == 4:
            hz = struct.unpack(">I", blob)[0]
        else:
            continue
        out.append(round(hz / 1000))
    return sorted(set(out), reverse=True)


def _acpu_khz() -> Optional[int]:
    """Actual CPU clock from the common-clock framework, when readable.

    debugfs is 0700 root on this rootfs, so an unprivileged run gets nothing
    and the field is simply absent. It is recorded when available because on a
    fixed-clock board it is the only direct evidence of the rate.
    """
    summary = _read_text(Path("/sys/kernel/debug/clk/clk_summary"))
    if not summary:
        return None
    for line in summary.splitlines():
        fields = line.split()
        if len(fields) > 4 and fields[0] == "acpu":
            try:
                return round(int(fields[4]) / 1000)
            except ValueError:
                return None
    return None


def clock_state() -> dict:
    """Current CPU and accelerator clocks.

    Latency on a board whose clocks float is a property of the governor's mood
    as much as of the workload: on the Orin, the same engine measured 0.73 ms
    with the GPU at one operating point and 0.45 ms at another, at identical
    temperature. Recording the clocks makes that visible in the report instead
    of surfacing as unexplained variance between runs.
    """
    import glob

    state: Dict[str, Any] = {}
    cur, mx = [], []
    for cpu in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*")):
        freq = cpu / "cpufreq"
        for name, sink in (("scaling_cur_freq", cur), ("scaling_max_freq", mx)):
            value = _read_text(freq / name)
            if value:
                sink.append(int(value))
    if cur:
        state["cpu_cur_khz"] = {"min": min(cur), "max": max(cur)}
    if mx:
        state["cpu_max_khz"] = sorted(set(mx))

    # Whether the platform can scale the CPU clock at all. Without this the
    # Ultra96 reports no governor and no cpu_pinned, which reads identically to
    # a Jetson whose clocks were left floating -- the opposite of the truth.
    # The U96 PetaLinux kernel ships no cpufreq driver, so the A53s sit at
    # whatever the PMU set and nothing can move them mid-run.
    state["cpu_dvfs"] = bool(cur or mx)
    if not state["cpu_dvfs"]:
        opps = _declared_opp_khz()
        if opps:
            state["cpu_opp_khz"] = opps
        fixed = _acpu_khz()
        if fixed:
            state["cpu_fixed_khz"] = fixed

    for pattern in _GPU_FREQ_GLOBS:
        hits = sorted(glob.glob(pattern))
        if hits:
            state["gpu_cur_hz"] = {}
            for path in hits[:4]:
                value = _read_text(Path(path))
                if value:
                    state["gpu_cur_hz"][Path(path).parents[1].name] = int(value)
            for path in hits[:1]:
                mx_path = Path(path).with_name("max_freq")
                value = _read_text(mx_path)
                if value:
                    state["gpu_max_hz"] = int(value)
            break

    # jetson_clocks pins CPU, GPU and EMC to the ceiling of the active power
    # mode. Whether it ran is not directly readable, so infer it: every CPU at
    # its maximum and the GPU at its own is the signature.
    if cur and mx:
        state["cpu_pinned"] = min(cur) >= min(mx)
    if state.get("gpu_cur_hz") and state.get("gpu_max_hz"):
        state["gpu_pinned"] = (max(state["gpu_cur_hz"].values())
                               >= state["gpu_max_hz"])
    return state


def accelerator_state(xmodel: Optional[Path] = None) -> dict:
    """DPU-side identity: device node presence and the compiled model target.

    The fingerprint is read from the `meta.json` that `vai_c_xir` writes next
    to the xmodel, not from VART: querying the runtime for a fingerprint is
    exactly the call that segfaults on this stack, which is why the runtime
    check is disabled in the first place.
    """
    state: Dict[str, Any] = {"dev_dpu": Path("/dev/dpu").exists()}
    if xmodel is None:
        return state
    meta_path = Path(xmodel).parent / "meta.json"
    if not meta_path.is_file():
        return state
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, ValueError):
        return state
    target = meta.get("target")
    state["xmodel_target"] = target
    state["xmodel_kernels"] = meta.get("kernel")
    if isinstance(target, str) and "_" in target:
        tail = target.rsplit("_", 1)[-1]
        if all(c in "0123456789abcdefABCDEF" for c in tail) and len(tail) >= 8:
            state["dpu_fingerprint"] = "0x" + tail.lower()
    return state


def device_identity() -> dict:
    """Which board this run happened on.

    The campaign spans two boards and the aggregator has to keep their rows
    apart; inferring the device from a hostname works until someone renames a
    board. `ASPIR_DEVICE` overrides the detection when a board is not
    recognised.
    """
    override = os.environ.get("ASPIR_DEVICE")
    info: Dict[str, Any] = {"device": override or "unknown"}

    model = None
    for path in (Path("/proc/device-tree/model"), Path("/sys/firmware/devicetree/base/model")):
        try:
            model = path.read_bytes().decode("utf-8", "ignore").strip("\x00 \n")
            break
        except OSError:
            continue
    if model:
        info["model"] = model

    # The Ultra96's device tree has no `model` node, and its `/dev/dpu` comes
    # and goes: the CPU route unloads the PL before measuring, precisely so the
    # baseline is not taken with the accelerator powered. Detection must not
    # depend on either, or every CPU run on this board reports a different
    # device from every DPU run and the cross-platform table splits in two.
    # `compatible` is a NUL-separated list and is always present.
    compatible = ""
    for path in (Path("/proc/device-tree/compatible"),
                 Path("/sys/firmware/devicetree/base/compatible")):
        try:
            compatible = path.read_bytes().decode("utf-8", "ignore").replace("\x00", ",").strip(", ")
            break
        except OSError:
            continue
    if compatible:
        info["soc_compatible"] = compatible

    tegra = Path("/etc/nv_tegra_release")
    if tegra.is_file():
        try:
            info["l4t"] = tegra.read_text().splitlines()[0].strip()
        except OSError:
            pass

    if not override:
        low = ((model or "") + " " + compatible).lower()
        if "orin" in low:
            info["device"] = "orin_nx" if "nx" in low else "orin"
        elif tegra.is_file():
            info["device"] = "jetson"
        elif "ultra96" in low or "zynqmp" in low or Path("/dev/dpu").exists():
            info["device"] = "u96v2"
        elif platform.machine() in ("x86_64", "amd64"):
            info["device"] = "host"
    return info


def die_temperatures() -> Optional[dict]:
    """One-shot read of every thermal sensor, in degrees C."""
    try:
        from .measurement.energy import read_temperatures_once
    except Exception:       # pragma: no cover - measurement stack unavailable
        return None
    temps = read_temperatures_once()
    return temps or None


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def collect(
    *,
    route: str,
    mode: str,
    experiment: Any,
    params: dict,
    artifacts: Optional[Dict[str, Optional[Path]]] = None,
    script_file: Optional[str | Path] = None,
    repetition: int = 0,
    tag: Optional[str] = None,
) -> dict:
    """Build the `run` block of a schema-v2 report.

    Args:
        route: canonical route name (`cpu_neon`, `dpu_int8`, `cpu_recon`).
        mode: what this run measured (`latency`, `energy`, `combined`, ...).
        experiment: an `experiments.Experiment`, or None for runs that do not
            belong to one.
        params: the resolved command-line parameters of the run.
        artifacts: named files whose exact content matters, hashed into the
            report (model, dataset, calibration set, ...).
        script_file: `__file__` of the calling script.
        repetition: index of this repetition within a sweep.
        tag: free-form label a sweep can use to group runs.
    """
    repo_root = None
    if experiment is not None and getattr(experiment, "layout", None) is not None:
        repo_root = experiment.layout.repo_root

    xmodel = (artifacts or {}).get("xmodel")

    block = {
        "route": route,
        "mode": mode,
        "repetition": int(repetition),
        "tag": tag,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "params": params,
        "code": code_identity(repo_root, Path(script_file) if script_file else None),
        "software": software_versions(),
        "host": host_state(),
        "device": device_identity(),
        "clocks": clock_state(),
        "accelerator": accelerator_state(xmodel),
        "artifacts": {
            name: file_fingerprint(path)
            for name, path in sorted((artifacts or {}).items())
        },
        "thermal_start_c": die_temperatures(),
    }
    if experiment is not None:
        block["experiment"] = experiment.to_dict()
    return block


def finalise(run_block: dict) -> dict:
    """Close the run block: end timestamp and end-of-run temperatures."""
    run_block["finished_at"] = datetime.now().isoformat(timespec="seconds")
    run_block["thermal_end_c"] = die_temperatures()
    return run_block


def run_suffix(route: str, params: dict) -> Optional[str]:
    """The part of a report name that distinguishes configurations of a route.

    Single source of truth: the runners name their reports with it and the
    sweep driver globs for it when deciding what has already been measured.
    """
    threads = params.get("threads")
    if route in ("cpu_neon", "cpu_recon") and threads:
        precision = params.get("precision")
        suffix = "t{}".format(int(threads))
        # FP32 keeps the historical name so earlier runs stay addressable.
        if precision and precision != "fp32":
            suffix = "{}_{}".format(precision, suffix)
        return suffix
    precision = params.get("precision")
    if route == "gpu_trt" and precision:
        return str(precision).lower()
    return None


def report_filename(route: str, mode: str, experiment_id: str,
                    repetition: int, timestamp: str,
                    suffix: Optional[str] = None) -> str:
    """Deterministic report name, so a sweep can tell what it already ran."""
    parts = [route, mode, experiment_id]
    if suffix:
        parts.append(suffix)
    parts.append("r{}".format(repetition))
    parts.append(timestamp.replace(":", "-"))
    return "_".join(parts) + ".json"


def report_glob(route: str, mode: str, experiment_id: str,
                repetition: int, suffix: Optional[str] = None) -> str:
    """Glob matching every report of one configuration, any timestamp."""
    parts = [route, mode, experiment_id]
    if suffix:
        parts.append(suffix)
    parts.append("r{}".format(repetition))
    return "_".join(parts) + "_*.json"
