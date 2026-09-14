#!/usr/bin/env python3
"""
Report what a target board actually has, before any measurement is attempted.

Two uses. Practically, it turns the first session on a new board into one
command instead of a dozen round trips: runtimes, versions, power rails,
governor and free space in a single output. Methodologically, it is the
environment record the paper's reproducibility paragraph refers to — and it
prints the same fields on both boards, so differences between them are visible
rather than assumed.

    python3 scripts/probe_device.py            # human readable
    python3 scripts/probe_device.py --json     # machine readable
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
for _src in (_HERE.parents[2] / "app" / "src", _HERE.parents[1] / "src"):
    if (_src / "aspir_unet").is_dir():
        sys.path.insert(0, str(_src))
        break


def _version(module: str) -> str:
    try:
        mod = __import__(module)
    except Exception as exc:
        return f"absent ({type(exc).__name__})"
    return getattr(mod, "__version__", "present")


def _cmd(*args: str) -> str:
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return (out.stdout or out.stderr).strip().splitlines()[0] if (out.stdout or out.stderr) else "unavailable"


def collect() -> dict:
    from aspir_unet import experiments, provenance
    from aspir_unet.measurement.energy import (
        EnergySampler, TemperatureSampler, _is_board_input_label,
        read_temperatures_once,
    )

    sampler = EnergySampler()
    rails = sampler.rail_labels
    board_rails = [r for r in rails if _is_board_input_label(r)]

    layout = experiments.discover_layout(__file__)
    known = experiments.list_experiments(layout)

    info = {
        "device": provenance.device_identity(),
        "host": provenance.host_state(),
        "software": provenance.software_versions(),
        "runtimes": {
            name: _version(name)
            for name in ("numpy", "onnxruntime", "skimage", "torch",
                         "tensorrt", "pycuda", "cuda")
        },
        "vart": {
            "xir": _version("xir"),
            "vart": _version("vart"),
            "dev_dpu": Path("/dev/dpu").exists(),
        },
        "power": {
            "rails_found": len(rails),
            "rails": rails,
            "board_input_rails": board_rails,
            "temperature_sensors": TemperatureSampler().sensor_labels,
            "temperatures_c": read_temperatures_once(),
        },
        "jetson": {
            "nvpmodel": _cmd("nvpmodel", "-q"),
            "l4t": Path("/etc/nv_tegra_release").read_text().strip().splitlines()[0]
            if Path("/etc/nv_tegra_release").is_file() else None,
        },
        "experiments": {
            "models_root": str(layout.models_root),
            "count": len(known),
            "ids": known,
        },
        "disk_free_gb": round(shutil.disk_usage(str(layout.models_root
                                                    if layout.models_root.exists()
                                                    else Path.home())).free / 2**30, 1),
        "python": platform.python_version(),
    }

    # The measurement plane must resolve to exactly one rail, or the headline
    # power figure is either missing or double-counted.
    problems = []
    if not rails:
        problems.append("no power rails found: energy measurement will be empty")
    elif not board_rails:
        problems.append(
            "power rails found but none recognised as the board/module input; "
            "set ASPIR_BOARD_INPUT_RAIL to the right label")
    elif len(board_rails) > 1:
        problems.append(f"several rails claim to be the input plane: {board_rails}")
    if info["runtimes"]["tensorrt"].startswith("absent") and \
            info["device"].get("device", "").startswith("orin"):
        problems.append("TensorRT not importable on a Jetson: the GPU route cannot run")
    if info["runtimes"]["cuda"].startswith("absent") and \
            info["runtimes"]["pycuda"].startswith("absent") and \
            info["device"].get("device", "").startswith("orin"):
        problems.append("neither cuda-python nor pycuda is installed: no device buffers")
    if info["runtimes"]["onnxruntime"].startswith("absent"):
        problems.append("onnxruntime absent: the CPU baseline cannot run")
    info["problems"] = problems
    return info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    info = collect()
    if args.json:
        print(json.dumps(info, indent=2))
        return 1 if info["problems"] else 0

    d = info["device"]
    print(f"device        {d.get('device')}  {d.get('model', '')}")
    if info["jetson"]["l4t"]:
        print(f"l4t           {info['jetson']['l4t']}")
        print(f"nvpmodel      {info['jetson']['nvpmodel']}")
    print(f"python        {info['python']}   kernel {info['software'].get('kernel')}")
    print(f"cpu           {info['host'].get('cpu_count')} cores, governor "
          f"{info['host'].get('cpu_governor')}, max "
          f"{info['host'].get('cpu_max_freq_khz')} kHz")
    print(f"disk free     {info['disk_free_gb']} GB")
    print("runtimes      " + ", ".join(f"{k}={v}" for k, v in info["runtimes"].items()))
    print(f"vart          {info['vart']}")
    print(f"power rails   {info['power']['rails_found']} found; input plane: "
          f"{info['power']['board_input_rails'] or 'NONE'}")
    if info["power"]["rails"]:
        print(f"              {', '.join(info['power']['rails'][:8])}"
              f"{' …' if len(info['power']['rails']) > 8 else ''}")
    print(f"temperature   {len(info['power']['temperature_sensors'])} sensors")
    print(f"experiments   {info['experiments']['count']} under "
          f"{info['experiments']['models_root']}")
    for p in info["problems"]:
        print(f"[PROBLEM]     {p}")
    if not info["problems"]:
        print("\nno blocking problems found")
    return 1 if info["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
