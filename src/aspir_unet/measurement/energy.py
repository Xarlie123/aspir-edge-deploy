"""
Background power and temperature samplers for the Avnet Ultra96-v2.

Two independent samplers, each on its own thread:

* `EnergySampler` polls every `powerN_input` channel exposed under
  `/sys/class/hwmon/hwmonN/` at 200 Hz. The chip itself computes V x I
  with its calibrated PMBus telemetry, so we just integrate P(t)
  trapezoidally to get energy in joules. Channels are tagged with the
  paired `powerN_label` (`pin1`, `pout3`, ...) so the report can break
  energy down per rail and per chip.

* `TemperatureSampler` polls every `tempN_input` channel at 5 Hz
  (temperatures change slowly). Reports min/avg/max per channel and
  the raw timeline so we can plot the thermal trace alongside the
  inference timeline.

Hardware seen on a u96v2-sbc 2022.2 image:

    hwmon0  irps5401   PMIC #1 — 5 outputs feeding PS rails
    hwmon1  irps5401   PMIC #2 — 5 outputs feeding PL rails
    hwmon2  ir38060    front-end 12 V step-down (pin1 = total board input)
    hwmon3  iio_hwmon  Zynq SYSMON bridge (die temps + internal voltages)

Why direct `powerN_input` rather than V x I in software:

    The earlier implementation paired `inN_input` and `currN_input` by
    index. That is fine for ir38060 (one channel) but wrong for irps5401
    where inN holds {vin1..vin5, vout1..vout5} and currN holds
    {iin1..iin4, iout1..iout5}: vin5 x iout1 is nonsense. The chip's own
    PMBus block already does V x I per rail with manufacturer
    calibration, exposed as `powerN_input` in microwatts — we just read
    that.

Why two threads:

    Energy integration wants tight, jitter-free 5 ms cadence so the
    trapezoidal sum tracks short DPU pulses. Temperature changes on a
    100 ms+ time-scale, so reading it in the same loop wastes I2C
    bandwidth and lengthens the energy loop. Splitting them keeps each
    cadence pure and lets the temperature thread block on its own
    rate.

Both samplers degrade gracefully: on a host without hwmon (e.g. unit
tests on a laptop), `start()` does nothing and `report()` returns an
empty struct.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Thread
from typing import Dict, List, Tuple


HWMON_ROOT = Path("/sys/class/hwmon")
ENERGY_SAMPLING_HZ = 200            # 5 ms cadence
TEMPERATURE_SAMPLING_HZ = 5         # 200 ms cadence


# ---------------------------------------------------------------------------
# Sysfs discovery helpers
# ---------------------------------------------------------------------------

def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (FileNotFoundError, ValueError, PermissionError, OSError):
        return None


def _read_str(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


# Two hwmon devices on the Ultra96-V2 report the same `name`, "irps5401", so
# labelling a rail with that name alone collapses ten distinct rails into five
# and one PMIC silently overwrites the other. Which one survived depended on
# directory order, and it was the wrong one: the PS controller, whose rails say
# nothing about what the DPU costs.
#
# Roles are from the board schematic (Avnet Ultra96-V2, sheets 15 and 16), not
# inferred from the measured response:
#
#   0x13/0x43  U110x  programmable logic
#              A VCCAUX 1.8 V · B VCCO_HP 1.2 V · C VCCO_PSDDR 1.1 V
#              D VCCINT (PL core -- where the DPU draws) · LDO 3.3 V DisplayPort
#   0x14/0x44  U120x  processing system
#              A VCCPSAUX 1.8 V · B VCC_PSINTLP · C VCCO 3.3 V
#              D VCC_PSINTFP (PS full-power domain, the A53s) · LDO 1.2 V PSPLL
#
# Rails are therefore named irps5401_pl.* and irps5401_ps.*. The bare
# `irps5401.*` key in reports written before 2026-08-30 is the PS controller.
_CHIP_ALIASES = {
    ("irps5401", "5-0043"): "irps5401_pl",
    ("irps5401", "5-0044"): "irps5401_ps",
}


def _device_id(entry: Path) -> str:
    """The bus address behind an hwmon node, e.g. "5-0043"."""
    try:
        return (entry / "device").resolve().name
    except OSError:
        return ""


def _chip_names() -> dict:
    """Map each hwmon directory to a label prefix that is unique on this board.

    Known controllers get the role name from the schematic. Anything else keeps
    its hwmon `name`, qualified by bus address only when that name is shared --
    so a board with one controller per name is unaffected, and a board with two
    never silently merges them.
    """
    names: dict = {}
    seen: dict = {}
    if not HWMON_ROOT.is_dir():
        return names
    for entry in sorted(HWMON_ROOT.iterdir()):
        name = _read_str(entry / "name")
        if not name:
            continue
        seen[name] = seen.get(name, 0) + 1
    for entry in sorted(HWMON_ROOT.iterdir()):
        name = _read_str(entry / "name")
        if not name:
            continue
        dev = _device_id(entry)
        alias = _CHIP_ALIASES.get((name, dev))
        if alias:
            names[entry] = alias
        elif seen.get(name, 0) > 1 and dev:
            names[entry] = f"{name}@{dev}"
        else:
            names[entry] = name
    return names


def _enumerate_channels(suffix: str, max_idx: int = 33) -> List[Tuple[str, str, Path]]:
    """Walk `/sys/class/hwmon/hwmon*` and yield every `<suffix>N_input`
    file alongside the chip name and a stable rail label.

    Returns list of (chip_name, label, input_path), where label is
    `<chip>.<suffix>N_label_value>` if a sibling label exists, else
    `<chip>.<suffix>N`.
    """
    out: List[Tuple[str, str, Path]] = []
    if not HWMON_ROOT.is_dir():
        return out
    chips = _chip_names()
    for entry in sorted(HWMON_ROOT.iterdir()):
        name = _read_str(entry / "name")
        if not name:
            continue
        chip = chips.get(entry, name)
        for i in range(1, max_idx):
            ip = entry / f"{suffix}{i}_input"
            if not ip.is_file():
                continue
            lab = _read_str(entry / f"{suffix}{i}_label") or f"{suffix}{i}"
            out.append((chip, f"{chip}.{lab}", ip))
    return out


def _enumerate_computed_power(max_idx: int = 33) -> List[Tuple[str, str, Path, Path]]:
    """Rails that expose voltage and current but no pre-computed power.

    The Ultra96's regulators publish `powerN_input` directly. The Jetson's
    INA3221 driver, depending on the kernel, may publish only `inN_input`
    (millivolts) and `currN_input` (milliamps), in which case power has to be
    formed as V x I. Returns (chip, label, voltage_path, current_path).
    """
    out: List[Tuple[str, str, Path, Path]] = []
    if not HWMON_ROOT.is_dir():
        return out
    chips = _chip_names()
    for entry in sorted(HWMON_ROOT.iterdir()):
        name = _read_str(entry / "name")
        if not name:
            continue
        chip = chips.get(entry, name)
        for i in range(1, max_idx):
            volt = entry / f"in{i}_input"
            curr = entry / f"curr{i}_input"
            power = entry / f"power{i}_input"
            if power.is_file():
                continue                     # already covered by the direct path
            if not (volt.is_file() and curr.is_file()):
                continue
            # Only labelled channels are real rails. The INA3221 has three,
            # but its driver also exposes shunt voltages as in4..in6 and a
            # summation current as curr4; pairing those by index multiplies a
            # shunt voltage in microvolts by a summation current and calls the
            # product watts, which is how a 5 W module came to report a 40 W
            # rail. An unlabelled channel is not a rail.
            lab = (_read_str(entry / f"in{i}_label")
                   or _read_str(entry / f"curr{i}_label"))
            if not lab:
                continue
            out.append((chip, f"{chip}.{lab}", volt, curr))
    return out


THERMAL_ROOT = Path("/sys/class/thermal")


def _enumerate_thermal_zones() -> List[Tuple[str, str, Path]]:
    """Thermal zones, which is where the Jetson publishes its temperatures.

    The Ultra96's sensors appear as hwmon `tempN_input`; the Orin's do not —
    they live under /sys/class/thermal/thermal_zoneN/. Reading both means the
    thermal state of a run is recorded on either board, which the energy
    protocol depends on: a comparison taken from different thermal states is
    not a comparison.
    """
    out: List[Tuple[str, str, Path]] = []
    if not THERMAL_ROOT.is_dir():
        return out
    for zone in sorted(THERMAL_ROOT.glob("thermal_zone*")):
        temp = zone / "temp"
        if not temp.is_file():
            continue
        label = _read_str(zone / "type") or zone.name
        out.append(("thermal", f"zone.{label}", temp))
    return out


def read_temperatures_once() -> Dict[str, float]:
    """One-shot read of every thermal sensor, in degrees C.

    Used for the start/end temperature stamps in a run report, where a
    background sampler would be overkill. Returns {} when no sensor is
    exposed (e.g. a host machine without the board's hwmon tree).
    """
    out: Dict[str, float] = {}
    for _, label, input_path in (_enumerate_channels("temp")
                                 + _enumerate_thermal_zones()):
        mdeg = _read_int(input_path)
        if mdeg is not None:
            out[label] = round(mdeg / 1000.0, 2)
    return out


# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------

@dataclass
class EnergyReport:
    duration_s: float
    n_samples: int
    rail_energy_j: Dict[str, float] = field(default_factory=dict)
    rail_avg_w: Dict[str, float] = field(default_factory=dict)
    board_input_w_avg: float = 0.0
    board_input_energy_j: float = 0.0
    board_input_rails: List[str] = field(default_factory=list)

    @property
    def total_energy_j(self) -> float:
        """Sum of every sampled rail. Diagnostic only; not physical because
        sum(rails) double-counts the regulator hierarchy."""
        return sum(self.rail_energy_j.values())

    @property
    def total_avg_power_w(self) -> float:
        return sum(self.rail_avg_w.values())

    def to_dict(self) -> dict:
        return {
            "duration_s": round(self.duration_s, 4),
            "n_samples": self.n_samples,
            "board_input_w_avg": round(self.board_input_w_avg, 3),
            "board_input_energy_j": round(self.board_input_energy_j, 4),
            # Which rails were summed into the headline figure. The two boards
            # do not measure at the same plane (board input on the Ultra96,
            # module input on the Orin), so every report says what its number
            # covers instead of leaving it to be assumed.
            "board_input_rails": sorted(self.board_input_rails),
            "sum_of_rails_w": round(self.total_avg_power_w, 3),
            "sum_of_rails_energy_j": round(self.total_energy_j, 4),
            "per_rail": {
                lbl: {
                    "energy_j": round(self.rail_energy_j.get(lbl, 0.0), 5),
                    "avg_power_w": round(self.rail_avg_w.get(lbl, 0.0), 4),
                }
                for lbl in sorted(self.rail_energy_j)
            },
        }


# Total-input rails of the boards we measure on. The label is what the driver
# publishes, so the same code picks the right plane on either device.
#   Ultra96-V2 : ir38060 front-end 12 V step-down, single channel.
#   Jetson Orin: INA3221 channel carrying module input; the label differs
#                between carriers (devkit vs custom), hence a small set.
_JETSON_INPUT_RAILS = {
    "vin_sys_5v0", "vdd_in", "pom_5v_in", "sys5v", "vin_sys", "vdd_sys_in",
}

# Escape hatch: if a board labels its input rail in yet another way, name it
# explicitly rather than silently summing the wrong thing.
_BOARD_INPUT_ENV = "ASPIR_BOARD_INPUT_RAIL"


def _is_board_input_label(label: str) -> bool:
    """Is this rail the board's (or module's) total input?

    The headline power figure must come from exactly one plane. Summing
    downstream regulator rails on top of their upstream feed would
    double-count the regulator hierarchy.
    """
    override = os.environ.get(_BOARD_INPUT_ENV)
    if override:
        return label.lower() == override.lower() or \
            label.lower().endswith("." + override.lower())

    chip, rail = label.split(".", 1)
    rail_l = rail.lower()

    # Ultra96-V2: ir38060 is the front-end 12 V step-down with a single power
    # channel — labelled pin1 or pout1 depending on the PMBus config. Either
    # way it is the best single-number proxy for total board power. irps5401
    # rails sit downstream of it.
    #
    # The two IRPS5401 controllers are labelled irps5401_pl and irps5401_ps
    # (see _CHIP_ALIASES). An exact match on "irps5401" stopped excluding them
    # when they were renamed, and their pin1/pout1 channels then matched the
    # generic test below: board input was reported as the sum of five rails,
    # 5.73 W at idle against 4.97 W on ir38060 alone. Match the family.
    if chip == "irps5401" or chip.startswith("irps5401_") \
            or chip.startswith("irps5401@"):
        return False
    if rail_l in ("pin1", "pout1"):
        return True

    # Jetson INA3221: pick the module-input channel, never the CPU/GPU/SoC
    # channels, which are downstream of it.
    return rail_l in _JETSON_INPUT_RAILS


class EnergySampler:
    """Background per-rail power sampler. Use as a context manager."""

    def __init__(self, sampling_hz: int = ENERGY_SAMPLING_HZ) -> None:
        self._period_s = 1.0 / sampling_hz
        self._channels = _enumerate_channels("power")
        # Rails that publish only voltage and current (the Jetson's INA3221 on
        # some kernels) are sampled as V x I by the same loop, so the rest of
        # the stack does not care which board it is running on.
        self._computed = _enumerate_computed_power()
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._samples: List[Tuple[float, Dict[str, float]]] = []
        self._t_start_ns: int = 0
        self._t_stop_ns: int = 0

    @property
    def available(self) -> bool:
        return bool(self._channels) or bool(self._computed)

    @property
    def rail_labels(self) -> List[str]:
        return ([lbl for _, lbl, _ in self._channels]
                + [lbl for _, lbl, _, _ in self._computed])

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("EnergySampler already started")
        self._stop_event.clear()
        self._samples.clear()
        self._t_start_ns = time.perf_counter_ns()
        if not self.available:
            return
        self._thread = Thread(target=self._loop, daemon=True, name="energy-sampler")
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            t_rel = (time.perf_counter_ns() - self._t_start_ns) / 1e9
            row: Dict[str, float] = {}
            for _, lbl, ip in self._channels:
                uw = _read_int(ip)
                if uw is not None:
                    row[lbl] = uw / 1e6          # uW -> W
            for _, lbl, vp, cp in self._computed:
                mv, ma = _read_int(vp), _read_int(cp)
                if mv is not None and ma is not None:
                    row[lbl] = (mv * ma) / 1e6   # mV * mA -> W
            self._samples.append((t_rel, row))
            self._stop_event.wait(self._period_s)

    def stop(self) -> None:
        if self._thread is None:
            self._t_stop_ns = time.perf_counter_ns()
            return
        self._stop_event.set()
        self._thread.join()
        self._thread = None
        self._t_stop_ns = time.perf_counter_ns()

    def series(self) -> Dict[str, list]:
        """The raw samples, not just their averages.

        A power trace across an idle-to-load-to-idle transition is the figure
        that shows the measurement is real; an average cannot produce it. The
        sampler already holds the samples, so the only thing missing was
        handing them over.
        """
        out: Dict[str, list] = {"t_s": [t for t, _ in self._samples]}
        for label in self.rail_labels:
            out[label] = [row.get(label, float("nan")) for _, row in self._samples]
        return out

    def report(self) -> EnergyReport:
        duration_s = max(0.0, (self._t_stop_ns - self._t_start_ns) / 1e9)
        if not self._samples:
            return EnergyReport(duration_s=duration_s, n_samples=0)

        rail_energy: Dict[str, float] = {lbl: 0.0 for lbl in self.rail_labels}
        for (t0, p0), (t1, p1) in zip(self._samples, self._samples[1:]):
            dt = t1 - t0
            for lbl in rail_energy:
                rail_energy[lbl] += 0.5 * dt * (p0.get(lbl, 0.0) + p1.get(lbl, 0.0))

        rail_avg = {
            lbl: (rail_energy[lbl] / duration_s) if duration_s > 0 else 0.0
            for lbl in rail_energy
        }
        board_lbls = [l for l in rail_avg if _is_board_input_label(l)]
        board_w = sum(rail_avg[l] for l in board_lbls)
        board_j = sum(rail_energy[l] for l in board_lbls)
        return EnergyReport(
            duration_s=duration_s,
            n_samples=len(self._samples),
            rail_energy_j=rail_energy,
            rail_avg_w=rail_avg,
            board_input_w_avg=board_w,
            board_input_energy_j=board_j,
            board_input_rails=board_lbls,
        )

    def __enter__(self) -> "EnergySampler":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# Temperature
# ---------------------------------------------------------------------------

@dataclass
class TemperatureReport:
    duration_s: float
    n_samples: int
    stats_c: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "duration_s": round(self.duration_s, 4),
            "n_samples": self.n_samples,
            "per_sensor": {
                lbl: {
                    "min_c": round(v[0], 2),
                    "avg_c": round(v[1], 2),
                    "max_c": round(v[2], 2),
                }
                for lbl, v in sorted(self.stats_c.items())
            },
        }


class TemperatureSampler:
    """Background thermal sampler. Same API as EnergySampler but on a
    slower cadence — temperatures change on 100 ms+ time-scales."""

    def __init__(self, sampling_hz: int = TEMPERATURE_SAMPLING_HZ) -> None:
        self._period_s = 1.0 / sampling_hz
        self._channels = _enumerate_channels("temp") + _enumerate_thermal_zones()
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._samples: List[Tuple[float, Dict[str, float]]] = []
        self._t_start_ns: int = 0
        self._t_stop_ns: int = 0

    @property
    def available(self) -> bool:
        return len(self._channels) > 0

    @property
    def sensor_labels(self) -> List[str]:
        return [lbl for _, lbl, _ in self._channels]

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("TemperatureSampler already started")
        self._stop_event.clear()
        self._samples.clear()
        self._t_start_ns = time.perf_counter_ns()
        if not self.available:
            return
        self._thread = Thread(target=self._loop, daemon=True, name="temp-sampler")
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            t_rel = (time.perf_counter_ns() - self._t_start_ns) / 1e9
            row: Dict[str, float] = {}
            for _, lbl, ip in self._channels:
                mdeg = _read_int(ip)
                if mdeg is not None:
                    row[lbl] = mdeg / 1000.0           # millidegC -> degC
            self._samples.append((t_rel, row))
            self._stop_event.wait(self._period_s)

    def stop(self) -> None:
        if self._thread is None:
            self._t_stop_ns = time.perf_counter_ns()
            return
        self._stop_event.set()
        self._thread.join()
        self._thread = None
        self._t_stop_ns = time.perf_counter_ns()

    def report(self) -> TemperatureReport:
        duration_s = max(0.0, (self._t_stop_ns - self._t_start_ns) / 1e9)
        if not self._samples:
            return TemperatureReport(duration_s=duration_s, n_samples=0)
        per_label: Dict[str, List[float]] = {lbl: [] for _, lbl, _ in self._channels}
        for _, row in self._samples:
            for lbl, v in row.items():
                per_label.setdefault(lbl, []).append(v)
        stats: Dict[str, Tuple[float, float, float]] = {}
        for lbl, vals in per_label.items():
            if not vals:
                continue
            stats[lbl] = (min(vals), sum(vals) / len(vals), max(vals))
        return TemperatureReport(
            duration_s=duration_s,
            n_samples=len(self._samples),
            stats_c=stats,
        )

    def __enter__(self) -> "TemperatureSampler":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()
