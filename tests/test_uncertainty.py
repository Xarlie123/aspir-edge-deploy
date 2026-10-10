"""Checks for the dynamic-energy uncertainty (src/aspir_unet/measurement/uncertainty.py).

Plain-assert script; no board needed:

    python3 tests/test_uncertainty.py
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aspir_unet.measurement.uncertainty import (  # noqa: E402
    active_blocks,
    dynamic_energy_uncertainty,
    energy_uncertainty,
    idle_statistics,
    sensor_resolution,
)


def _segment(duration_s, hz, level, noise, rng, transient=0.0):
    t = np.arange(0.0, duration_s, 1.0 / hz)
    p = level + noise * rng.standard_normal(t.size) + transient * np.exp(-t / 0.5)
    return t, p


def test_idle_uses_one_second_means_after_settling():
    rng = np.random.default_rng(0)
    t, p = _segment(10.0, 100, 8.0, 0.05, rng, transient=1.0)
    idle = idle_statistics(t, p)
    assert idle["n"] == 7, idle["n"]                     # 10 s - 2.5 s settle -> 7 windows
    assert abs(idle["mean_W"] - 8.0) < 0.01, "the transient is excluded"
    # 1 s means of 100 samples: sigma ~ 0.05 / sqrt(100)
    assert 0.002 < idle["std_W"] < 0.012, idle["std_W"]
    assert abs(idle["raw_std_W"] - 0.05) < 0.005
    assert math.isclose(idle["sem_W"], idle["std_W"] / math.sqrt(7))
    print(f"  idle {idle['mean_W']:.3f} W, sigma {idle['std_W']:.4f} W over {idle['n']} windows")


def test_blocks_use_frame_times():
    rng = np.random.default_rng(1)
    t, p = _segment(30.0, 100, 11.0, 0.02, rng)
    frames = np.arange(0.0, 30.0, 1 / 1000.0)            # 1000 fps
    blocks = active_blocks(t, p, frame_t_s=frames)
    assert len(blocks["power_W"]) == 10 and blocks["source"] == "frame_times"
    assert all(abs(x - 1.0) < 0.01 for x in blocks["time_ms"]), blocks["time_ms"]
    fallback = active_blocks(t, p, fps=1000.0)
    assert fallback["source"] == "fps" and set(fallback["time_ms"]) == {1.0}
    print("  10 blocks of the settled load pass, t_b from frame times (or 1/fps)")


def test_block_term_matches_the_analytical_value():
    rng = np.random.default_rng(2)
    P = 7.0 + rng.normal(0, 0.1, 10)
    t = 8.0 + 0.5 * (P - 7.0) + rng.normal(0, 0.05, 10)
    out = dynamic_energy_uncertainty(P * t, P, t, {"mean_W": 6.0, "std_W": 0.0, "sem_W": 0.0, "n": 7})
    pd = P.mean() - 6.0
    c = np.cov(P, t, ddof=1)
    expected = math.sqrt((t.mean() ** 2 * c[0, 0] + pd ** 2 * c[1, 1]
                          + 2 * t.mean() * pd * c[0, 1]) / 10)
    assert math.isclose(out["dynamic_energy_sigma_blocks_mj"], expected, rel_tol=1e-12)
    assert math.isclose(out["dynamic_energy_pt_mj"], pd * t.mean())
    print(f"  delta-method block term {expected:.4f} mJ reproduced")


def test_idle_term_is_the_standard_deviation():
    P, t = np.full(10, 7.0), np.full(10, 8.0)
    out = dynamic_energy_uncertainty(P * t, P, t, {"mean_W": 6.0, "std_W": 0.2, "sem_W": 0.05, "n": 16})
    assert math.isclose(out["dynamic_energy_sigma_idle_mj"], 1.6)
    assert math.isclose(out["dynamic_energy_sigma_idle_sem_mj"], 0.4)
    print("  sigma_idle * t, not the standard error")


def test_sigma_matches_known_variance():
    rng = np.random.default_rng(3)
    t0, s_p, s_idle = 8.0, 0.3, 0.1
    analytical = math.sqrt((s_idle * t0) ** 2 + (t0 * s_p / math.sqrt(10)) ** 2)
    est = []
    for _ in range(4000):
        P = rng.normal(7.0, s_p, 10)
        est.append((P.mean() - rng.normal(6.0, s_idle)) * t0)
    assert abs(np.std(est) / analytical - 1) < 0.05
    print(f"  true spread {np.std(est):.4f} vs analytical {analytical:.4f} mJ")


def test_quantised_sensor_gets_a_resolution_term():
    """The Ultra96 board input reads 5.00 or 6.25 W: sigma is 0, resolution is not."""
    t_i = np.arange(0, 10, 0.055)
    t_a = np.arange(0, 30, 0.06)
    idle_p = np.full(t_i.size, 5.0)
    active_p = np.full(t_a.size, 6.25)
    u = energy_uncertainty(t_i, idle_p, t_a, active_p, fps=900.0)
    assert u["idle_power_std_W"] == 0.0 and u["dynamic_energy_sigma_mj"] == 0.0
    assert u["sensor_resolution_W"] == 1.25
    t_frame = 1000.0 / 900.0
    assert math.isclose(u["dynamic_energy_sigma_resolution_mj"], 1.25 / math.sqrt(6) * t_frame)
    assert math.isclose(u["dynamic_energy_sigma_with_resolution_mj"],
                        u["dynamic_energy_sigma_resolution_mj"])
    assert sensor_resolution(np.full(5, 3.0)) is None
    print(f"  1.25 W step -> +/-{u['dynamic_energy_sigma_resolution_mj']:.3f} mJ on "
          f"{u['dynamic_energy_pt_mj']:.3f} mJ")


def test_missing_baseline_leaves_dynamic_fields_empty():
    P, t = np.array([7.0, 7.1]), np.array([8.0, 8.1])
    out = dynamic_energy_uncertainty(P * t, P, t, None)
    assert not any(k.startswith("dynamic_") for k in out)
    assert set(out) == {"n_blocks", "energy_block_mean_mj", "energy_block_sem_mj"}
    print("  no idle baseline -> block statistics only")


def test_energy_mode_reports_the_uncertainty_block():
    """measure_energy_mode end to end, on a fake hwmon tree with one board-input rail."""
    import json
    import tempfile
    import time

    import aspir_unet.measurement.energy as energy
    from aspir_unet.bench import measure_energy_mode

    with tempfile.TemporaryDirectory() as tmp:
        chip = Path(tmp) / "hwmon0"
        chip.mkdir()
        (chip / "name").write_text("ir38060\n")
        (chip / "power1_label").write_text("pout1\n")
        (chip / "power1_input").write_text("5000000\n")      # 5.0 W in uW
        saved = energy.HWMON_ROOT
        energy.HWMON_ROOT = Path(tmp)
        try:
            def run_one(_):
                (chip / "power1_input").write_text("6250000\n")
                time.sleep(0.002)
            series = {}
            out = measure_energy_mode(run_one, np.zeros((2, 1, 4, 4), np.float32), idle_s=4.0,
                                      active_s=4.0, thermal_warmup_s=0.0, series_out=series)
        finally:
            energy.HWMON_ROOT = saved
    u = out["uncertainty"]
    assert "dynamic_energy_pt_mj" in u, u
    assert u["blocks"]["source"] == "frame_times"
    assert u["sensor_resolution_W"] == 1.25
    assert "frame_t_s" in series["active"]
    json.dumps(out)                                         # the report stays serialisable
    print(f"  E_dyn {u['dynamic_energy_pt_mj']:.3f} mJ +/- {u['dynamic_energy_sigma_with_resolution_mj']:.3f}"
          " (resolution-limited), frame times archived")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in tests:
        print(f"\n{fn.__name__}")
        try:
            fn()
            print("  PASS")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
