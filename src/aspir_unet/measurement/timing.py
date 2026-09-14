"""
Per-inference latency timer, designed to be called from a tight loop.

Uses time.perf_counter_ns() for monotonic, high-resolution timestamps.
Skips the first N inferences (cold cache, first-time DPU job allocation,
ORT first-time graph optimisation) so the reported statistics reflect
steady-state behaviour. The `T0` warm-up is the standard convention in
the Vitis-AI / ONNX Runtime / TensorRT benchmarking literature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter_ns
from typing import Iterator, List


@dataclass
class TimingReport:
    """Steady-state latency statistics over the inference batch."""

    n_total: int
    n_warmup: int
    samples_ms: List[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        return sum(self.samples_ms) / max(1, len(self.samples_ms))

    @property
    def median_ms(self) -> float:
        s = sorted(self.samples_ms)
        m = len(s) // 2
        return (s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])) if s else 0.0

    @property
    def p95_ms(self) -> float:
        s = sorted(self.samples_ms)
        return s[int(0.95 * (len(s) - 1))] if s else 0.0

    @property
    def p99_ms(self) -> float:
        s = sorted(self.samples_ms)
        return s[int(0.99 * (len(s) - 1))] if s else 0.0

    @property
    def fps(self) -> float:
        return 1000.0 / self.mean_ms if self.mean_ms > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "n_total": self.n_total,
            "n_warmup": self.n_warmup,
            "n_measured": len(self.samples_ms),
            "mean_ms": round(self.mean_ms, 3),
            "median_ms": round(self.median_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "fps": round(self.fps, 2),
        }


class TimingMeasurer:
    """Time a sequence of inferences, skipping a fixed warm-up prefix.

    Use as:

        timer = TimingMeasurer(n_total=len(inputs), warmup=5)
        for i in timer.indices():
            t0 = timer.tic()
            run_inference(inputs[i])
            timer.toc(t0)
        report = timer.report()
    """

    def __init__(self, n_total: int, warmup: int = 5) -> None:
        self.n_total = n_total
        self.warmup = min(warmup, max(0, n_total - 1))
        self._samples_ms: list[float] = []
        self._counter = 0

    def indices(self) -> Iterator[int]:
        for i in range(self.n_total):
            self._counter = i
            yield i

    @staticmethod
    def tic() -> int:
        return perf_counter_ns()

    def toc(self, t0_ns: int) -> float:
        elapsed_ms = (perf_counter_ns() - t0_ns) / 1e6
        if self._counter >= self.warmup:
            self._samples_ms.append(elapsed_ms)
        return elapsed_ms

    def report(self) -> TimingReport:
        return TimingReport(
            n_total=self.n_total,
            n_warmup=self.warmup,
            samples_ms=self._samples_ms,
        )
