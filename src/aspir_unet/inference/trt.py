"""
Jetson GPU path — TensorRT inference backend.

Mirrors the DPU backend's contract on purpose: `run_one(x_fp32)` takes and
returns float32 NCHW, and **everything a deployment actually pays is inside
the timed call** — host-to-device transfer, execution, device-to-host
transfer. The DPU runner counts its float-to-int8 conversion the same way.
Timing only the kernel would flatter the GPU and make the two platforms
incomparable, which is the single easiest way to invalidate the cross-device
comparison this repository exists to make.

The per-call breakdown (`last_breakdown`) keeps the transfer and execution
components separate, so a result can be reported at either boundary without
re-measuring.

Engine I/O stays float32 whatever the engine's internal precision: for an INT8
engine TensorRT inserts the quantise/dequantise reformat layers itself, and
they are then part of the measured cost — exactly as the DPU's host-side
conversion is.

Nothing here imports TensorRT or CUDA at module level, so the module parses on
a machine without either (host-side editing, unit tests, static analysis).
Engines are built on the device that runs them: a serialised engine is tied to
the GPU architecture and the TensorRT build that produced it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# CUDA memory, through whichever binding the board has
# ---------------------------------------------------------------------------

class _Cuda:
    """Minimal device-memory adapter over cuda-python or pycuda.

    JetPack images ship one or the other (sometimes both) and their APIs are
    not compatible, so the few operations we need are wrapped once here.
    """

    def __init__(self) -> None:
        self.backend = None
        try:
            from cuda import cudart          # type: ignore
            self._rt = cudart
            self.backend = "cuda-python"
        except Exception:
            try:
                import pycuda.driver as drv   # type: ignore
                import pycuda.autoinit        # noqa: F401  (creates the context)
                self._drv = drv
                self.backend = "pycuda"
            except Exception as exc:          # pragma: no cover
                raise RuntimeError(
                    "no CUDA python binding available: install cuda-python "
                    "(preferred on JetPack) or pycuda"
                ) from exc

    # -- helpers ------------------------------------------------------------

    def _check(self, ret) -> Any:
        """cuda-python returns (err, value); raise on a non-zero error code."""
        if isinstance(ret, tuple):
            err, *rest = ret
            if int(err) != 0:
                raise RuntimeError(f"CUDA call failed with status {int(err)}")
            return rest[0] if len(rest) == 1 else tuple(rest)
        return ret

    def malloc(self, nbytes: int) -> int:
        if self.backend == "cuda-python":
            return int(self._check(self._rt.cudaMalloc(nbytes)))
        return int(self._drv.mem_alloc(nbytes))

    def free(self, ptr: int) -> None:
        if self.backend == "cuda-python":
            self._rt.cudaFree(ptr)
        # pycuda frees on garbage collection of the allocation object

    def create_stream(self) -> int:
        if self.backend == "cuda-python":
            return int(self._check(self._rt.cudaStreamCreate()))
        return self._drv.Stream()

    def h2d(self, dst: int, host: np.ndarray, stream) -> None:
        if self.backend == "cuda-python":
            self._check(self._rt.cudaMemcpyAsync(
                dst, host.ctypes.data, host.nbytes,
                self._rt.cudaMemcpyKind.cudaMemcpyHostToDevice, stream))
        else:
            self._drv.memcpy_htod_async(dst, host, stream)

    def d2h(self, host: np.ndarray, src: int, stream) -> None:
        if self.backend == "cuda-python":
            self._check(self._rt.cudaMemcpyAsync(
                host.ctypes.data, src, host.nbytes,
                self._rt.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream))
        else:
            self._drv.memcpy_dtoh_async(host, src, stream)

    def sync(self, stream) -> None:
        if self.backend == "cuda-python":
            self._check(self._rt.cudaStreamSynchronize(stream))
        else:
            stream.synchronize()

    def stream_handle(self, stream) -> int:
        """The integer handle TensorRT wants."""
        if self.backend == "cuda-python":
            return int(stream)
        return stream.handle


_DTYPE_TOKENS = ("INT8", "FP8", "FP16", "BF16", "FP32", "INT32", "INT64", "BOOL")


def _datatype_of(layer: dict) -> str:
    """Arithmetic type of a layer, read from its tensor format strings.

    Asking TensorRT for an INT8 engine does not make every layer INT8: the
    builder picks per layer, and for a network this narrow (1 to 32 channels)
    FP16 kernels are often faster than INT8 ones. What fraction actually ran in
    INT8 is therefore a measured property of the engine, not a setting.

    Worse, with FP16 left enabled that choice is **not reproducible**. Seven
    builds here of one identical ONNX, identical flags, identical calibration
    cache and a flat 63-64 C gave INT8 fractions of 0, 0, 0, 25, 27, 31 and
    48 %: INT8 and FP16 kernels are close enough on this topology that the
    builder's own timing noise decides. Two consequences, both practical:

      * use `int8_nofp16` for anything that must be comparable across builds
        or across grid points — with FP16 off the builder has no discretion
        left and every build has produced the same composition;
      * never quote a latency from a default `int8` engine without the layer
        map that run recorded, because the next build may contain no INT8 at
        all.
    """
    for side in ("Outputs", "Inputs"):
        for tensor in layer.get(side, []) or []:
            fmt = str(tensor.get("Format/Datatype", "")).upper()
            for token in _DTYPE_TOKENS:
                if token in fmt:
                    return token
    return "unknown"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class TrtRunner:
    """Single-image inference over a serialised TensorRT engine."""

    def __init__(self, engine_path: Path | str, logger_severity: str = "WARNING") -> None:
        import tensorrt as trt                # local import: see module docstring

        self._trt = trt
        self.engine_path = Path(engine_path)
        self.trt_version = trt.__version__

        severity = getattr(trt.Logger, logger_severity, trt.Logger.WARNING)
        self._logger = trt.Logger(severity)
        runtime = trt.Runtime(self._logger)
        self.engine = runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"could not deserialise engine {self.engine_path}")
        self.context = self.engine.create_execution_context()

        self.cuda = _Cuda()
        self._stream = self.cuda.create_stream()

        # TensorRT 8.5 introduced the named-tensor API and 10.x dropped the
        # binding-index one; support both rather than pinning a JetPack.
        self._use_v3 = hasattr(self.engine, "num_io_tensors") and \
            hasattr(self.context, "execute_async_v3")

        self._inputs: List[dict] = []
        self._outputs: List[dict] = []
        self._bindings: List[int] = []
        self._setup_io()

        self.in_dims = tuple(self._inputs[0]["shape"])
        self.out_dims = tuple(self._outputs[0]["shape"])
        self.last_breakdown: Dict[str, float] = {}

    # -- setup --------------------------------------------------------------

    def _setup_io(self) -> None:
        trt = self._trt
        if self._use_v3:
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                shape = tuple(self.context.get_tensor_shape(name))
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                host = np.empty(shape, dtype=dtype)
                dev = self.cuda.malloc(host.nbytes)
                entry = {"name": name, "shape": shape, "dtype": dtype,
                         "host": host, "dev": dev}
                (self._inputs if is_input else self._outputs).append(entry)
                self.context.set_tensor_address(name, dev)
        else:
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                shape = tuple(self.context.get_binding_shape(i))
                dtype = trt.nptype(self.engine.get_binding_dtype(i))
                host = np.empty(shape, dtype=dtype)
                dev = self.cuda.malloc(host.nbytes)
                entry = {"name": name, "shape": shape, "dtype": dtype,
                         "host": host, "dev": dev, "index": i}
                (self._inputs if self.engine.binding_is_input(i)
                 else self._outputs).append(entry)
                self._bindings.append(dev)

        if not self._inputs or not self._outputs:
            raise RuntimeError(
                f"engine {self.engine_path.name} exposes "
                f"{len(self._inputs)} inputs and {len(self._outputs)} outputs; "
                f"expected at least one of each"
            )

    # -- inference ----------------------------------------------------------

    def run_one(self, x_fp32: np.ndarray) -> np.ndarray:
        """Run one (1, 1, H, W) float32 frame, transfers included."""
        from time import perf_counter_ns

        inp, out = self._inputs[0], self._outputs[0]
        x = x_fp32
        if x.ndim == 3:
            x = x[None]
        x = np.ascontiguousarray(x.astype(inp["dtype"], copy=False))
        if x.shape != inp["shape"]:
            x = x.reshape(inp["shape"])

        t0 = perf_counter_ns()
        np.copyto(inp["host"], x)
        self.cuda.h2d(inp["dev"], inp["host"], self._stream)
        t1 = perf_counter_ns()

        handle = self.cuda.stream_handle(self._stream)
        if self._use_v3:
            ok = self.context.execute_async_v3(handle)
        else:
            ok = self.context.execute_async_v2(self._bindings, handle)
        if not ok:
            raise RuntimeError("TensorRT execution returned failure")
        self.cuda.sync(self._stream)
        t2 = perf_counter_ns()

        self.cuda.d2h(out["host"], out["dev"], self._stream)
        self.cuda.sync(self._stream)
        t3 = perf_counter_ns()

        self.last_breakdown = {
            "h2d_ms": (t1 - t0) / 1e6,
            "execute_ms": (t2 - t1) / 1e6,
            "d2h_ms": (t3 - t2) / 1e6,
        }
        return out["host"].astype(np.float32, copy=True)

    def run_batch(self, xs: np.ndarray) -> Iterator[np.ndarray]:
        for i in range(xs.shape[0]):
            yield self.run_one(xs[i:i + 1])

    # -- introspection ------------------------------------------------------

    def engine_info(self) -> dict:
        """Layer-level precision map of the engine.

        TensorRT builds mixed-precision engines by default: asking for INT8
        does not mean every layer runs in INT8. The paper has to state what
        fraction actually did, so the map is recorded with the run.
        """
        trt = self._trt
        info: Dict[str, Any] = {
            "tensorrt_version": self.trt_version,
            "engine": self.engine_path.name,
            "cuda_binding": self.cuda.backend,
            "io": [{"name": e["name"], "shape": list(e["shape"]),
                    "dtype": np.dtype(e["dtype"]).name}
                   for e in self._inputs + self._outputs],
        }
        try:
            import json
            inspector = self.engine.create_engine_inspector()
            if hasattr(inspector, "execution_context"):
                inspector.execution_context = self.context
            raw = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
            layers = json.loads(raw)
            entries = layers.get("Layers", layers) if isinstance(layers, dict) else layers
            entries = entries if isinstance(entries, list) else []
            counts: Dict[str, int] = {}
            kinds: Dict[str, int] = {}
            for layer in entries:
                if not isinstance(layer, dict):
                    continue
                # TensorRT 10 publishes no per-layer "Precision" field: the
                # arithmetic type is encoded in each tensor's Format/Datatype
                # string ("Row major linear FP32", "Channel major INT8 ...").
                # The layer's output type is what it computed in.
                prec = layer.get("Precision") or _datatype_of(layer)
                counts[str(prec)] = counts.get(str(prec), 0) + 1
                kind = str(layer.get("LayerType", "unknown"))
                kinds[kind] = kinds.get(kind, 0) + 1
            info["n_layers"] = len(entries)
            info["layer_precision_counts"] = counts
            info["layer_type_counts"] = kinds
            info["layer_information"] = entries
        except Exception as exc:              # pragma: no cover - version drift
            info["layer_precision_error"] = str(exc)
        return info

    def close(self) -> None:
        for entry in self._inputs + self._outputs:
            try:
                self.cuda.free(entry["dev"])
            except Exception:
                pass
        self._inputs, self._outputs, self._bindings = [], [], []
        self.context = None
        self.engine = None


# ---------------------------------------------------------------------------
# Engine building (runs on the device that will execute the engine)
# ---------------------------------------------------------------------------

def build_engine(
    onnx_path: Path | str,
    engine_path: Path | str,
    precision: str = "int8",
    calib_npz: Optional[Path | str] = None,
    calib_key: str = "reconstructions",
    calib_limit: int = 200,
    workspace_mib: int = 1024,
    logger_severity: str = "INFO",
) -> dict:
    """Build a serialised TensorRT engine from an ONNX file.

    Args:
        precision: `int8`, `fp16` or `fp32`. INT8 requires `calib_npz`; the
            calibration frames must be the same ones the DPU quantiser used,
            drawn from the training split — different calibration data would
            make part of any quality difference an artefact of the data
            rather than of the quantisation scheme.
    Returns a build report: flags, timings, calibration provenance, and the
    engine's layer-precision breakdown.
    """
    import json
    import time

    import tensorrt as trt

    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    # Three INT8 variants, because "INT8" is a request to the builder rather
    # than a description of the engine. Each names a different, defensible
    # deployment choice, and the layer map records what each actually got:
    #
    #   int8         INT8 calibrated, FP16 also allowed — what a practitioner
    #                gets by default, and the fastest engine TensorRT can make
    #   int8_nofp16  INT8 the only reduced precision; anything TensorRT will
    #                not run in INT8 falls back to FP32
    #   int8_strict  INT8 with precision constraints obeyed — the closest
    #                comparator to an accelerator that has no float path at all
    variant = precision.lower()
    aliases = {
        "int8": ("int8", True, False),
        "int8_nofp16": ("int8", False, False),
        "int8_strict": ("int8", True, True),
        "fp16": ("fp16", True, False),
        "fp32": ("fp32", False, False),
    }
    if variant not in aliases:
        raise ValueError(f"unknown precision '{precision}'; "
                         f"known: {sorted(aliases)}")
    precision, allow_fp16, obey_precision = aliases[variant]

    logger = trt.Logger(getattr(trt.Logger, logger_severity, trt.Logger.INFO))
    builder = trt.Builder(logger)

    flags = 0
    explicit = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)
    if explicit is not None:
        flags = 1 << int(explicit)
    network = builder.create_network(flags)

    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("ONNX parse failed:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    # Without DETAILED verbosity the engine inspector reports layer *names*
    # only, and the question the paper has to answer — what fraction of the
    # network actually ran in INT8 — becomes unanswerable after the fact.
    if hasattr(config, "profiling_verbosity"):
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                     workspace_mib << 20)
    elif hasattr(config, "max_workspace_size"):
        config.max_workspace_size = workspace_mib << 20

    report: Dict[str, Any] = {
        "tensorrt_version": trt.__version__,
        "onnx": str(onnx_path),
        "engine": str(engine_path),
        "variant": variant,
        "precision_requested": precision,
        "fp16_allowed": bool(allow_fp16),
        "obey_precision_constraints": bool(obey_precision),
        "workspace_mib": workspace_mib,
    }

    calibrator = None
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        if calib_npz is None:
            raise ValueError("INT8 needs --calib-npz (training-split frames)")
        config.set_flag(trt.BuilderFlag.INT8)
        if allow_fp16 and builder.platform_has_fast_fp16:
            # Leaving FP16 available is the default deployment behaviour; the
            # layer map records what TensorRT actually chose.
            config.set_flag(trt.BuilderFlag.FP16)

        calib_npz = Path(calib_npz)
        cache_path = engine_path.with_suffix(".calib_cache")
        cuda = _Cuda()

        class _Calibrator(trt.IInt8EntropyCalibrator2):
            """Feeds calibration frames one at a time (static batch-1 engine)."""

            def __init__(self) -> None:
                super().__init__()
                npz = np.load(str(calib_npz))
                key = calib_key if calib_key in npz.files else npz.files[0]
                frames = npz[key]
                if frames.ndim == 3:
                    frames = frames[:, None]
                self.frames = np.ascontiguousarray(
                    frames[:calib_limit].astype(np.float32))
                self.index = 0
                self.nbytes = int(self.frames[0:1].nbytes)
                self.device = cuda.malloc(self.nbytes)
                self.stream = cuda.create_stream()

            def get_batch_size(self) -> int:
                return 1

            def get_batch(self, names, p_str=None):
                if self.index >= len(self.frames):
                    return None
                frame = np.ascontiguousarray(self.frames[self.index:self.index + 1])
                cuda.h2d(self.device, frame, self.stream)
                cuda.sync(self.stream)
                self.index += 1
                return [int(self.device)]

            def read_calibration_cache(self):
                if cache_path.is_file():
                    return cache_path.read_bytes()
                return None

            def write_calibration_cache(self, cache):
                cache_path.write_bytes(bytes(cache))

        calibrator = _Calibrator()
        config.int8_calibrator = calibrator
        report["calibration"] = {
            "npz": str(calib_npz),
            "key": calib_key,
            "frames": int(len(calibrator.frames)),
            "cache": str(cache_path),
            "cache_reused": cache_path.is_file(),
        }

    if obey_precision:
        # Setting the flag alone does nothing useful: it makes TensorRT obey
        # constraints that must first be attached to individual layers. Without
        # them the builder still chooses freely — and on this network it
        # chooses FP16 almost everywhere, which is how a "forced INT8" engine
        # ends up containing no INT8 at all.
        for name in ("OBEY_PRECISION_CONSTRAINTS", "STRICT_TYPES"):
            flag = getattr(trt.BuilderFlag, name, None)
            if flag is not None:
                config.set_flag(flag)
                report["precision_constraint_flag"] = name
                break

        forceable = {getattr(trt.LayerType, n) for n in
                     ("CONVOLUTION", "MATRIX_MULTIPLY", "FULLY_CONNECTED")
                     if hasattr(trt.LayerType, n)}
        forced = 0
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if layer.type in forceable:
                layer.precision = trt.DataType.INT8
                forced += 1
        report["layers_forced_int8"] = forced

    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_s = time.perf_counter() - t0
    if serialized is None:
        raise RuntimeError("TensorRT returned no engine; see the builder log")
    engine_path.write_bytes(bytes(serialized))
    report["build_seconds"] = round(build_s, 1)
    report["engine_bytes"] = engine_path.stat().st_size

    # Report what the engine actually is, not what was asked for.
    runner = TrtRunner(engine_path)
    try:
        info = runner.engine_info()
        report["engine_info"] = {k: v for k, v in info.items()
                                 if k != "layer_information"}
        layers = info.get("layer_information")
        if layers is not None:
            layers_path = engine_path.with_suffix(".layers.json")
            layers_path.write_text(json.dumps(layers, indent=1))
            report["layer_information_file"] = str(layers_path)
    finally:
        runner.close()
    return report
