"""Build TensorRT .engine files on the deployed machine from models/dot-pt/*.pt."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import config

MIN_BYTES = 1024


def _status(cb, msg: str) -> None:
    if cb:
        cb(msg)
    print(msg, flush=True)


def _source_dir() -> Path:
    return Path(getattr(config, "MODEL_SOURCE_DIR", config.PROJECT_ROOT / "models" / "source"))


def _pt_dir() -> Path:
    return Path(getattr(config, "MODEL_PT_DIR", config.PROJECT_ROOT / "models" / "dot-pt"))


def _engine_dir() -> Path:
    return Path(getattr(config, "MODEL_ENGINE_DIR", config.PROJECT_ROOT / "models" / "dot-engine"))


def tensorrt_available() -> bool:
    try:
        import tensorrt  # noqa: F401
        return True
    except ImportError:
        return False


def _copy_pt_into_runtime(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == src.stat().st_size:
        if dest.stat().st_mtime >= src.stat().st_mtime:
            return
    shutil.copy2(src, dest)


def _engine_is_current(pt_path: Path, engine_path: Path) -> bool:
    try:
        if not engine_path.is_file() or engine_path.stat().st_size <= MIN_BYTES:
            return False
        return engine_path.stat().st_mtime >= pt_path.stat().st_mtime
    except OSError:
        return False


def _export_onnx(pt_path: Path, imgsz: int, batch: int) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(pt_path))
    export_path = model.export(
        format="onnx",
        imgsz=imgsz,
        batch=batch,
        dynamic=False,
        simplify=True,
        opset=17,
        verbose=False,
    )
    return Path(export_path)


def _set_workspace(trt, builder_config, workspace_gb: int) -> None:
    nbytes = int(workspace_gb) * (1024 ** 3)
    if hasattr(builder_config, "set_memory_pool_limit"):
        try:
            builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, nbytes)
            return
        except Exception:
            pass
    if hasattr(builder_config, "max_workspace_size"):
        builder_config.max_workspace_size = nbytes


def _serialize_engine(builder, network, builder_config):
    if hasattr(builder, "build_serialized_network"):
        return builder.build_serialized_network(network, builder_config)
    engine = builder.build_engine(network, builder_config)
    if engine is None:
        return None
    return engine.serialize()


def _build_engine(onnx_path: Path, engine_path: Path, on_status) -> None:
    import tensorrt as trt

    _status(on_status, f"Building TensorRT engine for {engine_path.name}…")
    start = time.time()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as fh:
        if not parser.parse(fh.read()):
            errors = []
            for i in range(parser.num_errors):
                errors.append(str(parser.get_error(i)))
            raise RuntimeError("TensorRT ONNX parse failed: " + "; ".join(errors))

    builder_config = builder.create_builder_config()
    _set_workspace(trt, builder_config, int(getattr(config, "ENGINE_WORKSPACE_GB", 4)))

    if getattr(config, "ENGINE_USE_FP16", True) and getattr(builder, "platform_has_fast_fp16", False):
        builder_config.set_flag(trt.BuilderFlag.FP16)

    if getattr(config, "ENGINE_USE_SPARSE", True) and hasattr(trt.BuilderFlag, "SPARSE_WEIGHTS"):
        try:
            builder_config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
        except Exception:
            pass

    if hasattr(trt.BuilderFlag, "PREFER_PRECISION_CONSTRAINTS"):
        try:
            builder_config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        except Exception:
            pass

    if hasattr(builder_config, "builder_optimization_level"):
        try:
            builder_config.builder_optimization_level = 5
        except Exception:
            pass

    batch = int(getattr(config, "ENGINE_BATCH", 1))
    imgsz = int(getattr(config, "ENGINE_IMGSZ", 640))
    if network.num_inputs < 1:
        raise RuntimeError("ONNX network has no inputs")
    input_tensor = network.get_input(0)
    profile = builder.create_optimization_profile()
    shape = (batch, 3, imgsz, imgsz)
    profile.set_shape(input_tensor.name, shape, shape, shape)
    builder_config.add_optimization_profile(profile)

    serialized = _serialize_engine(builder, network, builder_config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = engine_path.with_suffix(".engine.part")
    tmp.write_bytes(bytes(serialized))
    tmp.replace(engine_path)
    elapsed = time.time() - start
    size_mb = engine_path.stat().st_size / (1024 * 1024)
    _status(
        on_status,
        f"Saved {engine_path.name} ({size_mb:.0f} MB, {elapsed / 60:.1f} min)",
    )


def _convert_one(pt_path: Path, engine_path: Path, on_status) -> None:
    imgsz = int(getattr(config, "ENGINE_IMGSZ", 640))
    batch = int(getattr(config, "ENGINE_BATCH", 1))
    onnx_file = None
    try:
        _status(on_status, f"Exporting {pt_path.name} to ONNX…")
        onnx_file = _export_onnx(pt_path, imgsz, batch)
        _build_engine(onnx_file, engine_path, on_status)
    finally:
        if onnx_file is not None and onnx_file.exists():
            try:
                onnx_file.unlink()
            except OSError:
                pass


def ensure_engines(on_status=None) -> None:
    """Build models/dot-engine/<stem>.engine for every .pt in models/dot-pt/."""
    source = _source_dir()
    pt_dir = _pt_dir()
    engine_dir = _engine_dir()
    source.mkdir(parents=True, exist_ok=True)
    pt_dir.mkdir(parents=True, exist_ok=True)
    engine_dir.mkdir(parents=True, exist_ok=True)

    for src in source.glob("*.pt"):
        if src.is_file() and src.stat().st_size > MIN_BYTES:
            _copy_pt_into_runtime(src, pt_dir / src.name)

    pts = sorted(
        p for p in pt_dir.glob("*.pt")
        if p.is_file() and p.stat().st_size > MIN_BYTES
    )
    if not pts:
        return

    if not tensorrt_available():
        _status(on_status, "TensorRT not installed — using .pt files (no .engine build).")
        return

    jobs = [p for p in pts if not _engine_is_current(p, engine_dir / f"{p.stem}.engine")]
    if not jobs:
        _status(on_status, "TensorRT engines are up to date.")
        return

    for i, pt_path in enumerate(jobs, 1):
        engine_path = engine_dir / f"{pt_path.stem}.engine"
        _status(on_status, f"Building engine {i}/{len(jobs)}: {pt_path.name}")
        try:
            _convert_one(pt_path, engine_path, on_status)
        except Exception as e:
            _status(on_status, f"Could not build {pt_path.stem}.engine: {e}")
