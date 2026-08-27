"""
Copy datasets onto a fast local disk before training.

On Windows, YOLO walking every image from D: / USB / OneDrive looks
frozen. We copy once to %LOCALAPPDATA%\\FruitSorterTrainer\\datasets
(always on C:) and train from that copy. Original files are not moved.
"""
import hashlib
import os
import shutil
import sys
from pathlib import Path
from typing import Callable, Optional

from app_paths import datasets_cache_dir
from schemas import DatasetConfig

LogFn = Callable[[str], None]

_CLOUD_MARKERS = (
    "onedrive",
    "google drive",
    "googledrive",
    "dropbox",
    "icloud",
    "sharepoint",
    "box.com",
)

_SKIP_SUFFIXES = {".cache"}
_SKIP_NAMES = {".ds_store", "thumbs.db", "desktop.ini"}
STAMP_NAME = ".staged_from"


def _log(log: Optional[LogFn], message: str) -> None:
    (log or print)(message)


def _normalize(path: Path) -> str:
    return str(path).replace("\\", "/").lower()


def is_cloud_or_network_path(path: Path) -> bool:
    text = _normalize(path)
    if text.startswith("//") or str(path).startswith("\\\\"):
        return True
    return any(marker in text for marker in _CLOUD_MARKERS)


def needs_local_stage(path: Path) -> bool:
    """True when training should copy this folder onto C: first."""
    if sys.platform != "win32":
        return is_cloud_or_network_path(path)
    try:
        resolved = path.expanduser().resolve()
    except OSError:
        resolved = path
    if is_cloud_or_network_path(resolved):
        return True
    drive = resolved.drive.upper()
    if drive and drive != "C:":
        return True
    return False


def staging_message(path: Path) -> Optional[str]:
    if not needs_local_stage(path):
        return None
    dest = datasets_cache_dir()
    return (
        "Dataset is not on the local C: drive "
        "(or is in cloud/network storage). "
        f"Training will copy it once to {dest} and read from there. "
        "Your original files are left in place."
    )


def _looks_like_dataset(root: Path) -> bool:
    if (root / "train" / "images").exists():
        return (root / "valid" / "images").exists() or (root / "val" / "images").exists()
    if (root / "images" / "train").exists():
        return (root / "images" / "val").exists() or (root / "images" / "valid").exists()
    return (root / "data.yaml").exists()


def _staged_dir_for(source: Path) -> Path:
    key = str(source.resolve()).encode("utf-8", errors="replace")
    digest = hashlib.sha256(key).hexdigest()[:12]
    raw = "".join(
        c if c.isalnum() or c in "-_" else "_" for c in source.name
    )
    safe_name = raw or "dataset"
    return datasets_cache_dir() / f"{safe_name}_{digest}"


def _copy_tree(src: Path, dst: Path, log: Optional[LogFn]) -> int:
    copied = 0
    dst.mkdir(parents=True, exist_ok=True)
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__", ".git")]
        rel = os.path.relpath(dirpath, src)
        dest_dir = dst if rel == "." else dst / rel
        dest_dir.mkdir(parents=True, exist_ok=True)
        for name in filenames:
            if name.lower() in _SKIP_NAMES or name == STAMP_NAME:
                continue
            if Path(name).suffix.lower() in _SKIP_SUFFIXES:
                continue
            source_file = Path(dirpath) / name
            shutil.copy2(source_file, dest_dir / name)
            copied += 1
            if copied % 200 == 0:
                _log(log, f"[dataset_stage] Copied {copied} files...")
    return copied


def ensure_local_copy(source: Path, log: Optional[LogFn] = None) -> Path:
    """
    Return a folder YOLO should read from. Copies onto C: when needed.
    Reuses an existing copy of the same source path.
    """
    source = source.expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Dataset folder not found: {source}")

    if not needs_local_stage(source):
        _log(log, f"[dataset_stage] Using local dataset: {source}")
        return source.resolve()

    dest = _staged_dir_for(source)
    stamp = dest / STAMP_NAME
    source_key = str(source.resolve())
    existing = (
        stamp.read_text(encoding="utf-8").strip() if stamp.exists() else ""
    )

    if dest.exists() and existing == source_key and _looks_like_dataset(dest):
        _log(log, f"[dataset_stage] Reusing local C: copy at {dest}")
        return dest

    if dest.exists():
        shutil.rmtree(dest)

    _log(log, "[dataset_stage] Copying dataset to C: for faster training...")
    _log(log, f"[dataset_stage] From: {source.resolve()}")
    _log(log, f"[dataset_stage] To:   {dest}")
    copied = _copy_tree(source, dest, log)
    stamp.write_text(source_key, encoding="utf-8")
    _log(
        log,
        f"[dataset_stage] Copy finished ({copied} files). "
        "Training will use the C: copy.",
    )
    return dest


def stage_dataset_for_training(
    dataset: DatasetConfig, log: Optional[LogFn] = None
) -> DatasetConfig:
    """
    Rewrite dataset paths so resolve_data_yaml / YOLO read from C: on Windows.
    """
    ds = dataset.model_copy(deep=True)
    if ds.use_raw_yaml:
        if not ds.raw_yaml_path:
            return ds
        yaml_path = Path(ds.raw_yaml_path)
        parent = yaml_path.parent
        if needs_local_stage(parent) or needs_local_stage(yaml_path):
            staged_parent = ensure_local_copy(parent, log)
            ds.raw_yaml_path = str(staged_parent / yaml_path.name)
        return ds

    if not ds.dataset_root or not ds.dataset_root.strip():
        return ds
    staged = ensure_local_copy(Path(ds.dataset_root), log)
    ds.dataset_root = str(staged)
    return ds
