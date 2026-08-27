"""
Lists .pt weights for the Train dropdown: Annotate's models, then trained runs.
"""
from pathlib import Path
from typing import Dict, List

from app_paths import default_runs_dir
import config  # noqa: E402


def _annotate_label(stem: str) -> str:
    labels = getattr(config, "ANNOTATE_PT_LABELS", {})
    return labels.get(stem, stem)


def _add(seen: set[str], models: list, path: Path, name: str) -> None:
    if not path.is_file() or path.suffix.lower() != ".pt":
        return
    resolved = str(path.resolve())
    if resolved in seen:
        return
    seen.add(resolved)
    models.append({"name": name, "path": resolved})


def list_model_groups() -> List[Dict]:
    """[{id, title, models: [{name, path}]}] for the Train dropdown."""
    seen: set[str] = set()
    groups: List[Dict] = []

    annotate: list = []
    pt_dir = config.MODEL_PT_DIR
    if pt_dir.is_dir():
        pts = sorted(pt_dir.glob("*.pt"), key=lambda p: p.name.lower())
        pts.sort(key=lambda p: (p.stem != "segmentation", p.name.lower()))
        for pt_file in pts:
            _add(seen, annotate, pt_file, _annotate_label(pt_file.stem))
    if annotate:
        groups.append({"id": "annotate", "title": "Annotate", "models": annotate})

    trained: list = []
    runs_dir = default_runs_dir()
    if runs_dir.is_dir():
        for pt_file in sorted(runs_dir.rglob("*.pt")):
            if pt_file.parent.name != "weights":
                continue
            run_name = pt_file.parent.parent.name
            _add(seen, trained, pt_file, f"{run_name} / {pt_file.name}")
    if trained:
        groups.append({"id": "trained", "title": "Trained runs", "models": trained})

    return groups


def list_available_models() -> List[Dict[str, str]]:
    """Flat [{name, path}] for callers that do not use groups."""
    return [m for group in list_model_groups() for m in group["models"]]


def detect_devices() -> List[Dict[str, str]]:
    """Detect available training devices (CUDA, Apple MPS, CPU)."""
    devices: List[Dict[str, str]] = []
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                devices.append({"value": str(i), "label": f"GPU {i}: {name}"})
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            devices.append({"value": "mps", "label": "Apple GPU (MPS)"})
    except Exception:
        pass
    devices.append({"value": "cpu", "label": "CPU"})
    return devices
