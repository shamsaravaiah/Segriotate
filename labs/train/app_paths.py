"""Writable paths for training jobs, under workspace/train/."""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT = Path(__file__).resolve().parents[2]
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

import config  # noqa: E402


def resource_root() -> Path:
    return config.PROJECT_ROOT


def workspace_root() -> Path:
    config.ensure_workspace()
    return config.WORKSPACE


def train_root() -> Path:
    config.ensure_workspace()
    return config.TRAIN_ROOT


def writable_root() -> Path:
    """Scratch, presets, and other Train-generated files."""
    return train_root()


def user_data_dir() -> Path:
    return train_root()


def default_runs_dir() -> Path:
    path = config.TRAIN_RUNS_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def datasets_cache_dir() -> Path:
    path = config.TRAIN_CACHE_DIR / "datasets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def last_dataset_path() -> Path:
    return workspace_root() / "last_dataset.json"
