"""Remember the last train/val/test folder created in Annotate."""
from __future__ import annotations

import json
from pathlib import Path

from app_paths import last_dataset_path


def save_last_dataset(info: dict) -> None:
    path = last_dataset_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def load_last_dataset() -> dict | None:
    path = last_dataset_path()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None
