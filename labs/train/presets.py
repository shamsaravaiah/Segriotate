"""
Save/load named training configurations (e.g. one preset per fruit type)
so a repeat run doesn't mean re-entering every hyperparameter by hand.
"""
import json
from pathlib import Path
from typing import List, Dict

from app_paths import writable_root

PRESETS_DIR = writable_root() / "presets"
PRESETS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_filename(name: str) -> str:
    keep = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)
    return f"{keep}.json"


def save_preset(name: str, config: dict):
    path = PRESETS_DIR / _safe_filename(name)
    with open(path, "w") as f:
        json.dump({"name": name, "config": config}, f, indent=2)


def list_presets() -> List[Dict]:
    presets = []
    for p in sorted(PRESETS_DIR.glob("*.json")):
        try:
            with open(p, "r") as f:
                data = json.load(f)
            presets.append({"name": data.get("name", p.stem)})
        except Exception:
            continue
    return presets


def load_preset(name: str) -> Dict:
    path = PRESETS_DIR / _safe_filename(name)
    if not path.exists():
        raise FileNotFoundError(f"Preset not found: {name}")
    with open(path, "r") as f:
        return json.load(f)["config"]


def delete_preset(name: str):
    path = PRESETS_DIR / _safe_filename(name)
    if path.exists():
        path.unlink()
