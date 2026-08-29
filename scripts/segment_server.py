"""
Local inference + UI server for Segri-Labs.

1. /          -- home (Sort, Annotate, or Train)
2. /sort      -- unsupervised similarity bins (beta)
3. /annotate  -- the label editor (HTML)
4. /train     -- YOLO training UI
4. /detect    -- YOLO segmentation.pt on the current image
5. /segment   -- click-to-segment fallback (.pt or .engine: FastSAM, MobileSAM, …)
6. /label     -- read/write YOLO .txt files in labels/
7. /project/split-dataset -- copy train/val/test (+ CSV) from the open folders
8. /project/train/* -- start/stop/poll training jobs
9. /media     -- serve images from the chosen images folder

Usage (browser):
    python scripts/segment_server.py
    then open http://127.0.0.1:8765

Usage (desktop):
    python desktop_app.py
"""

import base64
import io
import json
import os
import sys
import threading
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[0]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))
import config  # noqa: E402

_TRAIN_DIR = config.PROJECT_ROOT / "labs" / "train"
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))
_SORT_DIR = config.PROJECT_ROOT / "labs" / "sort"
if str(_SORT_DIR) not in sys.path:
    sys.path.insert(0, str(_SORT_DIR))
from flask import Flask, abort, jsonify, request, send_from_directory  # noqa: E402
from flask_cors import CORS  # noqa: E402
from PIL import Image  # noqa: E402

from ensure_models import ensure_models  # noqa: E402
from build_engines import ensure_engines  # noqa: E402
from split_yolo_dataset import split_dataset  # noqa: E402
from last_dataset import load_last_dataset, save_last_dataset  # noqa: E402
from dataset_utils import inspect_dataset_root, validate_dataset  # noqa: E402
from schemas import DatasetConfig, TrainConfig  # noqa: E402
from job_manager import job_manager  # noqa: E402
from model_registry import detect_devices, list_model_groups  # noqa: E402
from app_paths import default_runs_dir  # noqa: E402
import presets as train_presets  # noqa: E402
import job as sort_job  # noqa: E402

PORT = getattr(config, "SERVER_PORT", 8765)
PT_DIR = config.PROJECT_ROOT / "models" / "dot-pt"
ENGINE_DIR = config.PROJECT_ROOT / "models" / "dot-engine"
CLICK_FORMATS = ("pt", "engine")
DEFAULT_CLICK_FORMAT = "pt"
CLICK_SKIP_STEMS = {"segmentation"}  # Auto-Detect YOLO, not a click model
CLICK_LABELS = getattr(config, "CLICK_MODEL_LABELS", {
    "FastSAM-s": "FastSAM",
    "FastSAM-x": "FastSAM-x",
    "mobile_sam": "MobileSAM",
})
TOOLS_DIR = config.PROJECT_ROOT / "tools"
VENDOR_DIR = TOOLS_DIR / "vendor"
ICON_PNG = (
    config.PROJECT_ROOT
    / "Segriotate.app"
    / "Contents"
    / "Resources"
    / "segriotate_icon_1024.png"
)
ICON_ICO = config.PROJECT_ROOT / "Segriotate.ico"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LABEL_SIDECAR_STEMS = {"class_profile", "classes"}
CLASS_PROFILES_PATH = getattr(
    config, "CLASS_PROFILES_PATH", config.PROJECT_ROOT / "class_profiles.json"
)

app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "OPTIONS"],
        "allow_headers": ["Content-Type"],
    }},
)


@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp

PT_DIR.mkdir(parents=True, exist_ok=True)
ENGINE_DIR.mkdir(parents=True, exist_ok=True)

detect_model = None
_boot = {
    "ready": False,
    "fatal": None,
    "message": "Launching app. Please wait",
}
_boot_started = False
_click_models: dict = {}
_click_lock = threading.Lock()

_images_dir: Path | None = None
_labels_dir: Path | None = None


def get_images_dir() -> Path | None:
    """Only the folder the user picked — never a project default."""
    return _images_dir


def clear_images_dir() -> None:
    global _images_dir
    _images_dir = None


def _set_boot_message(msg: str) -> None:
    _boot["message"] = msg
    print(msg, flush=True)


def _boot_models() -> None:
    global detect_model
    try:
        _set_boot_message("Launching app. Please wait")
        ensure_models(_set_boot_message)
        ensure_engines(_set_boot_message)
        if not config.MODEL_PATH.exists():
            _set_boot_message(
                "Starting editor. Copy segmentation.pt into models/dot-pt for Auto-Detect."
            )
            _boot["ready"] = True
            return
        _set_boot_message("Loading Auto-Detect model…")
        from ultralytics import YOLO  # local import so Flask can bind first

        detect_model = YOLO(str(config.MODEL_PATH))
        if detect_model.task != "segment":
            raise RuntimeError(
                f"{config.MODEL_PATH} is a '{detect_model.task}' model, not a segmentation model."
            )
        _set_boot_message("Ready")
        _boot["ready"] = True
        print("Click-to-segment models load on first use from models/dot-pt or models/dot-engine.\n")
    except Exception as e:
        _boot["fatal"] = str(e)
        _set_boot_message(f"Failed to start: {e}")


def start_boot_thread() -> None:
    global _boot_started
    if _boot_started:
        return
    _boot_started = True
    threading.Thread(target=_boot_models, daemon=True, name="segriotate-boot").start()


start_boot_thread()


def paired_labels_dir(images_folder: Path) -> Path:
    """Pair an image folder with workspace/labels/<name>_labels (or an older labels/ folder)."""
    name = images_folder.name.strip() or "images"
    if name in {".", ".."}:
        name = "images"
    candidates = [
        config.LABELS_ROOT / f"{name}_labels",
        config.LABELS_ROOT / f"{name}-labels",
        config.LEGACY_LABELS_ROOT / f"{name}_labels",
        config.LEGACY_LABELS_ROOT / f"{name}-labels",
    ]
    for path in candidates:
        resolved = path.resolve()
        if resolved.is_dir():
            return resolved
    return (config.LABELS_ROOT / f"{name}_labels").resolve()


def set_images_dir(path: str | Path) -> Path:
    global _images_dir
    folder = Path(path).expanduser().resolve()
    if not folder.is_dir():
        raise ValueError(f"not a directory: {folder}")
    _images_dir = folder
    set_labels_dir(paired_labels_dir(folder))
    sort_job.reveal_folder()
    # A newly picked dump is the whole folder, not the last Sort bin.
    sort_job.set_session(None)
    return folder


def get_labels_dir() -> Path | None:
    """Folder the user picked for YOLO .txt files, if any."""
    return _labels_dir


def set_labels_dir(path: str | Path) -> Path:
    """Set the labels output folder, creating it (and parents) if needed."""
    global _labels_dir
    folder = Path(path).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if not folder.is_dir():
        raise ValueError(f"not a directory: {folder}")
    _labels_dir = folder
    ensure_label_files()
    return folder


def ensure_label_files() -> int:
    """Give every image an empty .txt so labels map 1:1 onto the images."""
    if get_images_dir() is None or _labels_dir is None:
        return 0
    existing = {
        p.stem for p in _labels_dir.glob("*.txt")
        if p.stem not in LABEL_SIDECAR_STEMS
    }
    created = 0
    for item in list_image_files():
        stem = item["stem"]
        if stem in existing or stem in LABEL_SIDECAR_STEMS:
            continue
        try:
            (_labels_dir / f"{stem}.txt").write_text("", encoding="utf-8")
        except OSError:
            continue
        created += 1
    if created:
        print(f"Created {created} empty labels in {_labels_dir}", flush=True)
    return created


def effective_labels_dir() -> Path:
    """Folder YOLO .txt files are written to.

    When an images folder is open, prefer its paired labels dir
    (workspace/labels/<folder>_labels) unless the user picked a different one.
    """
    images = get_images_dir()
    if images is not None:
        paired = paired_labels_dir(images)
        if _labels_dir is None or _labels_dir.resolve() == paired.resolve():
            return paired
        return _labels_dir
    return _labels_dir if _labels_dir is not None else config.LABEL_DIR


def list_image_files() -> list[dict]:
    folder = get_images_dir()
    if folder is None or not folder.exists():
        return []
    files = []
    for p in sorted(folder.iterdir(), key=lambda x: x.name.lower()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            files.append({"name": p.name, "stem": p.stem})
    return files


def _count_yolo_file(path: Path) -> tuple[dict[int, int], set[int]]:
    """Return (masks_per_class, class ids present) for one YOLO .txt."""
    masks: dict[int, int] = {}
    present: set[int] = set()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return masks, present
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        token = line.split(None, 1)[0]
        try:
            cid = int(float(token))
        except ValueError:
            continue
        if cid < 0:
            continue
        masks[cid] = masks.get(cid, 0) + 1
        present.add(cid)
    return masks, present


def label_stats() -> dict:
    """Mask and image counts per class for the open image folder's labels."""
    folder = get_labels_dir()
    stems = [item["stem"] for item in list_image_files()]
    if not stems and folder is not None and folder.is_dir():
        stems = sorted(
            p.stem for p in folder.glob("*.txt") if p.stem not in LABEL_SIDECAR_STEMS
        )
    masks: dict[int, int] = {}
    images: dict[int, int] = {}
    labelled = 0
    if folder is not None and folder.is_dir():
        for stem in stems:
            path = folder / f"{stem}.txt"
            if not path.is_file():
                continue
            file_masks, present = _count_yolo_file(path)
            if not present:
                continue
            labelled += 1
            for cid, n in file_masks.items():
                masks[cid] = masks.get(cid, 0) + n
            for cid in present:
                images[cid] = images.get(cid, 0) + 1
    return {
        "ok": True,
        "masks": {str(k): v for k, v in sorted(masks.items())},
        "images": {str(k): v for k, v in sorted(images.items())},
        "images_total": len(stems),
        "images_labelled": labelled,
    }


def empty_class_profiles() -> dict:
    return {"version": 1, "active": "", "profiles": {}}


def read_class_profiles() -> dict:
    """Load class_profiles.json, treating anything unreadable as 'no profiles'."""
    try:
        data = json.loads(CLASS_PROFILES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty_class_profiles()
    if not isinstance(data, dict) or not isinstance(data.get("profiles"), dict):
        return empty_class_profiles()
    profiles = data.get("profiles") or {}
    if "Sort bins" in profiles:
        profiles.pop("Sort bins", None)
        if data.get("active") == "Sort bins":
            data["active"] = next(iter(profiles), "")
        data["profiles"] = profiles
    return data


def write_class_profiles(data: dict) -> None:
    """Write via a temp file so an interrupted save cannot truncate the profiles."""
    payload = json.dumps(data, indent=2) + "\n"

    def _atomic(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)

    _atomic(CLASS_PROFILES_PATH)
    # Keep the old project-root copy in sync so git/backups still see it.
    _atomic(config.LEGACY_CLASS_PROFILES_PATH)


def clean_class_profile(value) -> dict | None:
    """Normalise one profile, or None if it has no usable class list."""
    if not isinstance(value, dict) or not isinstance(value.get("classes"), list):
        return None
    names = [str(c) for c in value["classes"]]
    raw_groups = value.get("classGroup")
    groups = raw_groups if isinstance(raw_groups, list) else []
    raw_order = value.get("groupOrder")
    order = [str(g) for g in raw_order if g] if isinstance(raw_order, list) else []
    return {
        "classes": names,
        "classGroup": [
            str(groups[i]) if i < len(groups) and groups[i] else "Custom"
            for i in range(len(names))
        ],
        "groupOrder": order,
    }


def write_label_sidecars(profile_name: str, class_names: list[str]) -> Path:
    """Write class_profile.txt and classes.txt into the current labels folder."""
    folder = get_labels_dir()
    if folder is None:
        raise ValueError("no labels folder selected")
    name = str(profile_name or "").strip()
    if not name:
        raise ValueError("profile name is empty")
    names = [str(c).replace("\n", " ").replace("\r", "") for c in class_names]
    (folder / "class_profile.txt").write_text(name + "\n", encoding="utf-8")
    classes_body = "".join(f"{c}\n" for c in names)
    (folder / "classes.txt").write_text(classes_body, encoding="utf-8")
    return folder


def remap_class(old_class_id: int):
    """Same logic as scripts/02_generate_labels.py -- keep them in sync."""
    if config.FORCE_CLASS_ID is not None:
        return config.FORCE_CLASS_ID
    if config.CLASS_MAP:
        return config.CLASS_MAP.get(old_class_id)
    return old_class_id


def click_format_dir(fmt: str) -> Path:
    if fmt == "pt":
        return PT_DIR
    if fmt == "engine":
        return ENGINE_DIR
    raise ValueError(f"unknown click format: {fmt}")


def click_kind(stem: str) -> str:
    s = stem.lower().replace("-", "_")
    if s.startswith("fastsam"):
        return "fastsam"
    if "sam" in s:
        return "sam"
    return "yolo"


def click_label(stem: str) -> str:
    return CLICK_LABELS.get(stem, stem)


def list_click_models(fmt: str) -> list[dict]:
    folder = click_format_dir(fmt)
    suffix = f".{fmt}"
    items = []
    if not folder.is_dir():
        return items
    for p in sorted(folder.iterdir(), key=lambda x: x.name.lower()):
        if not p.is_file() or p.suffix.lower() != suffix:
            continue
        if p.stem in CLICK_SKIP_STEMS:
            continue
        items.append({
            "id": p.stem,
            "label": click_label(p.stem),
            "file": p.name,
            "kind": click_kind(p.stem),
        })
    return items


def get_click_model(fmt: str, stem: str):
    """Lazy-load a click model from models/dot-pt or models/dot-engine."""
    fmt = str(fmt or DEFAULT_CLICK_FORMAT).strip().lower()
    stem = Path(str(stem or "")).name
    if fmt not in CLICK_FORMATS:
        raise ValueError(f"unknown click format: {fmt}")
    if not stem or stem in {".", ".."} or stem in CLICK_SKIP_STEMS:
        raise ValueError("unknown click model")
    path = (click_format_dir(fmt) / f"{stem}.{fmt}").resolve()
    folder = click_format_dir(fmt).resolve()
    if path.parent != folder:
        raise ValueError("invalid click model path")
    if not path.is_file():
        raise ValueError(f"model not found: {path}")
    kind = click_kind(stem)
    cache_key = f"{fmt}:{stem}"
    spec = {"file": path.name, "kind": kind, "label": click_label(stem), "path": path}
    with _click_lock:
        if cache_key not in _click_models:
            print(f"Loading {path} for click-fallback...")
            from ultralytics import FastSAM, SAM, YOLO

            path_str = str(path)
            if kind == "fastsam":
                _click_models[cache_key] = FastSAM(path_str)
            elif kind == "sam":
                _click_models[cache_key] = SAM(path_str)
            else:
                _click_models[cache_key] = YOLO(path_str)
            print(f"{spec['label']} ({fmt}) loaded.")
        return cache_key, _click_models[cache_key], spec


def _has_mask(results) -> bool:
    result = results[0] if results else None
    return result is not None and result.masks is not None and len(result.masks) > 0


def run_click_predict(kind: str, model, img, px: float, py: float, conf: float):
    """Point-prompt inference. MobileSAM/SAM want a flat [x, y] for one click."""
    if kind == "fastsam":
        return model(img, points=[[px, py]], labels=[1], conf=conf, verbose=False)

    import numpy as np

    source = np.asarray(img)
    x, y = float(px), float(py)
    attempts = (
        {"points": [x, y], "labels": [1]},
        {"points": [[x, y]], "labels": [1]},
        {"points": [[[x, y]]], "labels": [[1]]},
    )
    last = None
    last_err = None
    for kwargs in attempts:
        try:
            last = model.predict(source, verbose=False, **kwargs)
        except Exception as e:
            last_err = e
            continue
        if _has_mask(last):
            return last
    if last is not None:
        return last
    if last_err is not None:
        raise last_err
    return last


def decode_image(img_b64: str) -> Image.Image:
    if "," in img_b64[:60]:
        img_b64 = img_b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")


def label_file(stem: str) -> Path:
    """Resolve a YOLO .txt path inside the chosen labels folder, rejecting path traversal."""
    stem = Path(str(stem)).name
    if not stem or stem in {".", ".."}:
        raise ValueError("invalid label stem")
    folder = effective_labels_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = (folder / f"{stem}.txt").resolve()
    if path.parent != folder.resolve():
        raise ValueError("invalid label stem")
    return path


@app.route("/health")
def health():
    return jsonify({
        "status": "ok" if not _boot["fatal"] else "error",
        "app": "segriotate",
        "ready": bool(_boot["ready"]),
        "message": _boot["message"],
        "error": _boot["fatal"],
        "detect_model": str(config.MODEL_PATH),
        "detect_loaded": detect_model is not None,
        "click_models_pt": list_click_models("pt"),
        "click_models_engine": list_click_models("engine"),
        "click_models_loaded": sorted(_click_models),
        "label_dir": str(effective_labels_dir()),
        "labels_dir_set": get_labels_dir() is not None,
        "images_dir": str(get_images_dir()) if get_images_dir() else None,
    })


@app.route("/click-models")
def click_models():
    """List click-to-segment weights in models/dot-pt and models/dot-engine."""
    return jsonify({
        "formats": list(CLICK_FORMATS),
        "pt": list_click_models("pt"),
        "engine": list_click_models("engine"),
        "dirs": {"pt": str(PT_DIR), "engine": str(ENGINE_DIR)},
    })


@app.route("/")
def index():
    if not _boot["ready"]:
        return send_from_directory(TOOLS_DIR, "launching.html")
    return send_from_directory(TOOLS_DIR, "home.html")


@app.route("/annotate")
def annotate():
    if not _boot["ready"]:
        return send_from_directory(TOOLS_DIR, "launching.html")
    return send_from_directory(TOOLS_DIR, "label_editor.html")


@app.route("/train")
def train_page():
    return send_from_directory(TOOLS_DIR, "trainer.html")


@app.route("/sort")
def sort_page():
    return send_from_directory(TOOLS_DIR, "sorter.html")


@app.route("/labs.css")
def labs_css():
    return send_from_directory(TOOLS_DIR, "labs.css")


@app.route("/icon.png")
def app_icon_png():
    return send_from_directory(ICON_PNG.parent, ICON_PNG.name)


@app.route("/favicon.ico")
def favicon():
    if ICON_ICO.is_file():
        return send_from_directory(ICON_ICO.parent, ICON_ICO.name)
    return send_from_directory(ICON_PNG.parent, ICON_PNG.name)


@app.route("/vendor/<path:filename>")
def vendor(filename):
    return send_from_directory(VENDOR_DIR, filename)


@app.route("/project/images")
def project_images():
    folder = get_images_dir()
    return jsonify({
        "dir": str(folder) if folder else None,
        "files": list_image_files(),
    })


@app.route("/project/images-dir", methods=["POST"])
def project_images_dir():
    data = request.get_json(force=True, silent=True) or {}
    try:
        folder = set_images_dir(data.get("path", ""))
    except (TypeError, ValueError) as e:
        return jsonify({"error": str(e)}), 400
    labels = get_labels_dir()
    return jsonify({
        "ok": True,
        "dir": str(folder),
        "files": list_image_files(),
        "labels_dir": str(labels) if labels else None,
    })


@app.route("/project/labels-dir", methods=["GET", "POST"])
def project_labels_dir():
    if request.method == "GET":
        folder = get_labels_dir()
        if folder is None and get_images_dir() is not None:
            folder = effective_labels_dir()
        return jsonify({
            "dir": str(folder) if folder else None,
            "effective": str(effective_labels_dir()),
        })
    data = request.get_json(force=True, silent=True) or {}
    try:
        folder = set_labels_dir(data.get("path", ""))
    except (TypeError, ValueError, OSError) as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"ok": True, "dir": str(folder)})


@app.route("/project/class-profiles", methods=["GET", "POST", "PUT", "OPTIONS"])
def project_class_profiles():
    """Named class lists the editor's left panel saves and reloads."""
    if request.method == "OPTIONS":
        return ("", 204)
    if request.method == "GET":
        return jsonify({"path": str(CLASS_PROFILES_PATH), **read_class_profiles()})

    payload = request.get_json(force=True, silent=True) or {}
    incoming = payload.get("profiles")
    if not isinstance(incoming, dict):
        return jsonify({"error": "profiles must be an object"}), 400
    profiles = {}
    for name, value in incoming.items():
        label = str(name).strip()
        cleaned = clean_class_profile(value)
        if label and cleaned is not None:
            profiles[label] = cleaned
    existing = read_class_profiles()
    # A stale Annotate page load used to POST {} and wipe saved profiles.
    # Only allow an empty catalog when the user explicitly deleted the last one.
    if not profiles and (existing.get("profiles") or {}) and not payload.get("allow_empty"):
        return jsonify({"ok": True, "path": str(CLASS_PROFILES_PATH), **existing})
    active = str(payload.get("active") or "").strip()
    if active not in profiles:
        active = next(iter(profiles), "")
    doc = {"version": 1, "active": active, "profiles": profiles}
    try:
        write_class_profiles(doc)
    except OSError as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"ok": True, "path": str(CLASS_PROFILES_PATH), **doc})


@app.route("/project/labels-taxonomy", methods=["POST", "OPTIONS"])
def project_labels_taxonomy():
    """Sidecar files in the labels folder: which profile, and id → name."""
    if request.method == "OPTIONS":
        return ("", 204)
    payload = request.get_json(force=True, silent=True) or {}
    profile = str(payload.get("profile") or "").strip()
    classes = payload.get("classes")
    if not isinstance(classes, list):
        classes = []
    try:
        folder = write_label_sidecars(profile, classes)
    except (TypeError, ValueError, OSError) as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({
        "ok": True,
        "dir": str(folder),
        "profile": profile,
        "classes": [str(c) for c in classes],
    })


@app.route("/media/<path:filename>")
def media(filename):
    folder = get_images_dir()
    if folder is None:
        abort(404)
    name = Path(filename).name
    path = (folder / name).resolve()
    if path.parent != folder.resolve() or not path.is_file():
        abort(404)
    return send_from_directory(folder, name)


@app.route("/project/split-dataset", methods=["POST", "OPTIONS"])
def project_split_dataset():
    """Copy the open images + labels into a user-chosen train/val/test folder."""
    if request.method == "OPTIONS":
        return ("", 204)
    images = get_images_dir()
    labels = get_labels_dir()
    if images is None:
        return jsonify({"error": "open an images folder first"}), 400
    if labels is None:
        return jsonify({"error": "no labels folder is set"}), 400
    data = request.get_json(force=True, silent=True) or {}
    out = (data.get("out") or "").strip()
    if not out:
        return jsonify({"error": "choose a dataset folder"}), 400
    try:
        train = float(data.get("train", 0.7))
        val = float(data.get("val", 0.2))
        test = float(data.get("test", 0.1))
        seed = int(data.get("seed", 42))
    except (TypeError, ValueError):
        return jsonify({"error": "train, val, test, and seed must be numbers"}), 400
    try:
        result = split_dataset(
            images, labels, out, train=train, val=val, test=test, seed=seed
        )
    except (ValueError, OSError) as e:
        return jsonify({"error": str(e)}), 400
    save_last_dataset({
        "root": result.get("out"),
        "yaml": result.get("yaml"),
        "counts": result.get("counts"),
        "labelled": result.get("labelled"),
        "csv": result.get("csv"),
    })
    return jsonify(result)


def _job_payload(job, since: int = 0) -> dict:
    logs = job.logs[since:] if since else job.logs[-500:]
    return {
        "job_id": job.job_id,
        "state": job.state,
        "progress": job.progress,
        "output_dir": job.output_dir,
        "error": job.error,
        "logs": logs,
        "log_count": len(job.logs),
    }


@app.route("/project/train/models")
def train_models():
    groups = list_model_groups()
    return jsonify({
        "groups": groups,
        "models": [m for g in groups for m in g["models"]],
    })


@app.route("/project/train/devices")
def train_devices():
    return jsonify({"devices": detect_devices()})


@app.route("/project/train/runtime")
def train_runtime():
    return jsonify({
        "platform": sys.platform,
        "project": str(default_runs_dir()),
        "workers": 0,
    })


@app.route("/project/train/validate", methods=["POST", "OPTIONS"])
def train_validate():
    if request.method == "OPTIONS":
        return ("", 204)
    data = request.get_json(force=True, silent=True) or {}
    try:
        dataset = DatasetConfig(**data)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "issues": []}), 400
    result = validate_dataset(dataset)
    return jsonify(result.model_dump())


@app.route("/project/train/presets", methods=["GET", "POST", "OPTIONS"])
def train_presets_collection():
    if request.method == "OPTIONS":
        return ("", 204)
    if request.method == "GET":
        return jsonify({"presets": train_presets.list_presets()})
    data = request.get_json(force=True, silent=True) or {}
    name = (data.get("name") or "").strip()
    config_data = data.get("config")
    if not name or not isinstance(config_data, dict):
        return jsonify({"error": "name and config are required"}), 400
    try:
        cfg = TrainConfig(**config_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    train_presets.save_preset(name, cfg.model_dump())
    return jsonify({"saved": True})


@app.route("/project/train/presets/<name>", methods=["GET", "DELETE", "OPTIONS"])
def train_preset_item(name):
    if request.method == "OPTIONS":
        return ("", 204)
    if request.method == "DELETE":
        train_presets.delete_preset(name)
        return jsonify({"deleted": True})
    try:
        return jsonify(train_presets.load_preset(name))
    except FileNotFoundError:
        return jsonify({"error": "Preset not found"}), 404


@app.route("/project/train/last-dataset")
def train_last_dataset():
    info = load_last_dataset()
    if not info:
        return jsonify({"ok": False, "last": None})
    yaml_path = (info.get("yaml") or "").strip()
    root = (info.get("root") or "").strip()
    inspect_root = None
    if yaml_path and Path(yaml_path).is_file():
        inspect_root = Path(yaml_path).parent
    elif root:
        inspect_root = Path(root)
    inspected = inspect_dataset_root(inspect_root) if inspect_root else {"ok": False}
    return jsonify({"ok": bool(inspected.get("ok")), "last": info, "inspect": inspected})


@app.route("/project/train/inspect", methods=["POST", "OPTIONS"])
def train_inspect():
    if request.method == "OPTIONS":
        return ("", 204)
    data = request.get_json(force=True, silent=True) or {}
    root = (data.get("root") or "").strip()
    if not root:
        return jsonify({"ok": False, "error": "choose a dataset folder"}), 400
    result = inspect_dataset_root(root)
    return jsonify(result), (200 if result.get("ok") else 400)


@app.route("/project/train/start", methods=["POST", "OPTIONS"])
def train_start():
    if request.method == "OPTIONS":
        return ("", 204)
    data = request.get_json(force=True, silent=True) or {}
    try:
        cfg = TrainConfig(**data)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    if cfg.dataset.dataset_root and not cfg.dataset.use_raw_yaml:
        yaml_path = Path(cfg.dataset.dataset_root) / "data.yaml"
        if yaml_path.is_file():
            cfg.dataset.use_raw_yaml = True
            cfg.dataset.raw_yaml_path = str(yaml_path)
    validation = validate_dataset(cfg.dataset)
    if not validation.ok:
        return jsonify({
            "error": "Dataset validation failed",
            "issues": [i.model_dump() for i in validation.issues],
        }), 400
    if not Path(cfg.base_model_path).is_file():
        return jsonify({"error": f"weights not found: {cfg.base_model_path}"}), 400
    if not (cfg.project or "").strip() or cfg.project.strip() == "runs":
        cfg.project = str(default_runs_dir())
    job_id = job_manager.submit(cfg)
    return jsonify({"job_id": job_id})


@app.route("/project/train/jobs")
def train_jobs():
    return jsonify({
        "jobs": [
            {
                "job_id": j.job_id,
                "state": j.state,
                "progress": j.progress,
                "output_dir": j.output_dir,
                "error": j.error,
            }
            for j in job_manager.list_jobs()
        ]
    })


@app.route("/project/train/jobs/<job_id>")
def train_job(job_id):
    job = job_manager.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    since = request.args.get("since", default=0, type=int) or 0
    return jsonify(_job_payload(job, since))


@app.route("/project/train/jobs/<job_id>/stop", methods=["POST", "OPTIONS"])
def train_stop(job_id):
    if request.method == "OPTIONS":
        return ("", 204)
    ok = job_manager.stop(job_id)
    if not ok:
        return jsonify({"error": "Job not running or not found"}), 400
    return jsonify({"stopped": True})


def _clean_bins(raw) -> list[dict]:
    bins = []
    if not isinstance(raw, list):
        return bins
    seen_ids = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        bin_id = str(item.get("id") or "").strip() or f"bin-{len(bins)}"
        if bin_id in seen_ids:
            bin_id = f"{bin_id}-{len(bins)}"
        seen_ids.add(bin_id)
        stems = []
        for stem in item.get("stems") or []:
            name = Path(str(stem)).name
            if name and name not in {".", ".."}:
                stems.append(name)
        bins.append({
            "id": bin_id,
            "name": str(item.get("name") or "").strip(),
            "unsorted": bool(item.get("unsorted")),
            "stems": stems,
        })
    return bins


def _labeled_stems(labels_dir: Path | None) -> set[str]:
    if labels_dir is None or not labels_dir.is_dir():
        return set()
    found: set[str] = set()
    for path in labels_dir.glob("*.txt"):
        if path.stem in LABEL_SIDECAR_STEMS:
            continue
        try:
            if path.read_text(encoding="utf-8").strip():
                found.add(path.stem)
        except OSError:
            continue
    return found


def _mark_bins_annotated(bins: list[dict]) -> list[dict]:
    images = get_images_dir()
    labels = get_labels_dir()
    if images is not None:
        labels = effective_labels_dir()
    elif labels is None:
        labels = None
    labeled = _labeled_stems(labels)
    marked = []
    for item in bins:
        stems = item.get("stems") or []
        annotated = bool(stems) and all(stem in labeled for stem in stems)
        marked.append({**item, "annotated": annotated})
    return marked


@app.route("/project/sort/status")
def sort_status():
    if sort_job.folder_hidden():
        return jsonify({
            **sort_job.status(),
            "images_dir": None,
            "files": [],
            "bins": None,
            "saved_embedder": None,
        })
    images = get_images_dir()
    saved = sort_job.load_bins(images) if images else None
    bins = _mark_bins_annotated(_clean_bins((saved or {}).get("bins"))) if saved else None
    return jsonify({
        **sort_job.status(),
        "images_dir": str(images) if images else None,
        "files": list_image_files(),
        "bins": bins,
        "saved_embedder": (saved or {}).get("embedder") if saved else None,
    })


@app.route("/project/sort/run", methods=["POST", "OPTIONS"])
def sort_run():
    if request.method == "OPTIONS":
        return ("", 204)
    images = get_images_dir()
    files = list_image_files()
    if images is None:
        return jsonify({"error": "open an images folder first"}), 400
    if not files:
        return jsonify({"error": "no images in that folder"}), 400
    ok = sort_job.start(files, images, detect_model)
    if not ok:
        return jsonify({"error": "a sort is already running"}), 409
    return jsonify({"ok": True, **sort_job.status()})


@app.route("/project/sort/clear", methods=["POST", "OPTIONS"])
def sort_clear():
    if request.method == "OPTIONS":
        return ("", 204)
    result = sort_job.clear_cache(get_images_dir())
    if not result.get("ok"):
        return jsonify(result), 409
    clear_images_dir()
    return jsonify(result)


@app.route("/project/sort/bins", methods=["GET", "PUT", "OPTIONS"])
def sort_bins():
    if request.method == "OPTIONS":
        return ("", 204)
    images = get_images_dir()
    if images is None:
        return jsonify({"error": "open an images folder first"}), 400
    if request.method == "GET":
        saved = sort_job.load_bins(images) or {"bins": [], "embedder": ""}
        bins = _mark_bins_annotated(_clean_bins(saved.get("bins")))
        return jsonify({
            "ok": True,
            "images_dir": str(images),
            "embedder": saved.get("embedder") or "",
            "bins": bins,
        })
    data = request.get_json(force=True, silent=True) or {}
    bins = _clean_bins(data.get("bins"))
    saved = sort_job.load_bins(images) or {}
    payload = {
        "images_dir": str(images),
        "embedder": str(data.get("embedder") or saved.get("embedder") or ""),
        "bins": bins,
    }
    path = sort_job.save_bins(images, payload)
    return jsonify({
        "ok": True,
        "path": str(path),
        **payload,
        "bins": _mark_bins_annotated(bins),
    })


@app.route("/project/sort/session", methods=["GET", "POST", "OPTIONS"])
def sort_session():
    if request.method == "OPTIONS":
        return ("", 204)
    if request.method == "POST":
        data = request.get_json(force=True, silent=True) or {}
        if data.get("clear"):
            sort_job.set_session(None)
            return jsonify({"ok": True, "session": None})
        return jsonify({"error": "unknown session action"}), 400
    session = sort_job.get_session()
    if not session:
        return jsonify({"ok": False, "session": None})
    images = session.get("images_dir")
    labels = session.get("labels_dir")
    try:
        if images and Path(str(images)).is_dir():
            current = get_images_dir()
            wanted = Path(str(images)).expanduser().resolve()
            if current is None or current.resolve() != wanted:
                set_images_dir(wanted)
        if labels:
            set_labels_dir(labels)
        elif get_images_dir() is not None:
            set_labels_dir(paired_labels_dir(get_images_dir()))
    except (TypeError, ValueError, OSError):
        pass
    session = {
        **session,
        "images_dir": str(get_images_dir()) if get_images_dir() else images,
        "labels_dir": str(get_labels_dir()) if get_labels_dir() else labels,
    }
    sort_job.set_session(session)
    return jsonify({"ok": True, "session": session})


@app.route("/project/sort/apply", methods=["POST", "OPTIONS"])
def sort_apply():
    """Open one bin in Annotate. Classes stay in the editor — Sort does not create them."""
    if request.method == "OPTIONS":
        return ("", 204)
    images = get_images_dir()
    if images is None:
        return jsonify({"error": "open an images folder first"}), 400
    data = request.get_json(force=True, silent=True) or {}
    bins = _clean_bins(data.get("bins"))
    if not bins:
        saved = sort_job.load_bins(images) or {}
        bins = _clean_bins(saved.get("bins"))
    bin_id = str(data.get("bin_id") or "").strip()
    chosen = next((item for item in bins if item["id"] == bin_id), None)
    if chosen is None or not chosen["stems"]:
        return jsonify({"error": "that bin has no images"}), 400
    saved = sort_job.load_bins(images) or {}
    sort_job.save_bins(images, {
        "images_dir": str(images),
        "embedder": str(data.get("embedder") or saved.get("embedder") or ""),
        "bins": bins,
    })
    try:
        set_labels_dir(paired_labels_dir(images))
    except OSError as e:
        return jsonify({"error": str(e)}), 500
    session = {
        "images_dir": str(images),
        "labels_dir": str(get_labels_dir()) if get_labels_dir() else None,
        "bin_id": chosen["id"],
        "stems": chosen["stems"],
    }
    sort_job.set_session(session)
    return jsonify({"ok": True, "session": session, "annotate": "/annotate?from=sort"})


@app.route("/project/label-stats", methods=["GET", "OPTIONS"])
def project_label_stats():
    if request.method == "OPTIONS":
        return ("", 204)
    return jsonify(label_stats())


@app.route("/label", methods=["GET", "POST", "PUT", "OPTIONS"])
def label():
    """Read or write a YOLO .txt in labels/. OPTIONS is required for browser CORS."""
    if request.method == "OPTIONS":
        return ("", 204)

    if request.method == "GET":
        try:
            path = label_file(request.args.get("stem", ""))
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        if not path.exists():
            return jsonify({"text": None, "path": str(path)})
        return jsonify({"text": path.read_text(encoding="utf-8"), "path": str(path)})

    data = request.get_json(force=True, silent=True) or {}
    try:
        path = label_file(data.get("stem", ""))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    text = data.get("text", "")
    if not isinstance(text, str):
        return jsonify({"error": "text must be a string"}), 400
    folder = effective_labels_dir()
    folder.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".txt.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return jsonify({"ok": True, "path": str(path)})


@app.route("/detect", methods=["POST"])
def detect():
    """Run the user's own model on one image, return all detected polygons."""
    data = request.get_json(force=True)
    try:
        img = decode_image(data["image"])
    except Exception as e:
        return jsonify({"error": f"could not decode image: {e}"}), 400

    conf = float(data.get("conf", config.MIN_CONFIDENCE))

    if detect_model is None:
        if not _boot["ready"]:
            return jsonify({"error": _boot["message"] or "Launching app. Please wait"}), 503
        return jsonify({
            "error": "Auto-Detect model not loaded. Put segmentation.pt in models/dot-pt.",
        }), 503

    try:
        results = detect_model.predict(img, conf=conf, verbose=False)
    except Exception as e:
        return jsonify({"error": f"inference failed: {e}"}), 500

    result = results[0]
    objects = []
    if result.masks is not None and result.boxes is not None:
        for polygon, cls_tensor in zip(result.masks.xyn, result.boxes.cls):
            old_id = int(cls_tensor.item())
            new_id = remap_class(old_id)
            if new_id is None:
                continue
            pts = polygon.tolist()
            if len(pts) < 3:
                continue
            objects.append({"classId": new_id, "points": pts})

    return jsonify({"objects": objects})


@app.route("/segment", methods=["POST"])
def segment():
    """Point-prompted click fallback (.pt or .engine) for objects YOLO missed."""
    data = request.get_json(force=True)
    try:
        img = decode_image(data["image"])
    except Exception as e:
        return jsonify({"error": f"could not decode image: {e}"}), 400

    if not _boot["ready"]:
        return jsonify({"error": _boot["message"] or "Launching app. Please wait"}), 503

    try:
        key, model, spec = get_click_model(
            data.get("format", DEFAULT_CLICK_FORMAT),
            data.get("model", ""),
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    x, y = float(data["x"]), float(data["y"])   # normalized 0-1
    conf = float(data.get("conf", 0.4))
    w, h = img.size
    px, py = x * w, y * h

    try:
        results = run_click_predict(spec["kind"], model, img, px, py, conf)
    except Exception as e:
        return jsonify({"error": f"inference failed: {e}"}), 500

    result = results[0] if results else None
    if result is None or result.masks is None or len(result.masks) == 0:
        return jsonify({"points": [], "model": key, "label": spec["label"]})

    polygon = result.masks.xyn[0].tolist()
    return jsonify({"points": polygon, "model": key, "label": spec["label"]})


if __name__ == "__main__":
    print(f"Segri-Labs running on http://127.0.0.1:{PORT}")
    print("Open that URL in a browser, or run: python desktop_app.py\n")
    app.run(host="127.0.0.1", port=PORT, threaded=True, use_reloader=False)
