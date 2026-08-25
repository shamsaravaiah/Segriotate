"""
Local inference + UI server for Segriotate.

1. /          -- the label editor (HTML)
2. /detect    -- YOLO segmentation.pt on the current image
3. /segment   -- click-to-segment fallback (.pt or .engine: FastSAM, MobileSAM, …)
4. /label     -- read/write YOLO .txt files in labels/
5. /media     -- serve images from the chosen images folder

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

os.chdir(config.PROJECT_ROOT)

from flask import Flask, abort, jsonify, request, send_from_directory  # noqa: E402
from flask_cors import CORS  # noqa: E402
from PIL import Image  # noqa: E402

from ensure_models import ensure_models  # noqa: E402
from build_engines import ensure_engines  # noqa: E402

PORT = getattr(config, "SERVER_PORT", 8765)
PT_DIR = config.PROJECT_ROOT / "models" / "dot-pt"
ENGINE_DIR = config.PROJECT_ROOT / "models" / "dot-engine"
CLICK_FORMATS = ("pt", "engine")
DEFAULT_CLICK_FORMAT = "pt"
CLICK_SKIP_STEMS = {"segmentation"}  # Auto-Detect YOLO, not a click model
CLICK_LABELS = {
    "FastSAM-s": "FastSAM",
    "FastSAM-x": "FastSAM-x",
    "mobile_sam": "MobileSAM",
}
TOOLS_DIR = config.PROJECT_ROOT / "tools"
VENDOR_DIR = TOOLS_DIR / "vendor"
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
    """labels/<folder-name>_labels under the project root (e.g. batch001 → labels/batch001_labels)."""
    name = images_folder.name.strip() or "images"
    if name in {".", ".."}:
        name = "images"
    labels_root = config.PROJECT_ROOT / "labels"
    dest = (labels_root / f"{name}_labels").resolve()
    legacy = (labels_root / f"{name}-labels").resolve()
    # Keep using an existing hyphen folder from older app versions.
    if not dest.exists() and legacy.is_dir():
        return legacy
    return dest


def set_images_dir(path: str | Path) -> Path:
    global _images_dir
    folder = Path(path).expanduser().resolve()
    if not folder.is_dir():
        raise ValueError(f"not a directory: {folder}")
    _images_dir = folder
    set_labels_dir(paired_labels_dir(folder))
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
    return data


def write_class_profiles(data: dict) -> None:
    """Write via a temp file so an interrupted save cannot truncate the profiles."""
    CLASS_PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CLASS_PROFILES_PATH.with_name(CLASS_PROFILES_PATH.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, CLASS_PROFILES_PATH)


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
    return send_from_directory(TOOLS_DIR, "label_editor.html")


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
    if not profiles:
        return jsonify({"error": "profiles must not be empty"}), 400
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
    print(f"Segriotate running on http://127.0.0.1:{PORT}")
    print("Open that URL in a browser, or run: python desktop_app.py\n")
    app.run(host="127.0.0.1", port=PORT, threaded=True, use_reloader=False)
