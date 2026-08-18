"""
Local inference + UI server for Segriotate.

1. /          -- the label editor (HTML)
2. /detect    -- YOLO segmentation.pt on the current image
3. /segment   -- FastSAM click-to-segment fallback
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
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

os.chdir(config.PROJECT_ROOT)

from flask import Flask, abort, jsonify, request, send_from_directory  # noqa: E402
from flask_cors import CORS  # noqa: E402
from PIL import Image  # noqa: E402
from ultralytics import YOLO, FastSAM  # noqa: E402

PORT = getattr(config, "SERVER_PORT", 8765)
FASTSAM_MODEL = "FastSAM-s.pt"
TOOLS_DIR = config.PROJECT_ROOT / "tools"
VENDOR_DIR = TOOLS_DIR / "vendor"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

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

if not config.MODEL_PATH.exists():
    sys.exit(f"Model not found at {config.MODEL_PATH} -- put your .pt file there first.")

print(f"Loading your model: {config.MODEL_PATH}")
detect_model = YOLO(str(config.MODEL_PATH))
if detect_model.task != "segment":
    sys.exit(f"{config.MODEL_PATH} is a '{detect_model.task}' model, not a segmentation model.")

print(f"Loading {FASTSAM_MODEL} for click-fallback (auto-downloads on first run)...")
fastsam_model = FastSAM(FASTSAM_MODEL)
print("Both models loaded.\n")

_images_dir: Path | None = None


def get_images_dir() -> Path | None:
    """Only the folder the user picked — never a project default."""
    return _images_dir


def set_images_dir(path: str | Path) -> Path:
    global _images_dir
    folder = Path(path).expanduser().resolve()
    if not folder.is_dir():
        raise ValueError(f"not a directory: {folder}")
    _images_dir = folder
    return folder


def list_image_files() -> list[dict]:
    folder = get_images_dir()
    if folder is None or not folder.exists():
        return []
    files = []
    for p in sorted(folder.iterdir(), key=lambda x: x.name.lower()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            files.append({"name": p.name, "stem": p.stem})
    return files


def remap_class(old_class_id: int):
    """Same logic as scripts/02_generate_labels.py -- keep them in sync."""
    if config.FORCE_CLASS_ID is not None:
        return config.FORCE_CLASS_ID
    if config.CLASS_MAP:
        return config.CLASS_MAP.get(old_class_id)
    return old_class_id


def decode_image(img_b64: str) -> Image.Image:
    if "," in img_b64[:60]:
        img_b64 = img_b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")


def label_file(stem: str) -> Path:
    """Resolve a YOLO .txt path inside labels/, rejecting path traversal."""
    stem = Path(str(stem)).name
    if not stem or stem in {".", ".."}:
        raise ValueError("invalid label stem")
    config.LABEL_DIR.mkdir(parents=True, exist_ok=True)
    path = (config.LABEL_DIR / f"{stem}.txt").resolve()
    if path.parent != config.LABEL_DIR.resolve():
        raise ValueError("invalid label stem")
    return path


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "app": "segriotate",
        "detect_model": str(config.MODEL_PATH),
        "fastsam_model": FASTSAM_MODEL,
        "label_dir": str(config.LABEL_DIR),
        "images_dir": str(get_images_dir()) if get_images_dir() else None,
    })


@app.route("/")
def index():
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
    return jsonify({"ok": True, "dir": str(folder), "files": list_image_files()})


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
    config.LABEL_DIR.mkdir(parents=True, exist_ok=True)
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
    """Point-prompted FastSAM fallback for one click, for objects the main model missed."""
    data = request.get_json(force=True)
    try:
        img = decode_image(data["image"])
    except Exception as e:
        return jsonify({"error": f"could not decode image: {e}"}), 400

    x, y = float(data["x"]), float(data["y"])   # normalized 0-1
    conf = float(data.get("conf", 0.4))
    w, h = img.size
    px, py = x * w, y * h

    try:
        results = fastsam_model(img, points=[[px, py]], labels=[1], conf=conf, verbose=False)
    except Exception as e:
        return jsonify({"error": f"inference failed: {e}"}), 500

    result = results[0]
    if result.masks is None or len(result.masks) == 0:
        return jsonify({"points": []})

    polygon = result.masks.xyn[0].tolist()
    return jsonify({"points": polygon})


if __name__ == "__main__":
    print(f"Segriotate running on http://127.0.0.1:{PORT}")
    print("Open that URL in a browser, or run: python desktop_app.py\n")
    app.run(host="127.0.0.1", port=PORT, threaded=True, use_reloader=False)
