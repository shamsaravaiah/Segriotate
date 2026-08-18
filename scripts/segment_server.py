"""
Local inference server for the label editor. Two jobs:

1. /detect  -- runs YOUR trained segmentation.pt on the current image
               (called automatically each time you switch images in the
               editor, so predictions show up live instead of needing the
               batch script run first).

2. /segment -- point-prompted fallback using FastSAM, for fruits your
               model misses (e.g. varieties it wasn't trained on). Only
               called when you click an unlabeled object with
               Click-to-Segment mode on.

3. /label   -- read/write YOLO .txt files in the project labels/ folder
               (save from the editor goes here automatically).

Both run fully locally once weights are on disk. FastSAM weights
(~140MB) download once on first run and need internet for that one time;
after that everything is offline.

Usage:
    python scripts/segment_server.py
Then open tools/label_editor.html in Chrome/Edge and leave this running.
"""

import base64
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

from flask import Flask, jsonify, request  # noqa: E402
from flask_cors import CORS  # noqa: E402
from PIL import Image  # noqa: E402
from ultralytics import YOLO, FastSAM  # noqa: E402

PORT = 8765
FASTSAM_MODEL = "FastSAM-s.pt"

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
        "detect_model": str(config.MODEL_PATH),
        "fastsam_model": FASTSAM_MODEL,
        "label_dir": str(config.LABEL_DIR),
    })


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
    print(f"Server running on http://127.0.0.1:{PORT}")
    print("Leave this running, then open tools/label_editor.html.\n")
    app.run(host="127.0.0.1", port=PORT, threaded=True)
