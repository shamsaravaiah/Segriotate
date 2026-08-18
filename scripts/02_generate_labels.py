"""
Runs the existing YOLO-seg model over every image, converts masks to YOLO
segmentation polygons, applies class remapping, and splits the results into:

    output/labels_auto/      <- high confidence, ready to train on directly
    review/no_detection/     <- model found nothing (image copied here)
    review/low_confidence/   <- at least one instance below AUTO_ACCEPT_CONFIDENCE
    review/multiple_objects/ <- crowded scenes (see MAX_INSTANCES_BEFORE_REVIEW)

Every review image also gets a YOLO-format .txt with the model's best guess
sitting next to it, so you're correcting predictions in CVAT rather than
labelling from scratch.

Usage:
    python scripts/02_generate_labels.py
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

from tqdm import tqdm  # noqa: E402
from ultralytics import YOLO  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def remap_class(old_class_id: int):
    """Return the new class id for an old class id, or None to drop it."""
    if config.FORCE_CLASS_ID is not None:
        return config.FORCE_CLASS_ID
    if config.CLASS_MAP:
        return config.CLASS_MAP.get(old_class_id)  # None if not in map -> dropped
    return old_class_id  # no remapping configured, keep as-is


def polygon_to_line(new_class_id: int, polygon_xyn) -> str | None:
    points = polygon_xyn.tolist()
    if len(points) < 3:
        return None
    flat = []
    for x, y in points:
        flat.append(f"{x:.6f}")
        flat.append(f"{y:.6f}")
    return f"{new_class_id} " + " ".join(flat)


def process_image(model, image_path: Path):
    """
    Returns:
        lines: list[str]   YOLO segmentation lines (already class-remapped)
        min_conf: float | None   lowest per-instance confidence, or None if no detections
        n_instances: int
    """
    results = model.predict(source=str(image_path), conf=config.MIN_CONFIDENCE, verbose=False)
    result = results[0]

    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return [], None, 0

    lines = []
    confs = []

    for polygon, cls_tensor, conf_tensor in zip(
        result.masks.xyn, result.boxes.cls, result.boxes.conf
    ):
        old_class_id = int(cls_tensor.item())
        new_class_id = remap_class(old_class_id)
        if new_class_id is None:
            continue  # this class isn't wanted in the new dataset

        line = polygon_to_line(new_class_id, polygon)
        if line is None:
            continue

        lines.append(line)
        confs.append(float(conf_tensor.item()))

    min_conf = min(confs) if confs else None
    return lines, min_conf, len(lines)


def route(image_path: Path, lines, min_conf, n_instances):
    """Decide auto-accept vs which review bucket, and write files."""
    if n_instances == 0:
        bucket = "no_detection"
    elif n_instances > config.MAX_INSTANCES_BEFORE_REVIEW:
        bucket = "multiple_objects"
    elif min_conf is not None and min_conf < config.AUTO_ACCEPT_CONFIDENCE:
        bucket = "low_confidence"
    else:
        bucket = None  # auto-accept

    label_text = "\n".join(lines)

    if bucket is None:
        out_label = config.AUTO_LABEL_DIR / f"{image_path.stem}.txt"
        out_label.write_text(label_text)
        return "auto"
    else:
        review_img_dir = config.REVIEW_DIR / bucket / "images"
        review_lbl_dir = config.REVIEW_DIR / bucket / "labels"
        review_img_dir.mkdir(parents=True, exist_ok=True)
        review_lbl_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(image_path, review_img_dir / image_path.name)
        (review_lbl_dir / f"{image_path.stem}.txt").write_text(label_text)
        return bucket


def main():
    if not config.MODEL_PATH.exists():
        sys.exit(f"Model not found at {config.MODEL_PATH} -- put your .pt file there.")

    images = sorted(
        p for p in config.IMAGE_DIR.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        sys.exit(f"No images found in {config.IMAGE_DIR}")

    print(f"Found {len(images)} images")

    config.AUTO_LABEL_DIR.mkdir(parents=True, exist_ok=True)
    for bucket in ("no_detection", "low_confidence", "multiple_objects"):
        (config.REVIEW_DIR / bucket / "images").mkdir(parents=True, exist_ok=True)
        (config.REVIEW_DIR / bucket / "labels").mkdir(parents=True, exist_ok=True)

    model = YOLO(str(config.MODEL_PATH))

    counts = {"auto": 0, "no_detection": 0, "low_confidence": 0, "multiple_objects": 0}
    errors = []

    for image_path in tqdm(images, desc="Labelling"):
        try:
            lines, min_conf, n_instances = process_image(model, image_path)
            bucket = route(image_path, lines, min_conf, n_instances)
            counts[bucket] += 1
        except Exception as e:  # noqa: BLE001
            errors.append((image_path, str(e)))

    print("\n--- Summary ---")
    total = len(images)
    for bucket, n in counts.items():
        pct = 100 * n / total if total else 0
        print(f"  {bucket:<18} {n:>6} ({pct:5.1f}%)")
    if errors:
        print(f"\n{len(errors)} image(s) failed:")
        for path, err in errors[:10]:
            print(f"  {path.name}: {err}")
        if len(errors) > 10:
            print(f"  ...and {len(errors) - 10} more")

    print(
        f"\nAuto-accepted labels: {config.AUTO_LABEL_DIR}\n"
        f"Review buckets:       {config.REVIEW_DIR}\n"
        f"Next: import each review/<bucket>/ folder into CVAT as a task, "
        f"fix the pre-filled masks, then export back to YOLO 1.1 format "
        f"and merge the corrected labels into output/labels_auto/ before "
        f"running 03_split_dataset.py."
    )


if __name__ == "__main__":
    main()
