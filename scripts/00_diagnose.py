"""
Quick diagnostic: run this to figure out exactly why polygons aren't
being generated. Prints model task type, ultralytics version, and what
came back for one image.

Usage:
    python scripts/00_diagnose.py path/to/one_image.jpg
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

import ultralytics  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def main():
    print(f"ultralytics version: {ultralytics.__version__}")
    print(f"MODEL_PATH: {config.MODEL_PATH}")

    model = YOLO(str(config.MODEL_PATH))

    print(f"model.task: {model.task}")
    print(f"\nmodel's full class list (model.names):")
    for cid, name in model.names.items():
        print(f"  {cid}: {name}")

    if model.task != "segment":
        print("\n!!! PROBLEM FOUND !!!")
        print(f"This model's task is '{model.task}', not 'segment'.")
        print("A detection or classification model will never produce masks.")
        print("Fix: point MODEL_PATH at a *-seg.pt model, or a model you")
        print("trained with task=segment.")
        return

    if len(sys.argv) < 2:
        # fall back to the first image in config.IMAGE_DIR
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        images = [p for p in config.IMAGE_DIR.rglob("*") if p.suffix.lower() in exts]
        if not images:
            sys.exit("No image given and none found in config.IMAGE_DIR.")
        image_path = images[0]
    else:
        image_path = Path(sys.argv[1])

    print(f"\nTesting on: {image_path}")

    results = model.predict(source=str(image_path), conf=config.MIN_CONFIDENCE, verbose=True)
    result = results[0]

    print(f"\nresult.boxes: {'None' if result.boxes is None else len(result.boxes)} detections")
    if result.boxes is not None and len(result.boxes) > 0:
        print(f"confidences: {[round(float(c), 3) for c in result.boxes.conf]}")
        print(f"class ids:   {[int(c) for c in result.boxes.cls]}")

    print(f"result.masks: {'None' if result.masks is None else len(result.masks)} masks")

    if result.boxes is not None and len(result.boxes) > 0 and result.masks is None:
        print("\n!!! PROBLEM FOUND !!!")
        print("Boxes were detected but masks is None. This model is producing")
        print("bounding boxes only -- it's a detection model, not segmentation,")
        print("even though model.task said 'segment' (or task mismatch).")
        print("Fix: re-check the model file, or re-export/retrain with segmentation heads.")

    if result.boxes is None or len(result.boxes) == 0:
        print("\nNo detections at all. Try lowering MIN_CONFIDENCE in config.py")
        print(f"(currently {config.MIN_CONFIDENCE}), or confirm this image actually")
        print("contains an object the model was trained to recognize.")


if __name__ == "__main__":
    main()
