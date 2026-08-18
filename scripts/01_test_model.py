"""
Run the existing YOLO-seg model on a handful of images so you can eyeball
the mask quality BEFORE burning time on all 10,000 images.

Usage:
    python scripts/01_test_model.py            # first 5 images
    python scripts/01_test_model.py --n 20      # first 20 images
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

from ultralytics import YOLO  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="number of images to test on")
    args = parser.parse_args()

    if not config.MODEL_PATH.exists():
        sys.exit(f"Model not found at {config.MODEL_PATH} -- put your .pt file there.")

    images = sorted(
        p for p in config.IMAGE_DIR.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        sys.exit(f"No images found in {config.IMAGE_DIR}")

    sample = images[: args.n]
    print(f"Testing on {len(sample)} image(s) from {config.IMAGE_DIR}")

    model = YOLO(str(config.MODEL_PATH))

    results = model.predict(
        source=[str(p) for p in sample],
        conf=config.MIN_CONFIDENCE,
        save=True,
        project=str(config.PROJECT_ROOT / "output"),
        name="test_predictions",
        exist_ok=True,
    )

    for r, p in zip(results, sample):
        n_instances = 0 if r.masks is None else len(r.masks.xyn)
        confs = [] if r.boxes is None else [round(float(c), 2) for c in r.boxes.conf]
        print(f"  {p.name}: {n_instances} instance(s), confidences={confs}")

    print(f"\nAnnotated images saved to output/test_predictions/. "
          f"Check them before running 02_generate_labels.py on the full set.")


if __name__ == "__main__":
    main()
