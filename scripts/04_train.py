"""
Trains (or fine-tunes) a YOLO segmentation model on dataset/data.yaml.

Usage:
    python scripts/04_train.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

from ultralytics import YOLO  # noqa: E402


def main():
    data_yaml = config.DATASET_DIR / "data.yaml"
    if not data_yaml.exists():
        sys.exit(f"{data_yaml} not found -- run 03_split_dataset.py first.")

    model = YOLO(config.TRAIN_BASE_MODEL)

    model.train(
        data=str(data_yaml),
        epochs=config.TRAIN_EPOCHS,
        imgsz=config.TRAIN_IMGSZ,
        batch=config.TRAIN_BATCH,
        project=config.TRAIN_PROJECT,
        name=config.TRAIN_NAME,
    )


if __name__ == "__main__":
    main()
