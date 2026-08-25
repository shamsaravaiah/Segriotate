"""
Splits the auto-accepted (and, once you've merged them back in, CVAT-reviewed)
images + labels into dataset/{images,labels}/{train,val,test}/ and writes
dataset/data.yaml.

Run this after labels exist in labels/ (from the live editor) or, as a
fallback, in output/labels_auto/ from 02_generate_labels.py.

Usage:
    python scripts/03_split_dataset.py
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

import shutil  # noqa: E402
import yaml  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main():
    random.seed(config.SPLIT_SEED)

    ratios = (config.TRAIN_RATIO, config.VAL_RATIO, config.TEST_RATIO)
    if abs(sum(ratios) - 1.0) > 1e-6:
        sys.exit(f"TRAIN/VAL/TEST ratios must sum to 1.0, got {sum(ratios)}")

    label_dir = config.LABEL_DIR
    if not any(label_dir.glob("*.txt")):
        label_dir = config.AUTO_LABEL_DIR
    if not label_dir.exists():
        sys.exit(f"No labels found in {config.LABEL_DIR} or {config.AUTO_LABEL_DIR}.")

    labelled_stems = {
        p.stem for p in label_dir.glob("*.txt")
        if p.stem not in {"classes", "class_profile"}
    }
    images = [
        p for p in config.IMAGE_DIR.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS and p.stem in labelled_stems
    ]

    if not images:
        sys.exit(f"No images with matching labels found. Check {label_dir}.")

    random.shuffle(images)
    n = len(images)
    train_end = int(n * config.TRAIN_RATIO)
    val_end = train_end + int(n * config.VAL_RATIO)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }

    for split, split_images in splits.items():
        img_out = config.DATASET_DIR / "images" / split
        lbl_out = config.DATASET_DIR / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for image_path in split_images:
            shutil.copy2(image_path, img_out / image_path.name)
            label_path = label_dir / f"{image_path.stem}.txt"
            shutil.copy2(label_path, lbl_out / label_path.name)

    data_yaml = {
        "path": str(config.DATASET_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {int(k): v for k, v in config.CLASS_NAMES.items()},
    }
    yaml_path = config.DATASET_DIR / "data.yaml"
    yaml_path.write_text(yaml.dump(data_yaml, sort_keys=False))

    print("Dataset created:")
    for split, split_images in splits.items():
        print(f"  {split:<6} {len(split_images)}")
    print(f"\nWrote {yaml_path}")


if __name__ == "__main__":
    main()
