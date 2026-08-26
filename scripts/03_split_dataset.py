"""
Splits annotated images + labels into train/val/test and writes data.yaml.

Prefer the Dataset panel in Segriotate (choose folder + percentages there).
This script is the same split from the command line:

    python scripts/03_split_dataset.py \\
        --images /path/to/batch001 \\
        --labels /path/to/labels/batch001_labels \\
        --out /path/to/my_dataset \\
        --train 0.7 --val 0.2 --test 0.1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from split_yolo_dataset import split_dataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", default=str(config.IMAGE_DIR), help="Folder of images (flat).")
    parser.add_argument("--labels", default="", help="Folder of YOLO .txt labels (flat).")
    parser.add_argument("--out", default=str(config.DATASET_DIR), help="Folder to write train/val/test into.")
    parser.add_argument("--train", type=float, default=config.TRAIN_RATIO)
    parser.add_argument("--val", type=float, default=config.VAL_RATIO)
    parser.add_argument("--test", type=float, default=config.TEST_RATIO)
    parser.add_argument("--seed", type=int, default=config.SPLIT_SEED)
    args = parser.parse_args()

    labels = Path(args.labels) if args.labels else config.LABEL_DIR
    if not args.labels and not any(labels.glob("*.txt")):
        labels = config.AUTO_LABEL_DIR

    try:
        result = split_dataset(
            args.images,
            labels,
            args.out,
            train=args.train,
            val=args.val,
            test=args.test,
            seed=args.seed,
        )
    except (ValueError, OSError) as e:
        sys.exit(str(e))

    counts = result["counts"]
    print("Dataset created:")
    for split in ("train", "val", "test"):
        print(f"  {split:<6} {counts.get(split, 0)}")
    print(f"\nWrote {result['yaml']}")
    print(f"Index  {result['csv']}")


if __name__ == "__main__":
    main()
