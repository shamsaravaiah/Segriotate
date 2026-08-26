"""
Split a flat YOLO folder pair (images + labels, same stems) into train/val/test.

Copies files (never moves) so the editor's folders stay intact. Writes:

    <out>/images/{train,val,test}/
    <out>/labels/{train,val,test}/
    <out>/data.yaml
    <out>/split.csv

Only images with at least one mask (a non-empty YOLO line) are copied.
Empty placeholder .txt files created by the editor are skipped.

Stratification assigns unclaimed images per class in id order so each
class keeps roughly the requested train/val/test ratio.
"""

from __future__ import annotations

import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SIDECAR_STEMS = {"classes", "class_profile"}


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def find_images(images_dir: Path) -> dict[str, Path]:
    stem_to_path: dict[str, Path] = {}
    for f in sorted(images_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            if f.stem in stem_to_path:
                continue
            stem_to_path[f.stem] = f
    return stem_to_path


def read_class_names(labels_dir: Path) -> dict[int, str]:
    path = labels_dir / "classes.txt"
    names: dict[int, str] = {}
    if path.is_file():
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        for i, line in enumerate(lines):
            names[i] = line.strip() or f"class_{i}"
    if names:
        return names
    try:
        import config
        return {int(k): str(v) for k, v in getattr(config, "CLASS_NAMES", {}).items()}
    except Exception:
        return {}


def parse_label_file(path: Path) -> tuple[set[int], int]:
    """Return (class ids present, number of YOLO lines)."""
    classes: set[int] = set()
    masks = 0
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return classes, masks
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
        classes.add(cid)
        masks += 1
    return classes, masks


def parse_labels(
    labels_dir: Path, known_stems: set[str]
) -> tuple[dict[str, set[int]], dict[str, int], set[str]]:
    """stem -> class ids, stem -> mask count, stems with empty/unparseable labels."""
    stem_classes: dict[str, set[int]] = {}
    stem_masks: dict[str, int] = {}
    empty_stems: set[str] = set()
    for txt_file in sorted(labels_dir.glob("*.txt")):
        stem = txt_file.stem
        if stem in SIDECAR_STEMS or stem not in known_stems:
            continue
        classes, masks = parse_label_file(txt_file)
        stem_masks[stem] = masks
        if classes:
            stem_classes[stem] = classes
        else:
            empty_stems.add(stem)
    return stem_classes, stem_masks, empty_stems


def cut_three_ways(
    items: list[str], train_ratio: float, val_ratio: float, rng: random.Random
) -> tuple[list[str], list[str], list[str]]:
    shuffled = items[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = min(round(n * train_ratio), n)
    n_val = min(round(n * val_ratio), n - n_train)
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return train, val, test


def assign_splits(
    stem_classes: dict[str, set[int]],
    leftover: set[str] | None,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[dict[str, str], dict[int, dict[str, int]]]:
    class_to_stems: dict[int, list[str]] = defaultdict(list)
    for stem, classes in stem_classes.items():
        for cid in classes:
            class_to_stems[cid].append(stem)

    rng = random.Random(seed)
    assigned: dict[str, str] = {}
    per_class: dict[int, dict[str, int]] = {}

    for class_id in sorted(class_to_stems.keys()):
        candidates = [s for s in class_to_stems[class_id] if s not in assigned]
        train_s, val_s, test_s = cut_three_ways(candidates, train_ratio, val_ratio, rng)
        for s in train_s:
            assigned[s] = "train"
        for s in val_s:
            assigned[s] = "val"
        for s in test_s:
            assigned[s] = "test"
        per_class[class_id] = {
            "train": sum(1 for s in class_to_stems[class_id] if assigned.get(s) == "train"),
            "val": sum(1 for s in class_to_stems[class_id] if assigned.get(s) == "val"),
            "test": sum(1 for s in class_to_stems[class_id] if assigned.get(s) == "test"),
        }

    rest = sorted((leftover or set()) - set(assigned.keys()))
    if rest:
        train_s, val_s, test_s = cut_three_ways(rest, train_ratio, val_ratio, rng)
        for s in train_s:
            assigned[s] = "train"
        for s in val_s:
            assigned[s] = "val"
        for s in test_s:
            assigned[s] = "test"

    return assigned, per_class


def _clear_split_dir(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for p in folder.iterdir():
        if p.is_file():
            p.unlink()


def _write_data_yaml(out_dir: Path, names: dict[int, str]) -> Path:
    try:
        import yaml
    except ImportError:
        yaml = None
    payload = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {int(k): v for k, v in sorted(names.items())},
    }
    path = out_dir / "data.yaml"
    if yaml is not None:
        path.write_text(yaml.dump(payload, sort_keys=False), encoding="utf-8")
        return path
    lines = [
        f"path: {payload['path']}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    for k, v in sorted(payload["names"].items()):
        lines.append(f"  {k}: {v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_split_csv(
    out_dir: Path,
    assigned: dict[str, str],
    stem_to_image: dict[str, Path],
    labels_dir: Path,
    stem_classes: dict[str, set[int]],
    stem_masks: dict[str, int],
    names: dict[int, str],
) -> Path:
    path = out_dir / "split.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "stem", "split", "image", "label", "class_ids", "class_names", "masks", "labelled",
        ])
        for stem in sorted(assigned.keys()):
            img = stem_to_image[stem]
            label = labels_dir / f"{stem}.txt"
            ids = sorted(stem_classes.get(stem, set()))
            writer.writerow([
                stem,
                assigned[stem],
                str(img.resolve()),
                str(label.resolve()) if label.exists() else "",
                ";".join(str(i) for i in ids),
                ";".join(names.get(i, f"class_{i}") for i in ids),
                stem_masks.get(stem, 0),
                "yes" if ids else "no",
            ])
    return path


def split_dataset(
    images_dir: str | Path,
    labels_dir: str | Path,
    out_dir: str | Path,
    train: float = 0.7,
    val: float = 0.2,
    test: float = 0.1,
    seed: int = 42,
) -> dict:
    images_dir = Path(images_dir).expanduser().resolve()
    labels_dir = Path(labels_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()

    if abs((train + val + test) - 1.0) > 1e-6:
        raise ValueError(f"train + val + test must sum to 1.0, got {train + val + test}")
    if min(train, val, test) < 0:
        raise ValueError("train, val, and test cannot be negative")
    if not images_dir.is_dir():
        raise ValueError(f"images folder does not exist: {images_dir}")
    if not labels_dir.is_dir():
        raise ValueError(f"labels folder does not exist: {labels_dir}")
    if out_dir == images_dir or out_dir == labels_dir:
        raise ValueError("dataset folder cannot be the images or labels folder")
    if images_dir.exists() and _is_inside(out_dir, images_dir):
        raise ValueError("dataset folder cannot be inside the images folder")
    if labels_dir.exists() and _is_inside(out_dir, labels_dir):
        raise ValueError("dataset folder cannot be inside the labels folder")
    if (out_dir / "images").resolve() == images_dir:
        raise ValueError("pick a different folder — this one would overwrite the source images")
    if (out_dir / "labels").resolve() == labels_dir:
        raise ValueError("pick a different folder — this one would overwrite the source labels")
    out_dir.mkdir(parents=True, exist_ok=True)

    stem_to_image = find_images(images_dir)
    if not stem_to_image:
        raise ValueError(f"no images found in {images_dir}")

    stem_classes, stem_masks, _empty_stems = parse_labels(labels_dir, set(stem_to_image.keys()))
    skipped = set(stem_to_image.keys()) - set(stem_classes.keys())
    if not stem_classes:
        raise ValueError(
            "no annotated images — draw at least one mask before creating the dataset"
        )
    assigned, per_class = assign_splits(stem_classes, None, train, val, seed)

    names = read_class_names(labels_dir)
    for cid in sorted({c for classes in stem_classes.values() for c in classes}):
        names.setdefault(cid, f"class_{cid}")

    for split_name in ("train", "val", "test"):
        _clear_split_dir(out_dir / "images" / split_name)
        _clear_split_dir(out_dir / "labels" / split_name)

    for stem, split_name in assigned.items():
        img_src = stem_to_image[stem]
        shutil.copy2(img_src, out_dir / "images" / split_name / img_src.name)
        label_src = labels_dir / f"{stem}.txt"
        if label_src.is_file():
            shutil.copy2(label_src, out_dir / "labels" / split_name / label_src.name)

    for sidecar in ("classes.txt", "class_profile.txt"):
        src = labels_dir / sidecar
        if src.is_file():
            shutil.copy2(src, out_dir / sidecar)

    yaml_path = _write_data_yaml(out_dir, names)
    csv_path = _write_split_csv(
        out_dir, assigned, stem_to_image, labels_dir, stem_classes, stem_masks, names
    )

    counts = {"train": 0, "val": 0, "test": 0}
    for split_name in assigned.values():
        counts[split_name] += 1

    return {
        "ok": True,
        "out": str(out_dir),
        "counts": counts,
        "labelled": len(stem_classes),
        "skipped": len(skipped),
        "background": len(skipped),
        "csv": str(csv_path),
        "yaml": str(yaml_path),
        "per_class": {
            str(cid): counts_c for cid, counts_c in sorted(per_class.items())
        },
        "names": {str(k): v for k, v in sorted(names.items())},
    }
