"""
Dataset handling.

Two paths are supported:
  1. Form-driven: user enters class count + names in the UI. We generate
     data.yaml automatically, guaranteeing nc/names stay in sync.
  2. Raw yaml: power users can point directly at an existing data.yaml.

Either way, before training starts we validate the dataset folder against
the class list so a mismatch is caught before wasting GPU time.
"""
import yaml
from pathlib import Path
from typing import List

from schemas import DatasetConfig, ValidationResult, ValidationIssue
from dataset_stage import staging_message


def find_val_split_root(root: Path) -> Path:
    """Roboflow-style `valid/` or this repo's splitter `val/`."""
    for name in ("valid", "val"):
        if (root / name / "images").exists() and (root / name / "labels").exists():
            return root / name
    return root / "valid"


def _ultralytics_layout(root: Path) -> bool:
    return (root / "images" / "train").is_dir()


def generate_data_yaml(dataset: DatasetConfig, write_path: Path) -> Path:
    """
    Build a data.yaml from form-entered classes and write it next to the
    dataset (or into a scratch config dir). Returns the path written.
    """
    root = Path(dataset.dataset_root)

    names = {c.id: c.name for c in sorted(dataset.classes, key=lambda c: c.id)}
    if not names:
        names = _names_from_yaml_or_classes(root, None)
    if not names:
        labels_dir = (
            root / "labels" / "train" if _ultralytics_layout(root) else root / "train" / "labels"
        )
        names = {i: f"class_{i}" for i in _scan_label_class_ids(labels_dir)}

    if _ultralytics_layout(root):
        val_rel = "images/val" if (root / "images" / "val").is_dir() else "images/valid"
        yaml_dict = {
            "path": str(root.as_posix()),
            "train": "images/train",
            "val": val_rel,
            "nc": len(names),
            "names": names,
        }
        if (root / "images" / "test").is_dir():
            yaml_dict["test"] = "images/test"
    else:
        val_images = find_val_split_root(root) / "images"
        yaml_dict = {
            "train": str((root / "train" / "images").as_posix()),
            "val": str(val_images.as_posix()),
            "nc": len(names),
            "names": names,
        }
        test_dir = root / "test" / "images"
        if test_dir.exists():
            yaml_dict["test"] = str(test_dir.as_posix())

    write_path.parent.mkdir(parents=True, exist_ok=True)
    with open(write_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_dict, f, sort_keys=False)

    return write_path


def resolve_data_yaml(dataset: DatasetConfig, scratch_dir: Path) -> Path:
    """
    Returns the data.yaml path to hand to model.train(), generating one
    from the form if the user didn't supply a raw yaml directly.
    """
    if dataset.use_raw_yaml:
        if not dataset.raw_yaml_path or not Path(dataset.raw_yaml_path).exists():
            raise FileNotFoundError(
                f"data.yaml not found at: {dataset.raw_yaml_path}"
            )
        return Path(dataset.raw_yaml_path)

    generated_path = scratch_dir / "data.yaml"
    return generate_data_yaml(dataset, generated_path)


def _scan_label_class_ids(labels_dir: Path) -> List[int]:
    """Scan all YOLO label .txt files and collect every class id used."""
    ids = set()
    if not labels_dir.exists():
        return []
    for txt_file in labels_dir.glob("*.txt"):
        try:
            with open(txt_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    cls_id = int(line.split()[0])
                    ids.add(cls_id)
        except (ValueError, IndexError):
            continue
    return sorted(ids)


def _labels_dir_for_images_dir(images_dir: Path) -> Path:
    """
    YOLO's convention is a sibling 'labels' folder for every 'images'
    folder (e.g. .../train/images -> .../train/labels). Swap the last
    'images' path segment for 'labels' to find it.
    """
    parts = list(images_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)
    # No 'images' segment found (non-standard layout) — nothing to swap.
    return images_dir


def _validate_raw_yaml(dataset: DatasetConfig) -> ValidationResult:
    """
    Validates a user-supplied data.yaml directly: that it exists, parses,
    declares train/val splits, and that those split paths actually exist
    on disk. This is the path power users take instead of the form, so
    none of the form-driven dataset_root/class-list checks apply here.
    """
    issues: List[ValidationIssue] = []

    if not dataset.raw_yaml_path:
        issues.append(ValidationIssue(level="error", message="No data.yaml path provided."))
        return ValidationResult(ok=False, issues=issues)

    yaml_path = Path(dataset.raw_yaml_path)
    if not yaml_path.exists():
        issues.append(ValidationIssue(
            level="error",
            message=f"data.yaml not found at: {yaml_path}"
        ))
        return ValidationResult(ok=False, issues=issues)

    hint = staging_message(yaml_path.parent)
    if hint:
        issues.append(ValidationIssue(level="warning", message=hint))

    try:
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        issues.append(ValidationIssue(level="error", message=f"Could not parse data.yaml: {e}"))
        return ValidationResult(ok=False, issues=issues)

    if not isinstance(data, dict):
        issues.append(ValidationIssue(level="error", message="data.yaml does not contain a mapping at the top level."))
        return ValidationResult(ok=False, issues=issues)

    train_rel = data.get("train")
    val_rel = data.get("val", data.get("valid"))

    if not train_rel:
        issues.append(ValidationIssue(level="error", message="data.yaml is missing a 'train' key."))
    if not val_rel:
        issues.append(ValidationIssue(level="error", message="data.yaml is missing a 'val' key."))
    if not data.get("nc") and not data.get("names"):
        issues.append(ValidationIssue(level="warning", message="data.yaml has neither 'nc' nor 'names' — class count is unclear."))

    # Split paths in data.yaml are relative to `path` when set, otherwise
    # to the yaml file's own folder (ultralytics convention).
    yaml_path_key = data.get("path")
    if yaml_path_key:
        base = Path(str(yaml_path_key))
        base_dir = base if base.is_absolute() else (yaml_path.parent / base).resolve()
    else:
        base_dir = yaml_path.parent

    def _resolve(rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else (base_dir / p).resolve()

    train_dir = _resolve(train_rel) if train_rel else None
    val_dir = _resolve(val_rel) if val_rel else None

    for label, split_dir in (("train", train_dir), ("val", val_dir)):
        if split_dir is None:
            continue
        if not split_dir.exists():
            issues.append(ValidationIssue(
                level="error",
                message=f"'{label}' path from data.yaml does not exist: {split_dir}"
            ))

    detected_ids: List[int] = []
    if train_dir is not None and train_dir.exists():
        labels_dir = _labels_dir_for_images_dir(train_dir)
        if labels_dir.exists():
            detected_ids = _scan_label_class_ids(labels_dir)
            if not detected_ids:
                issues.append(ValidationIssue(
                    level="warning",
                    message=f"No label class ids found under {labels_dir} — check labels exist."
                ))
        else:
            issues.append(ValidationIssue(
                level="warning",
                message=f"Could not find a labels folder next to {train_dir} (expected {labels_dir})."
            ))

    ok = not any(i.level == "error" for i in issues)
    return ValidationResult(ok=ok, issues=issues, detected_class_ids=detected_ids)


def _validate_form_dataset(dataset: DatasetConfig) -> ValidationResult:
    """
    Checks the dataset folder structure and cross-checks label class ids
    against the entered class list, BEFORE training starts. Only applies
    to the form-driven path (use_raw_yaml=False).
    """
    issues: List[ValidationIssue] = []

    if not dataset.dataset_root or not dataset.dataset_root.strip():
        # Path("") resolves to the server's current working directory,
        # which always exists — so without this explicit check, leaving
        # the field blank silently passes the "root exists" check below
        # and instead fails later with a confusing relative-path error.
        issues.append(ValidationIssue(level="error", message="No dataset root folder provided."))
        return ValidationResult(ok=False, issues=issues)

    root = Path(dataset.dataset_root)

    if not root.exists():
        issues.append(ValidationIssue(
            level="error",
            message=f"Dataset root folder does not exist: {root}"
        ))
        return ValidationResult(ok=False, issues=issues)

    if _ultralytics_layout(root):
        train_img_dir = root / "images" / "train"
        train_lab_dir = root / "labels" / "train"
        val_img_dir = root / "images" / "val"
        if not val_img_dir.is_dir():
            val_img_dir = root / "images" / "valid"
        val_lab_dir = root / "labels" / "val"
        if not val_lab_dir.is_dir():
            val_lab_dir = root / "labels" / "valid"
        required_dirs = [train_img_dir, train_lab_dir, val_img_dir, val_lab_dir]
        missing_val_msg = (
            f"Expected a validation split at {root / 'images' / 'val'} "
            f"or {root / 'images' / 'valid'} (with matching labels/)."
        )
    else:
        train_img_dir = root / "train" / "images"
        train_lab_dir = root / "train" / "labels"
        val_root = find_val_split_root(root)
        val_img_dir = val_root / "images"
        val_lab_dir = val_root / "labels"
        required_dirs = [train_img_dir, train_lab_dir]
        missing_val_msg = (
            f"Expected a validation split at {root / 'valid'} or {root / 'val'} "
            f"(each with images/ and labels/)."
        )

    for d in required_dirs:
        if not d.exists():
            issues.append(ValidationIssue(
                level="error",
                message=f"Expected folder missing: {d}"
            ))

    if not val_img_dir.exists() or not val_lab_dir.exists():
        issues.append(ValidationIssue(level="error", message=missing_val_msg))

    hint = staging_message(root)
    if hint:
        issues.append(ValidationIssue(level="warning", message=hint))

    if any(i.level == "error" for i in issues):
        return ValidationResult(ok=False, issues=issues)

    # Count images vs labels roughly matching
    train_images = list(train_img_dir.glob("*"))
    train_labels = list(train_lab_dir.glob("*.txt"))
    if len(train_images) == 0:
        issues.append(ValidationIssue(level="error", message="No training images found."))
    if len(train_labels) == 0:
        issues.append(ValidationIssue(level="error", message="No training label files found."))
    if abs(len(train_images) - len(train_labels)) > 0 and len(train_labels) > 0:
        issues.append(ValidationIssue(
            level="warning",
            message=(f"Image count ({len(train_images)}) and label count "
                      f"({len(train_labels)}) differ — some images may be unlabeled.")
        ))

    detected_ids = _scan_label_class_ids(train_lab_dir)

    entered_ids = {c.id for c in dataset.classes}
    if entered_ids:
        extra_ids = [i for i in detected_ids if i not in entered_ids]
        if extra_ids:
            max_entered = max(entered_ids)
            issues.append(ValidationIssue(
                level="error",
                message=(f"Label files use class id(s) {extra_ids} that aren't in "
                          f"your class list (you defined ids 0-{max_entered}). "
                          f"Add these classes or fix your labels before training.")
            ))

        unused_ids = [i for i in entered_ids if i not in detected_ids]
        if unused_ids:
            issues.append(ValidationIssue(
                level="warning",
                message=(f"Class id(s) {unused_ids} are defined but never appear in "
                          f"training labels — check these classes have examples.")
            ))

    ok = not any(i.level == "error" for i in issues)
    return ValidationResult(ok=ok, issues=issues, detected_class_ids=detected_ids)


def validate_dataset(dataset: DatasetConfig) -> ValidationResult:
    """
    Dispatches to the right validation path. Raw-yaml mode validates the
    yaml file and the splits it points to; form mode validates the
    dataset_root folder structure and cross-checks the class list. These
    are different inputs with different failure modes, so they can't
    share one set of folder checks (this used to always require
    dataset_root/train/valid even in raw-yaml mode, which is wrong).
    """
    if dataset.use_raw_yaml:
        return _validate_raw_yaml(dataset)
    return _validate_form_dataset(dataset)


def _count_images(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    n = 0
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
            n += 1
    return n


def _names_from_yaml_or_classes(root: Path, yaml_data: dict | None) -> dict:
    names = {}
    if yaml_data and isinstance(yaml_data.get("names"), dict):
        names = {int(k): str(v) for k, v in yaml_data["names"].items()}
    elif yaml_data and isinstance(yaml_data.get("names"), list):
        names = {i: str(v) for i, v in enumerate(yaml_data["names"])}
    classes_txt = root / "classes.txt"
    if not names and classes_txt.is_file():
        for i, line in enumerate(classes_txt.read_text(encoding="utf-8").splitlines()):
            if line.strip():
                names[i] = line.strip()
    return names


def inspect_dataset_root(root: str | Path) -> dict:
    """Describe a folder so Train can use Segriotate output or an external split."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        return {"ok": False, "error": f"not a folder: {root}"}

    yaml_path = root / "data.yaml"
    yaml_data = None
    if yaml_path.is_file():
        try:
            yaml_data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as e:
            return {"ok": False, "error": f"could not read data.yaml: {e}"}

    layout = None
    counts = {"train": 0, "val": 0, "test": 0}
    if (root / "images" / "train").is_dir():
        layout = "ultralytics"
        counts["train"] = _count_images(root / "images" / "train")
        counts["val"] = _count_images(root / "images" / "val") or _count_images(root / "images" / "valid")
        counts["test"] = _count_images(root / "images" / "test")
    elif (root / "train" / "images").is_dir():
        layout = "roboflow"
        counts["train"] = _count_images(root / "train" / "images")
        val_root = find_val_split_root(root)
        counts["val"] = _count_images(val_root / "images")
        counts["test"] = _count_images(root / "test" / "images")
    elif yaml_path.is_file():
        layout = "yaml"
    else:
        return {
            "ok": False,
            "error": "Need data.yaml, or images/train + labels/train, or train/images + train/labels",
            "root": str(root),
        }

    names = _names_from_yaml_or_classes(root, yaml_data if isinstance(yaml_data, dict) else None)
    yaml_str = str(yaml_path) if yaml_path.is_file() else None
    return {
        "ok": True,
        "root": str(root),
        "layout": layout,
        "yaml": yaml_str,
        "use_raw_yaml": bool(yaml_str),
        "counts": counts,
        "names": {str(k): v for k, v in sorted(names.items())},
        "labelled": sum(counts.values()),
    }