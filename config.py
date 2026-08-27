"""
Central configuration for the auto-labelling pipeline.
Edit the values below, then run the scripts in scripts/ in order:

    01_test_model.py       -> sanity check your model on a handful of images
    02_generate_labels.py  -> run the model over all images, split into
                               auto-accepted vs needs-review, write YOLO .txt
    03_split_dataset.py    -> build train/val/test folders from the
                               auto-accepted labels
    04_train.py             -> train (or fine-tune) a YOLO-seg model
"""

import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

# App-created files live here so they stay out of the code tree.
# Existing labels/ and class_profiles.json at the project root are still read.
WORKSPACE = PROJECT_ROOT / "workspace"
LABELS_ROOT = WORKSPACE / "labels"
LEGACY_LABELS_ROOT = PROJECT_ROOT / "labels"
PROFILES_DIR = WORKSPACE / "profiles"
LOGS_DIR = WORKSPACE / "logs"
LOG_PATH = LOGS_DIR / "segriotate.log"
TRAIN_ROOT = WORKSPACE / "train"
TRAIN_RUNS_DIR = TRAIN_ROOT / "runs"
TRAIN_PRESETS_DIR = TRAIN_ROOT / "presets"
TRAIN_SCRATCH_DIR = TRAIN_ROOT / "job_scratch"
TRAIN_CACHE_DIR = TRAIN_ROOT / "cache"

MODEL_PATH = PROJECT_ROOT / "models" / "dot-pt" / "segmentation.pt"   # your existing YOLO-seg model
MODEL_PT_DIR = PROJECT_ROOT / "models" / "dot-pt"
MODEL_ENGINE_DIR = PROJECT_ROOT / "models" / "dot-engine"
MODEL_SOURCE_DIR = PROJECT_ROOT / "models" / "source"        # optional drop folder; copied into dot-pt
IMAGE_DIR = PROJECT_ROOT / "images"                          # the 10,000 raw images
LABEL_DIR = LABELS_ROOT                                      # YOLO .txt annotations (editor writes here)
LEGACY_CLASS_PROFILES_PATH = PROJECT_ROOT / "class_profiles.json"

CLICK_MODEL_LABELS = {
    "FastSAM-s": "FastSAM",
    "FastSAM-x": "FastSAM-x",
    "mobile_sam": "MobileSAM",
}
ANNOTATE_PT_LABELS = {
    "segmentation": "Auto-Detect (segmentation)",
    **CLICK_MODEL_LABELS,
}

# Click-to-segment weights. Missing files are downloaded on first launch.
MODEL_DOWNLOADS = {
    "FastSAM-s.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/FastSAM-s.pt",
    "FastSAM-x.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/FastSAM-x.pt",
    "mobile_sam.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/mobile_sam.pt",
}

# Optional URL that copies your fruit YOLO-seg weights to MODEL_PATH.
# Leave empty if you place segmentation.pt in models/dot-pt/ yourself.
SEGMENTATION_DOWNLOAD_URL = ""

OUTPUT_DIR = PROJECT_ROOT / "output"
AUTO_LABEL_DIR = OUTPUT_DIR / "labels_auto"                  # high-confidence, ready to train on
REVIEW_DIR = PROJECT_ROOT / "review"                         # everything a human should check in CVAT

DATASET_DIR = PROJECT_ROOT / "dataset"                       # final train/val/test split

SERVER_PORT = 8765                                           # local UI + inference server

# TensorRT engines are built on the deployed machine from models/dot-pt/*.pt
# into models/dot-engine/. Mac/PC without TensorRT skip this and keep using .pt.
ENGINE_BATCH = 1          # live editor sends one image
ENGINE_IMGSZ = 640
ENGINE_WORKSPACE_GB = 4   # raise on a desktop GPU if the build fails
ENGINE_USE_FP16 = True
ENGINE_USE_SPARSE = True

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

# Minimum box confidence to even keep a detection at all.
MIN_CONFIDENCE = 0.25

# Per-instance confidence at/above which a detection is auto-accepted
# without human review. Anything below this (but above MIN_CONFIDENCE)
# is still written to the review set.
AUTO_ACCEPT_CONFIDENCE = 0.70

# If an image has more than this many detected instances, route the whole
# image to review even if every instance was individually confident
# (crowded scenes are the most error-prone for auto-masks).
MAX_INSTANCES_BEFORE_REVIEW = 15

# ---------------------------------------------------------------------------
# Class handling
# ---------------------------------------------------------------------------

# Named class profiles saved from the editor's left panel. Plain JSON, so it
# can be backed up, committed, or copied to another machine.
CLASS_PROFILES_PATH = PROFILES_DIR / "class_profiles.json"


def _move_legacy_train_dir(legacy: Path, dest: Path) -> None:
    """Move leftover workspace/<name> into workspace/train/<name> once."""
    try:
        if not legacy.exists() or legacy.resolve() == dest.resolve():
            return
        if dest.exists():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(legacy), str(dest))
    except OSError:
        pass


def ensure_workspace() -> None:
    """Create workspace folders and copy the old profiles file once if needed."""
    LABELS_ROOT.mkdir(parents=True, exist_ok=True)
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_ROOT.mkdir(parents=True, exist_ok=True)
    _move_legacy_train_dir(WORKSPACE / "runs", TRAIN_RUNS_DIR)
    _move_legacy_train_dir(WORKSPACE / "job_scratch", TRAIN_SCRATCH_DIR)
    _move_legacy_train_dir(WORKSPACE / "presets", TRAIN_PRESETS_DIR)
    _move_legacy_train_dir(WORKSPACE / "cache", TRAIN_CACHE_DIR)
    TRAIN_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    (TRAIN_CACHE_DIR / "datasets").mkdir(parents=True, exist_ok=True)
    if not CLASS_PROFILES_PATH.is_file() and LEGACY_CLASS_PROFILES_PATH.is_file():
        shutil.copy2(LEGACY_CLASS_PROFILES_PATH, CLASS_PROFILES_PATH)


ensure_workspace()

# Option A: collapse every detection into a single class (e.g. your existing
# model is only being used as a generic "mask generator"). Set to an int to
# use this, or None to disable.
FORCE_CLASS_ID = None

# Option B: remap the existing model's class ids to the ids your new
# dataset should use. Leave empty ({}) to keep the original class ids.
# Any class id NOT present in this map is dropped (its detections are
# skipped entirely) -- add an identity entry (e.g. 4: 4) to keep a class
# unchanged.
CLASS_MAP = {}

# Final class names for data.yaml, indexed by subclass id.
# Class types (EXP/DOM/RET/COM/NMR) are UI groups only and are not written
# to YOLO .txt files. Subclass ids start at 0:
#   Export Quality (EXP)   -> 0 Export Premium, 1 Prime Export
#   Domestic Premium (DOM) -> 2 Domestic Premium, 3 Prime Retail
#   Retail Standard (RET)  -> 4 Standard Retail, 5 Everyday Retail
#   Commercial (COM)       -> 6 Commercial, 7 Value
#   Non-Market (NMR)       -> 8 Processing, 9 Reject
CLASS_NAMES = {
    0: "Export Premium",
    1: "Prime Export",
    2: "Domestic Premium",
    3: "Prime Retail",
    4: "Standard Retail",
    5: "Everyday Retail",
    6: "Commercial",
    7: "Value",
    8: "Processing",
    9: "Reject",
}

# ---------------------------------------------------------------------------
# Dataset split
# ---------------------------------------------------------------------------

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
SPLIT_SEED = 42

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

TRAIN_BASE_MODEL = "yolo11n-seg.pt"   # starting weights for training/fine-tuning
TRAIN_EPOCHS = 100
TRAIN_IMGSZ = 640
TRAIN_BATCH = 16
TRAIN_PROJECT = "runs"
TRAIN_NAME = "my_segmentation"
