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

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "models" / "dot-pt" / "segmentation.pt"   # your existing YOLO-seg model
MODEL_PT_DIR = PROJECT_ROOT / "models" / "dot-pt"
MODEL_ENGINE_DIR = PROJECT_ROOT / "models" / "dot-engine"
MODEL_SOURCE_DIR = PROJECT_ROOT / "models" / "source"        # drop new .pt here; engines built on device
IMAGE_DIR = PROJECT_ROOT / "images"                          # the 10,000 raw images
LABEL_DIR = PROJECT_ROOT / "labels"                          # YOLO .txt annotations (editor writes here)

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

# TensorRT engines are built on the deployed machine from models/source/*.pt.
# Mac/PC without TensorRT skip this and keep using .pt files.
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
