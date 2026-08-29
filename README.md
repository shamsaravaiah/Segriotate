# SegriLabs

<p align="center">
  <img src="Segriotate.app/Contents/Resources/segriotate_icon_1024.png" alt="Segri-Labs icon" width="160">
</p>

A local desktop app for **YOLO segmentation labelling** and **model training**.

Open a folder of images, draw or auto-detect masks, export a train / val / test
dataset, then train (or fine-tune) a YOLO-seg model on that dataset. Labels and
weights stay on disk — nothing is uploaded.

Home has two labs:

- **Annotate** — label images and export a split dataset
- **Train** — pick weights, point at a dataset, run training

Everything runs on `127.0.0.1`. Images never leave the machine.

## What it does

### Annotate

- **Auto-Detect** — runs your YOLO segmentation model (`segmentation.pt`) on
  each image as soon as it loads, and pre-fills polygons.
- **Click-to-Segment** — point-prompt fallback (FastSAM / MobileSAM / any
  other `.pt` or TensorRT `.engine` you drop in) for fruit the model missed.
- **Manual polygons** — draw by hand when neither model is right (`N`).
- **Grading classes** — assign one or more classes per fruit (`0–9`,
  `Shift+0–9` to add a second class). Types such as EXP / DOM / RET are
  UI groups only; YOLO files store subclass ids `0–9`.
- **Save** — writes YOLO segmentation `.txt` to a paired labels folder under
  `workspace/labels/`. **Export ZIP** packs the current labels for backup or CVAT.
- **Dataset split** — copies only images that have at least one mask into
  `images/train`, `images/val`, `images/test` (and matching `labels/`) plus
  `data.yaml`. Empty photos are skipped.

### Train

- **Base model** — start from Annotate weights in `models/dot-pt/`, a previous
  run under `workspace/train/runs/`, or any `.pt` you browse to.
- **Dataset** — **Use last from Annotate**, pick another folder with
  `images/train` + `labels/train` (or `train/images` + `train/labels`), or
  point at an existing `data.yaml`.
- **Hyperparameters** — epochs, image size, batch, device (CUDA / Apple MPS /
  CPU), optimizer, learning rate, patience, seed.
- **Augmentation** — defaults assume a fixed camera (geometry off). Optional
  glare / HSV / mosaic for lighting.
- **Jobs** — start, watch logs, stop. One job at a time. Weights land in
  `workspace/train/runs/<run-name>/weights/` (`best.pt`, `last.pt`).

## How it works

```
python desktop_app.py   (or double-click Segriotate.app)
     |
     v
local Flask server starts on http://127.0.0.1:8765
     |
     v
Home  →  Annotate  or  Train
     |
     +-- Annotate
     |     Open Images…  →  labels saved to workspace/labels/<folder>_labels/
     |     Auto-Detect / Click-to-Segment / draw (N)
     |     assign class (0–9)  →  Save (S)
     |     Create train / val / test  →  folder with data.yaml
     |
     +-- Train
           pick .pt  →  pick dataset (last Annotate split or another folder)
           set epochs / device  →  Start
           weights → workspace/train/runs/<run-name>/weights/best.pt
```

`desktop_app.py` is a PyQt window around the HTML labs. It starts
`scripts/segment_server.py` in the background:

| Endpoint | Role |
|---|---|
| `/` | Home (Annotate or Train) |
| `/annotate` | label editor |
| `/train` | training UI |
| `/detect` | YOLO `segmentation.pt` on the current image |
| `/segment` | click-to-segment with the selected `.pt` / `.engine` |
| `/label` | read/write YOLO `.txt` in the labels folder |
| `/media` | serve images from the folder you opened |
| `/project/split-dataset` | copy labelled images into train/val/test |
| `/project/train/*` | start / poll / stop training jobs |

Click models load **lazily** on first click, from `models/dot-pt/` or
`models/dot-engine/`. `segmentation.pt` is the Auto-Detect model only; it is
not listed as a click model.

A batch alternative (`02_generate_labels.py` + CVAT, then `04_train.py`) is
included if you want a full-folder pass from the terminal instead of the UI.

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Then put weights in place (next section) and start:

```bash
source venv/bin/activate
python desktop_app.py
```

Or double-click **`Segriotate.app`** (macOS) in this folder. First launch can
take a minute while `segmentation.pt` loads. Quit with Cmd+Q — that also
stops the local server.

Browser instead of the desktop window:

```bash
python scripts/segment_server.py
```

then open http://127.0.0.1:8765

## 2. Models — where they go and how to download them

Git does **not** include weights (they are gitignored). On first launch the
app downloads FastSAM and MobileSAM into `models/dot-pt/` in the background
while a spinner shows **Launching app. Please wait**. Files that already
exist are skipped.

```
models/
  source/          ← optional drop folder (copied into dot-pt)
  dot-pt/          ← PyTorch .pt files the editor loads (Mac, PC, Orin)
    segmentation.pt
    FastSAM-s.pt
    FastSAM-x.pt      (optional, larger / slower)
    mobile_sam.pt     (optional)
  dot-engine/      ← TensorRT .engine files built from every .pt in dot-pt
```

### Auto-Detect (required for pre-filled polygons)

| File | Where | What it is |
|---|---|---|
| `models/dot-pt/segmentation.pt` | **your** trained YOLO-seg fruit model | Not a public download. Copy your `.pt` here, or set `SEGMENTATION_DOWNLOAD_URL` in `config.py`. |

The editor still opens without this file; Auto-Detect stays empty until you add it.

### Click-to-Segment (auto-downloaded)

These are fetched on first launch from Ultralytics if they are missing:

| File | Size | Download |
|---|---|---|
| `FastSAM-s.pt` | ~24 MB | https://github.com/ultralytics/assets/releases/download/v8.4.0/FastSAM-s.pt |
| `FastSAM-x.pt` | ~140 MB | https://github.com/ultralytics/assets/releases/download/v8.4.0/FastSAM-x.pt |
| `mobile_sam.pt` | ~40 MB | https://github.com/ultralytics/assets/releases/download/v8.4.0/mobile_sam.pt |

`FastSAM-s.pt` is the usual default (small and fast). `FastSAM-x.pt` is more
accurate and heavier. `mobile_sam.pt` is an alternative click model.

```bash
cd models/dot-pt

curl -L -O https://github.com/ultralytics/assets/releases/download/v8.4.0/FastSAM-s.pt
curl -L -O https://github.com/ultralytics/assets/releases/download/v8.4.0/FastSAM-x.pt
curl -L -O https://github.com/ultralytics/assets/releases/download/v8.4.0/mobile_sam.pt
```

Docs: [FastSAM](https://docs.ultralytics.com/models/fast-sam/),
[MobileSAM](https://docs.ultralytics.com/models/mobile-sam/). Original FastSAM
paper weights: [CASIA-IVA-Lab/FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM).

In Annotate, pick **Format** (`.pt` or `.engine`) then **Model**. Click
models are whatever files are in those folders; extra `.pt` files you add
will show up too.

### TensorRT (built on the deployed machine)

On startup, if TensorRT is installed (Orin / an NVIDIA PC), the app builds
an engine for **every** `.pt` in `models/dot-pt/`:

`models/dot-pt/foo.pt` → `models/dot-engine/foo.engine`

It skips an engine that already exists and is newer than the `.pt`. A failed
build of one file (common for FastSAM / MobileSAM; they are not YOLO export)
does not stop the editor. Mac and PCs without TensorRT skip the build and
keep using `.pt`.

Do not copy `.engine` files between machines. First engine build can take
several minutes per model.

### Training start weights

Train can start from `segmentation.pt`, any other `.pt` in `models/dot-pt/`,
or a previous run. Ultralytics nano-seg is optional if you want a public
starting point instead of your fruit model:

| File | Download |
|---|---|
| `yolo11n-seg.pt` | https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt |

The terminal script `scripts/04_train.py` still uses `TRAIN_BASE_MODEL` in
`config.py` (default `yolo11n-seg.pt`).

## 3. Annotate

1. Start the app. From **Home**, open **Annotate**.
2. **Open Images…** and pick a folder (for example `batch001`).
   Labels go to `workspace/labels/batch001_labels/` automatically. Use
   **Labels Folder…** only to override that.
3. Classes lock after images open so ids cannot change mid-batch. Set up
   types and class names (or load a profile) **before** opening images.
4. For each image:
   - **Auto-Detect: ON** (default) — your model pre-fills every object it
     finds as the image loads.
   - Fix a polygon: drag a vertex, double-click an edge to insert a point,
     right-click a vertex to delete it.
   - Press **1–9** or **0** to set the grading class on the selected
     polygon. **Shift+0–9** adds an extra class on the same fruit.
   - If a fruit was missed: keep **Click-to-Segment** on, click the fruit,
     then assign a class.
   - **N** draws a polygon by hand. **Enter** finishes it, **Esc** cancels.
   - **S** saves, **← / →** move to prev/next (auto-saving if Auto-Save is on).
5. In the **Dataset** panel, choose an output folder, set train / val / test
   percentages, and click **Create train / val / test**. Only images with at
   least one mask are copied. That folder is remembered for Train.

Jump to an image by clicking the number in the header, typing, and pressing
Enter.

### Where labels sit

| What | Path |
|---|---|
| Live YOLO `.txt` for folder `batch001` | `workspace/labels/batch001_labels/` |
| Class profile catalog | `workspace/profiles/class_profiles.json` |
| Sidecars in the labels folder | `class_profile.txt`, `classes.txt` |
| Logs | `workspace/logs/segriotate.log` |

Images stay in the folder you opened. Each image `photo.jpg` gets
`photo.txt` with the same stem. Older `labels/<folder>_labels/` folders at
the project root are still reused if they exist.

### Shortcuts

| Action | Key |
|---|---|
| Click fruit to segment | click |
| New polygon | `N` |
| Finish polygon | `Enter` |
| Cancel / deselect | `Esc` |
| Delete polygon | `Del` |
| Replace class 0–9 | `0…9` |
| Add extra class | `Shift+0…9` |
| Insert vertex | double-click edge |
| Delete vertex | right-click vertex |
| Pan | space + drag |
| Zoom | scroll |
| Prev / next image | `←` `→` |
| Save | `S` |
| Undo | `Ctrl+Z` / `Cmd+Z` |

## 4. Train

1. From **Home**, open **Train** (or use the header link).
2. **Base Model** — pick weights from the Annotate list (`models/dot-pt/`)
   or a previous **Trained runs** checkpoint. Or paste / browse a `.pt`.
3. **Dataset** — **Use last from Annotate**, or choose a folder that already
   has a YOLO layout (`images/train` + `labels/train`, or `train/images` +
   `train/labels`). Optionally tick **Use an existing data.yaml**.
4. **Hyperparameters** — epochs (default 20), image size 640, batch, workers
   (forced to 0 on Windows), device, optimizer, `lr0` / `lrf`, patience, seed.
   **Project folder** defaults to `workspace/train/runs`. **Run name** is the
   subfolder (default `run`).
5. **Augmentation** — leave geometric values at 0 for a fixed camera. Enable
   glare aug if lighting varies.
6. Review the summary, then **Start**. Watch the console. **Stop** kills the
   current job.

Output:

```
workspace/train/runs/<run-name>/
  weights/best.pt
  weights/last.pt
```

Those `.pt` files appear under **Trained runs** the next time you open Train,
so you can fine-tune again. `workspace/train/` is gitignored.

You can save the current form as a named preset and load it later.

## 5. Edit `config.py`

At minimum, check/set:

- `MIN_CONFIDENCE` / `AUTO_ACCEPT_CONFIDENCE` — how strict auto-accept is
  (used by the batch script; the live editor uses `MIN_CONFIDENCE` for detect)
- `FORCE_CLASS_ID` or `CLASS_MAP` — how old class ids map to your new classes
- `CLASS_NAMES` — names written into `data.yaml` by `03_split_dataset.py`
- `CLASS_PROFILES_PATH` — class profiles (`workspace/profiles/class_profiles.json`)
- `TRAIN_RUNS_DIR` — where Train writes weights (`workspace/train/runs`)

### Class profiles

The class list in Annotate's left panel can be saved as a named profile in
`workspace/profiles/class_profiles.json`. **Load profile** at the top of
Classes picks a saved list. After you build types and classes, type a name
under **Save this list as a profile** and click **Save profile** — that name
appears in the dropdown next to Default and is still there after you relaunch.

The file is plain JSON — back it up, commit it, or copy it to another machine.
A copy of an older `class_profiles.json` in the project root is used once if
the workspace file is missing.

When a labels folder is open, the editor also writes two sidecars there (YOLO
training ignores them because they do not match an image name):

- `class_profile.txt` — the profile name as shown in the app (`Default`, `test`, …)
- `classes.txt` — one class name per line; line 1 is id `0`, line 2 is id `1`, …

> Class ids are positions in the list. If you replace the class list after
> labelling images, ids get reused and existing `.txt` files will read back
> under the new names. Start a new labels folder when you change taxonomy.

Default subclasses (ids written to YOLO `.txt`):

| Id | Name | Type (UI group) |
|---|---|---|
| 0 | Export Premium | Export Quality (EXP) |
| 1 | Prime Export | Export Quality (EXP) |
| 2 | Domestic Premium | Domestic Premium (DOM) |
| 3 | Prime Retail | Domestic Premium (DOM) |
| 4 | Standard Retail | Retail Standard (RET) |
| 5 | Everyday Retail | Retail Standard (RET) |
| 6 | Commercial | Commercial (COM) |
| 7 | Value | Commercial (COM) |
| 8 | Processing | Non-Market (NMR) |
| 9 | Reject | Non-Market (NMR) |

## 6. Optional batch pipeline

Use this if you want a full-folder pass and CVAT review instead of live
grading, or to train from the terminal instead of the Train lab.

```bash
python scripts/00_diagnose.py          # check the .pt is a seg model
python scripts/01_test_model.py --n 10 # sanity check on a handful of images
python scripts/02_generate_labels.py   # auto vs review buckets
python scripts/03_split_dataset.py     # train/val/test from labels/
python scripts/04_train.py             # fine-tune TRAIN_BASE_MODEL
```

`02_generate_labels.py` writes:

- `output/labels_auto/` — high confidence, ready to train
- `review/no_detection/` — model found nothing
- `review/low_confidence/` — at least one instance below `AUTO_ACCEPT_CONFIDENCE`
- `review/multiple_objects/` — crowded scenes (`MAX_INSTANCES_BEFORE_REVIEW`)

### CVAT

Better if several people review, or you want CVAT tools (magnetic lasso,
interpolation, etc). Use with the batch script, not the live editor:

1. Run `python scripts/02_generate_labels.py`.
2. In CVAT, create a task and upload `review/<bucket>/images/`.
3. Import **YOLO 1.1 Segmentation**, pointing at `review/<bucket>/labels/`
   (plus an `obj.names` file listing `CLASS_NAMES` in order).
4. Correct the masks, export as **YOLO 1.1 Segmentation**.
5. Copy the corrected `.txt` files into `output/labels_auto/`.

Then run `03_split_dataset.py`.

`03_split_dataset.py` looks for `.txt` files in `labels/` first, then falls
back to `output/labels_auto/`. If you labelled in Annotate, copy or point it
at `workspace/labels/<folder>_labels/` as needed.

## 7. Notes

- If your existing model's masks are roughly 90%+ accurate, this saves most
  of the annotation time. If they are well under 70%, it is usually faster
  to label a smaller seed set by hand, retrain in **Train**, drop `best.pt`
  in as Auto-Detect, and continue.
- `CLASS_MAP` drops any class id not listed — add an identity entry
  (e.g. `4: 4`) for classes you want to keep unchanged.
- `.pt` files run on Mac / PC / Orin. `.engine` files are built on the
  deployed GPU from `models/dot-pt/`; do not copy an engine from another
  machine.
- No data leaves your machine unless you self-host CVAT elsewhere.
