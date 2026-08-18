# Fruit Grading — Segriotate

Uses your trained YOLO segmentation model to pre-label images live as you
grade them, with a FastSAM click-fallback for anything the model misses.

## Desktop app (recommended)

After setup (section 1 below), start Segriotate without opening a browser:

```bash
source venv/bin/activate
python desktop_app.py
```

Or double-click **`Segriotate.app`** (macOS) in this folder. First launch can
take a minute while models load. Put images in `images/` (they load
automatically) or use **Open Images…**. Labels are written to `labels/`.

Quit with Cmd+Q — that also stops the local server.

```
python desktop_app.py   (or double-click Segriotate.app)
     |
     v
your model runs automatically (Auto-Detect)  -->  polygons pre-filled
     |
     +--> model found it        -->  fix class / nudge points if needed
     |
     +--> model missed it       -->  Click-to-Segment (FastSAM) on that fruit
     |
     +--> still not right       -->  draw polygon by hand (N key)
     |
     v
assign grading class (1-9, 0)
     |
     v
Save (S) -> writes YOLO .txt to labels/
     |
     v
next image ...
     |
     v
train/val/test split (03_split_dataset.py)
     |
     v
train final model (04_train.py)
```

A batch alternative (`02_generate_labels.py` + CVAT) is also included if you
ever want to process all images upfront instead of live, one at a time —
see Option: CVAT below.

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Put your files in place:

- `models/segmentation.pt` — your existing YOLO segmentation model
- `images/` — all 10,000 source images (subfolders are fine)

## 2. Edit `config.py`

At minimum, check/set:

- `MIN_CONFIDENCE` / `AUTO_ACCEPT_CONFIDENCE` — how strict auto-accept is
- `FORCE_CLASS_ID` or `CLASS_MAP` — how old classes map to your new classes
- `CLASS_NAMES` — the final class names for training

## 3. Run the pipeline

```bash
# sanity check on a handful of images first
python scripts/01_test_model.py --n 10

# run over all 10,000 images, split into auto vs review
python scripts/02_generate_labels.py

# fix the review buckets in CVAT (see below), then:
python scripts/03_split_dataset.py

# train
python scripts/04_train.py
```

## 4. Reviewing / correcting labels

### Desktop app (recommended)

```bash
python desktop_app.py
```

or double-click `Segriotate.app`. Images in `images/` load automatically.
Saves go to `labels/`.

### Browser (optional)

`python scripts/segment_server.py` then open http://127.0.0.1:8765
(or `tools/label_editor.html` in Chrome against that server).

1. Start the desktop app (or the server + browser). The first run loads your
   `segmentation.pt` plus FastSAM. FastSAM weights (~140MB) download once.
2. **Open Images…** if you are not using the project `images/` folder.
3. For each image:
   - **Auto-Detect: ON** (default) — your model runs automatically and
     pre-fills every object it finds, as soon as the image loads.
   - Fix any pre-filled polygon: drag a vertex to move it, double-click an
     edge to insert a point, right-click a vertex to delete it.
   - Press **1–9** or **0** to assign the correct grading class to a
     selected polygon (see the class list in the left panel).
   - **If your model misses a fruit entirely** (common across varieties it
     wasn't trained on): turn on **Click-to-Segment**, click directly on
     that fruit, and FastSAM generates a polygon for it from that single
     point — then assign it a class as normal.
   - Press **N** to draw a polygon completely by hand instead, if neither
     auto-detect nor click-to-segment gets it right.
   - **S** saves the current image, **← / →** move to prev/next (auto-saving
     as you go).

This is the primary workflow — no separate batch step, no CVAT round-trip.
`scripts/02_generate_labels.py` is still there if you ever want a full
10,000-image batch pass instead (e.g. for a quick bulk confidence check),
but for grading-by-hand as described, the live editor above is simpler.

### Option: CVAT

Better if multiple people need to review, or you want CVAT's more powerful
editing tools (magnetic lasso, interpolation, etc). Use with the batch
script instead of the live server:

1. Run `python scripts/02_generate_labels.py` first — it sorts images into
   `output/labels_auto/` (high confidence) and `review/<bucket>/` folders.
2. In CVAT, create a task and upload `review/<bucket>/images/`.
3. Import annotations using the **YOLO 1.1 Segmentation** format, pointing
   at `review/<bucket>/labels/` (plus an `obj.names` file listing
   `CLASS_NAMES` in order — CVAT's import wizard will prompt for this).
4. Correct the masks, then export the task as **YOLO 1.1 Segmentation**.
5. Copy the corrected `.txt` files into `output/labels_auto/`.

Once merged back into `output/labels_auto/`, continue with step 3 above
(`03_split_dataset.py`).

## 5. Notes

- If your existing model's masks are roughly 90%+ accurate, this approach
  saves the vast majority of manual annotation time. If they're poor
  (well under 70%), it's usually faster to label a smaller seed set by hand
  in CVAT, retrain, and re-run this pipeline with the improved model.
- `CLASS_MAP` drops any class id not explicitly listed — add an identity
  entry (e.g. `4: 4`) for classes you want to keep unchanged.
- Everything here runs locally; no data leaves your machine unless you
  self-host CVAT elsewhere.
# Segriotate
