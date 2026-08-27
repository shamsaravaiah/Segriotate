"""
Runs as a SEPARATE PROCESS per training job (launched by job_manager.py).

Running training in its own process (rather than a thread inside the
FastAPI server) means:
  - A crash in ultralytics/CUDA can't take down the web server.
  - Stopping a job is a clean process kill, not fighting with Python's GIL.
  - stdout naturally streams line-by-line for live log display.

Usage: python training_runner.py <config_json_path>
"""
import multiprocessing
import sys
import json
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from schemas import TrainConfig
from dataset_utils import resolve_data_yaml
from dataset_stage import stage_dataset_for_training
from app_paths import default_runs_dir


def _resolve_device(requested: str) -> str:
    try:
        import torch
    except Exception:
        print("[training_runner] PyTorch not available, using CPU.", flush=True)
        return "cpu"

    req = (requested or "cpu").strip()
    if req.lower() == "cpu":
        return "cpu"
    if req.lower() == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        print("[training_runner] MPS not available, falling back to CPU.", flush=True)
        return "cpu"
    if torch.cuda.is_available():
        return req
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("[training_runner] CUDA not available, using MPS.", flush=True)
        return "mps"
    print("[training_runner] Requested GPU not available, falling back to CPU.", flush=True)
    return "cpu"


def _resolve_project(project: str) -> str:
    from dataset_stage import needs_local_stage

    raw = (project or "").strip()
    dest_default = default_runs_dir()
    dest_default.mkdir(parents=True, exist_ok=True)
    if not raw or raw == "runs":
        return str(dest_default)
    path = Path(raw)
    if path.is_absolute():
        if not needs_local_stage(path):
            path.mkdir(parents=True, exist_ok=True)
            return str(path)
        print(
            f"[training_runner] Project folder {path} is not on local disk; writing runs to {dest_default}",
            flush=True,
        )
        return str(dest_default)
    dest = dest_default if raw == "runs" else (dest_default / raw)
    dest.mkdir(parents=True, exist_ok=True)
    return str(dest)


def _resolve_workers(requested: int) -> int:
    if sys.platform == "win32":
        if requested and requested > 0:
            print("[training_runner] Forcing workers=0 on Windows (dataloader workers deadlock).", flush=True)
        return 0
    return requested


def main():
    if len(sys.argv) < 2:
        print("ERROR: missing config path argument", flush=True)
        sys.exit(1)

    config_path = Path(sys.argv[1])
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    cfg = TrainConfig(**raw)

    print("[training_runner] Staging dataset onto local disk if needed...", flush=True)
    try:
        cfg.dataset = stage_dataset_for_training(cfg.dataset, log=print)
    except Exception as e:
        print(f"ERROR: Could not copy dataset to local disk: {e}", flush=True)
        sys.exit(1)

    scratch_dir = config_path.parent
    try:
        data_yaml_path = resolve_data_yaml(cfg.dataset, scratch_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", flush=True)
        sys.exit(1)

    device = _resolve_device(cfg.device)
    workers = _resolve_workers(cfg.workers)
    project = _resolve_project(cfg.project)

    print(f"[training_runner] Using data.yaml: {data_yaml_path}", flush=True)
    print(f"[training_runner] Base model: {cfg.base_model_path}", flush=True)
    print(f"[training_runner] Device: {device}  workers: {workers}  project: {project}", flush=True)

    # Imported here (not top of file) so config/validation errors above
    # surface fast without waiting on the heavy torch/ultralytics import.
    from ultralytics import YOLO

    if cfg.resume_from:
        model = YOLO(cfg.resume_from)
    else:
        model = YOLO(cfg.base_model_path)

    # Emit machine-readable progress lines the job manager can parse,
    # instead of trying to scrape ultralytics' human-formatted output
    # (which changes across versions/verbosity, and whose tqdm progress
    # bar gets suppressed entirely when stdout isn't a real terminal —
    # exactly the case here, since we pipe it. Without this, the console
    # can go completely silent for the whole first epoch, which on CPU
    # with workers=0 can legitimately take many minutes and looks
    # indistinguishable from a hang).
    batch_state = {"count": 0, "total": None}

    def _on_pretrain_start(trainer):
        print(
            "[training_runner] Scanning/caching dataset. "
            "This can take a few minutes — it is not frozen.",
            flush=True,
        )

    def _on_epoch_start(trainer):
        batch_state["count"] = 0
        try:
            batch_state["total"] = len(trainer.train_loader)
        except Exception:
            batch_state["total"] = None

    def _on_batch_end(trainer):
        batch_state["count"] += 1
        epoch = trainer.epoch + 1  # trainer.epoch is 0-indexed
        total_epochs = trainer.epochs
        total_batches = batch_state["total"]
        print(
            f"[training_runner] BATCH epoch={epoch} total_epochs={total_epochs} "
            f"batch={batch_state['count']} total_batches={total_batches or '?'}",
            flush=True,
        )

    def _report_progress(trainer):
        epoch = trainer.epoch + 1
        total = trainer.epochs
        percent = round(100.0 * epoch / total, 1) if total else 0.0
        print(
            f"[training_runner] PROGRESS epoch={epoch} total_epochs={total} percent={percent}",
            flush=True,
        )

    model.add_callback("on_pretrain_routine_start", _on_pretrain_start)
    model.add_callback("on_train_epoch_start", _on_epoch_start)
    model.add_callback("on_train_batch_end", _on_batch_end)
    model.add_callback("on_train_epoch_end", _report_progress)

    train_kwargs = dict(
        data=str(data_yaml_path),
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        workers=workers,
        device=device,
        lr0=cfg.lr0,
        lrf=cfg.lrf,
        patience=cfg.patience,
        seed=cfg.seed,
        project=project,
        name=cfg.name,
        plots=True,
        save=True,
        resume=bool(cfg.resume_from),
        degrees=cfg.augmentation.degrees,
        flipud=cfg.augmentation.flipud,
        fliplr=cfg.augmentation.fliplr,
        translate=cfg.augmentation.translate,
        scale=cfg.augmentation.scale,
        shear=cfg.augmentation.shear,
        perspective=cfg.augmentation.perspective,
    )

    if cfg.optimizer != "auto":
        train_kwargs["optimizer"] = cfg.optimizer

    if cfg.augmentation.enable_glare_aug:
        train_kwargs.update(dict(
            hsv_h=cfg.augmentation.hsv_h,
            hsv_s=cfg.augmentation.hsv_s,
            hsv_v=cfg.augmentation.hsv_v,
            mosaic=cfg.augmentation.mosaic,
            mixup=cfg.augmentation.mixup,
            copy_paste=cfg.augmentation.copy_paste,
        ))
    else:
        # Explicitly zero these out so a fixed-camera setup doesn't get
        # ultralytics' aggressive default augmentation by accident.
        train_kwargs.update(dict(mosaic=0.0, mixup=0.0, copy_paste=0.0))

    print("[training_runner] Starting model.train() ...", flush=True)
    results = model.train(**train_kwargs)

    # Print a machine-parseable marker line the job manager looks for,
    # so the UI can show the output directory once training finishes.
    save_dir = getattr(results, "save_dir", None)
    print(f"[training_runner] DONE output_dir={save_dir}", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
