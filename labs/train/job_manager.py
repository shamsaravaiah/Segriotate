"""
Manages training jobs as subprocesses, one at a time.

Why one at a time: two training runs sharing a single GPU will fight over
VRAM and usually both degrade or crash. A simple FIFO queue avoids that
without needing a heavier task-queue system (Celery/RQ) for a single-user
local tool.
"""
import json
import os
import queue
import re
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from schemas import TrainConfig
from app_paths import writable_root

BACKEND_DIR = Path(__file__).resolve().parent
SCRATCH_ROOT = writable_root() / "job_scratch"
SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)

PROGRESS_RE = re.compile(
    r"PROGRESS epoch=(\d+) total_epochs=(\d+) percent=([\d.]+)"
)
BATCH_RE = re.compile(
    r"BATCH epoch=(\d+) total_epochs=(\d+) batch=(\d+) total_batches=(\d+|\?)"
)


class Job:
    def __init__(self, job_id: str, config: TrainConfig):
        self.job_id = job_id
        self.config = config
        self.state = "queued"  # queued, running, completed, failed, stopped
        self.logs: List[str] = []
        self.progress: Optional[dict] = None
        self.output_dir: Optional[str] = None
        self.error: Optional[str] = None
        self.process: Optional[subprocess.Popen] = None
        self.subscribers: List[queue.Queue] = []
        self.lock = threading.Lock()

    def append_log(self, line: str):
        with self.lock:
            self.logs.append(line)
            # Keep last 2000 lines to bound memory on very long runs
            if len(self.logs) > 2000:
                self.logs = self.logs[-2000:]
            for sub in self.subscribers:
                sub.put(line)

    def subscribe(self) -> queue.Queue:
        q = queue.Queue()
        with self.lock:
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue):
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)


class JobManager:
    def __init__(self):
        self.jobs: Dict[str, Job] = {}
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._current_job_id: Optional[str] = None
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def submit(self, config: TrainConfig) -> str:
        job_id = uuid.uuid4().hex[:12]
        job = Job(job_id, config)
        self.jobs[job_id] = job
        self._queue.put(job_id)
        job.append_log(f"[job_manager] Job {job_id} queued.")
        return job_id

    def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def list_jobs(self) -> List[Job]:
        # self.jobs is a plain dict, which preserves insertion order in
        # Python 3.7+ — jobs are inserted here in submit() in the order
        # they were created, so this already reflects submission order.
        # (Previously this sorted by job_id, which is a random uuid hex
        # string, scrambling the list on every refresh.)
        return list(self.jobs.values())

    def stop(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job:
            return False
        if job.process and job.process.poll() is None:
            job.process.terminate()
            job.state = "stopped"
            job.append_log("[job_manager] Job stopped by user.")
            return True
        if job.state == "queued":
            # Never launched yet — mark it stopped so the worker loop's
            # `if job.state == "stopped": continue` check skips it when
            # it's dequeued, instead of silently starting to train.
            job.state = "stopped"
            job.append_log("[job_manager] Job cancelled before it started.")
            return True
        return False

    def _worker_loop(self):
        while True:
            job_id = self._queue.get()
            job = self.jobs.get(job_id)
            if job is None or job.state == "stopped":
                continue
            self._current_job_id = job_id
            try:
                self._run_job(job)
            except Exception as e:
                job.state = "failed"
                job.error = str(e)
                job.append_log(f"[job_manager] Worker crashed: {e}")
            self._current_job_id = None

    def _training_cmd(self, config_path: Path) -> List[str]:
        if getattr(sys, "frozen", False):
            return [sys.executable, "--run-training", str(config_path)]
        return [sys.executable, str(BACKEND_DIR / "training_runner.py"), str(config_path)]

    def _run_job(self, job: Job):
        job.state = "running"
        job.append_log(f"[job_manager] Job {job.job_id} starting.")

        scratch_dir = SCRATCH_ROOT / job.job_id
        scratch_dir.mkdir(parents=True, exist_ok=True)
        config_path = scratch_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(job.config.model_dump(), f, indent=2)

        popen_kwargs = dict(
            args=self._training_cmd(config_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            # PYTHONUNBUFFERED is required in addition to bufsize=1:
            # bufsize=1 only controls how *we* read the pipe, not how
            # the child process buffers its own stdout. Since the
            # child's stdout is a pipe rather than a tty, CPython
            # block-buffers it by default — meaning training_runner's
            # own explicit flush=True prints show up fine, but
            # ultralytics' internal print() calls (which don't set
            # flush=True) can sit unflushed for a long time, making
            # the console look frozen even though training is
            # progressing normally.
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
                "MPLBACKEND": "Agg",
                "KMP_DUPLICATE_LIB_OK": "TRUE",
            },
        )
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        try:
            job.process = subprocess.Popen(**popen_kwargs)
        except Exception as e:
            job.state = "failed"
            job.error = str(e)
            job.append_log(f"[job_manager] Failed to launch: {e}")
            return

        try:
            for line in job.process.stdout:
                line = line.rstrip("\n").rstrip("\r")
                if not line.strip():
                    continue
                self._handle_log_line(job, line)
        except Exception as e:
            job.append_log(f"[job_manager] Log reader error: {e}")
        finally:
            if job.process:
                job.process.wait()

        if job.state == "stopped":
            return

        returncode = job.process.returncode if job.process else -1
        if returncode == 0:
            job.state = "completed"
            job.append_log(f"[job_manager] Job {job.job_id} completed.")
        else:
            job.state = "failed"
            job.error = f"Process exited with code {returncode}"
            job.append_log(f"[job_manager] Job {job.job_id} failed (exit {returncode}).")

    def _handle_log_line(self, job: Job, line: str):
        batch_match = BATCH_RE.search(line)
        if batch_match:
            # BATCH fires once per batch (~dozens to hundreds of
            # times per epoch) — it exists purely to drive the live
            # progress bar. Storing every one of these in job.logs
            # would flood the (size-capped) log buffer and evict
            # real ultralytics output within a few epochs, so it
            # updates progress but is deliberately NOT appended to
            # the log.
            epoch = int(batch_match.group(1))
            total_epochs = int(batch_match.group(2))
            batch = int(batch_match.group(3))
            total_batches_raw = batch_match.group(4)
            total_batches = int(total_batches_raw) if total_batches_raw != "?" else None

            if total_batches:
                percent = round(
                    100.0 * ((epoch - 1) + batch / total_batches) / total_epochs, 1
                ) if total_epochs else 0.0
            else:
                percent = round(100.0 * (epoch - 1) / total_epochs, 1) if total_epochs else 0.0

            job.progress = {
                "epoch": epoch,
                "total_epochs": total_epochs,
                "percent": percent,
                "batch": batch,
                "total_batches": total_batches,
            }
            return

        job.append_log(line)

        progress_match = PROGRESS_RE.search(line)
        if progress_match:
            job.progress = {
                "epoch": int(progress_match.group(1)),
                "total_epochs": int(progress_match.group(2)),
                "percent": float(progress_match.group(3)),
                "batch": None,
                "total_batches": None,
            }

        if "DONE output_dir=" in line:
            job.output_dir = line.split("DONE output_dir=", 1)[1].strip()


# Single shared instance used by the FastAPI app
job_manager = JobManager()
