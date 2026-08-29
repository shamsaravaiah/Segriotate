"""Background sort job and on-disk bins.json."""
from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path

import config  # noqa: E402
from pipeline import run_sort

_lock = threading.Lock()
_job = {
    "state": "idle",  # idle | running | done | error
    "message": "",
    "done": 0,
    "total": 0,
    "error": None,
    "embedder": "",
    "embedded": 0,
    "no_fruit": 0,
    "from_cache": False,
}
_session: dict | None = None
_thread: threading.Thread | None = None


_hide_folder = False


def sort_dir_for(images_dir: Path, create: bool = True) -> Path:
    config.ensure_workspace()
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in images_dir.name) or "images"
    path = config.SORT_ROOT / safe
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def folder_hidden() -> bool:
    with _lock:
        return _hide_folder


def reveal_folder() -> None:
    global _hide_folder
    with _lock:
        _hide_folder = False


def bins_path(images_dir: Path) -> Path:
    return sort_dir_for(images_dir) / "bins.json"


def hub_dir() -> Path:
    config.ensure_workspace()
    path = config.SORT_ROOT / "hub"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_bins(images_dir: Path) -> dict | None:
    path = bins_path(images_dir)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def save_bins(images_dir: Path, payload: dict) -> Path:
    path = bins_path(images_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2) + "\n"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(path)
    return path


def status() -> dict:
    with _lock:
        return dict(_job)


def get_session() -> dict | None:
    with _lock:
        return dict(_session) if _session else None


def set_session(data: dict | None) -> None:
    global _session
    with _lock:
        _session = dict(data) if data else None


def _set(**kwargs) -> None:
    with _lock:
        _job.update(kwargs)


def _reset_job() -> None:
    _job.update({
        "state": "idle",
        "message": "",
        "done": 0,
        "total": 0,
        "error": None,
        "embedder": "",
        "embedded": 0,
        "no_fruit": 0,
        "from_cache": False,
    })


def _delete_sort_folder(folder: Path) -> bool:
    if not folder.exists():
        return False
    if folder.is_file():
        folder.unlink()
        return True
    shutil.rmtree(folder, ignore_errors=True)
    return not folder.exists()


def clear_cache(images_dir: Path | None = None) -> dict:
    """Drop saved vectors and bins. Original photos on disk are not deleted."""
    global _hide_folder, _session
    del images_dir
    deleted: list[str] = []
    with _lock:
        if _job["state"] == "running":
            _hide_folder = True
        config.ensure_workspace()
        root = config.SORT_ROOT
        if root.is_dir():
            for child in list(root.iterdir()):
                if child.name == "hub":
                    continue
                if child.is_dir() and _delete_sort_folder(child):
                    deleted.append(child.name)
                elif child.is_file():
                    try:
                        child.unlink()
                        deleted.append(child.name)
                    except OSError:
                        pass
        _reset_job()
        _session = None
        _hide_folder = True
    print(f"[sort] Cleared cache: {deleted or 'nothing on disk'}", flush=True)
    return {"ok": True, "deleted": deleted}


def start(
    files: list[dict],
    images_dir: Path,
    detect_model=None,
) -> bool:
    global _thread
    cache_path = sort_dir_for(images_dir) / "embeddings.npz"
    has_cache = cache_path.is_file() and cache_path.with_suffix(".json").is_file()
    reveal_folder()
    with _lock:
        if _job["state"] == "running":
            return False
        _job.update({
            "state": "running",
            "message": "Grouping from cache…" if has_cache else "Starting sort…",
            "done": 0,
            "total": len(files),
            "error": None,
            "embedder": "",
            "embedded": 0,
            "no_fruit": 0,
            "from_cache": False,
        })

    def work():
        try:
            def log(msg: str) -> None:
                _set(message=msg)
                print(f"[sort] {msg}", flush=True)

            def progress(done: int, total: int, msg: str) -> None:
                _set(done=done, total=total, message=msg)

            result = run_sort(
                files,
                images_dir,
                detect_model=detect_model,
                hub_dir=hub_dir(),
                cache_path=cache_path,
                log=log,
                progress=progress,
            )
            if folder_hidden():
                return
            payload = {
                "images_dir": str(images_dir),
                "embedder": result.get("embedder") or "",
                "bins": result.get("bins") or [],
            }
            save_bins(images_dir, payload)
            if folder_hidden():
                return
            _set(
                state="done",
                message=(
                    "Grouped from cached embeddings"
                    if result.get("from_cache")
                    else "Sorted"
                ),
                done=_job["total"],
                total=_job["total"],
                embedder=payload["embedder"],
                embedded=int(result.get("embedded") or 0),
                no_fruit=len(result.get("no_fruit") or []),
                from_cache=bool(result.get("from_cache")),
            )
        except Exception as e:
            _set(state="error", error=str(e), message=f"Sort failed: {e}")

    _thread = threading.Thread(target=work, daemon=True, name="segriotate-sort")
    _thread.start()
    return True
