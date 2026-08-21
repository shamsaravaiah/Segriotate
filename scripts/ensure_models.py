"""Download missing .pt weights into models/dot-pt/ on first launch."""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

import config

USER_AGENT = "Segriotate/1.0 (+https://github.com/ultralytics/assets)"
MIN_BYTES = 1024


def _status(cb, msg: str) -> None:
    if cb:
        cb(msg)
    print(msg, flush=True)


def _complete(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > MIN_BYTES
    except OSError:
        return False


def _download(url: str, dest: Path, on_status) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        got = 0
        with open(tmp, "wb") as out:
            while True:
                chunk = resp.read(256 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                got += len(chunk)
                if total:
                    pct = min(99, got * 100 // total)
                    mb = got / (1024 * 1024)
                    tot_mb = total / (1024 * 1024)
                    _status(on_status, f"Downloading {dest.name}… {pct}% ({mb:.0f}/{tot_mb:.0f} MB)")
                else:
                    _status(on_status, f"Downloading {dest.name}… {got / (1024 * 1024):.0f} MB")
    if tmp.stat().st_size <= MIN_BYTES:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"download too small: {dest.name}")
    tmp.replace(dest)


def planned_downloads() -> list[tuple[Path, str]]:
    jobs: list[tuple[Path, str]] = []
    pt_dir = config.PROJECT_ROOT / "models" / "dot-pt"
    url = str(getattr(config, "SEGMENTATION_DOWNLOAD_URL", "") or "").strip()
    if url:
        jobs.append((Path(config.MODEL_PATH), url))
    for name, href in (getattr(config, "MODEL_DOWNLOADS", {}) or {}).items():
        href = str(href or "").strip()
        if not href:
            continue
        jobs.append((pt_dir / Path(str(name)).name, href))
    return jobs


def ensure_models(on_status=None) -> None:
    pt_dir = config.PROJECT_ROOT / "models" / "dot-pt"
    engine_dir = config.PROJECT_ROOT / "models" / "dot-engine"
    source_dir = Path(getattr(config, "MODEL_SOURCE_DIR", config.PROJECT_ROOT / "models" / "source"))
    pt_dir.mkdir(parents=True, exist_ok=True)
    engine_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    missing = [(path, url) for path, url in planned_downloads() if not _complete(path)]
    if not missing:
        _status(on_status, "Models are already on disk.")
        return

    errors: list[str] = []
    for i, (dest, url) in enumerate(missing, 1):
        _status(on_status, f"Downloading {dest.name} ({i}/{len(missing)})…")
        try:
            _download(url, dest, on_status)
            _status(on_status, f"Saved {dest.name}")
        except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as e:
            errors.append(f"{dest.name}: {e}")
            _status(on_status, f"Could not download {dest.name}: {e}")
    if errors:
        _status(on_status, "Some model downloads failed. The editor will still open.")
