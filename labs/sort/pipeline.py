"""
ResNet-50 visual vectors + self-adaptive hierarchical grouping.

Images stay in the original folder. Bins are only labels in the UI.
ResNet-50 ImageNet weights download automatically via torchvision.

Grouping uses average-linkage Manhattan distance, then scipy's inconsistency
coefficient to cut the tree at natural breaks (no user threshold).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

LogFn = Callable[[str], None]
ProgressFn = Callable[[int, int, str], None]

# Inconsistency cut: flag links that jump vs their local neighbourhood.
# Not a distance slider — relative to the tree the photos themselves build.
INCONSISTENCY_T = 1.1
INCONSISTENCY_DEPTH = 2
BATCH_SIZE = 16
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
EMBEDDER_ID = "resnet50"
CACHE_VERSION = 1


def _log(log: Optional[LogFn], message: str) -> None:
    (log or print)(message)


def _device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _transform():
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_tensor(path: Path, transform) -> Optional[object]:
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            return transform(img.convert("RGB"))
    except Exception as e:
        print(f"[sort] Skipping {path.name}: {e}", flush=True)
        return None


class ResNetEmbedder:
    def __init__(self, hub_dir: Path):
        self.hub_dir = hub_dir
        self.model = None
        self.device = None

    def load(self, log: Optional[LogFn] = None) -> None:
        import os
        import torch
        import torchvision.models as models

        self.hub_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TORCH_HOME", str(self.hub_dir))
        self.device = _device()
        _log(log, f"Loading ResNet-50 on {self.device}…")
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model = torch.nn.Sequential(*(list(resnet.children())[:-1]))
        self.model.eval()
        self.model.to(self.device)

    def embed_batch(self, tensors: list) -> np.ndarray:
        import torch

        if self.model is None:
            raise RuntimeError("ResNet-50 is not loaded")
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            vectors = self.model(batch)
            vectors = vectors.view(vectors.size(0), -1)
        return vectors.detach().float().cpu().numpy()


def l2_normalize(features: np.ndarray) -> np.ndarray:
    """Unit L2 rows so Manhattan distances stay on a comparable scale."""
    feats = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return feats / norms


def cluster_features(features: np.ndarray, log: Optional[LogFn] = None) -> np.ndarray:
    from scipy.cluster.hierarchy import fcluster, linkage

    n = int(features.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.int32)
    if n == 1:
        return np.zeros((1,), dtype=np.int32)
    _log(log, "Building hierarchical tree (Manhattan)…")
    tree = linkage(features, method="average", metric="cityblock")
    _log(log, "Finding natural groups (inconsistency)…")
    labels = fcluster(
        tree,
        t=INCONSISTENCY_T,
        criterion="inconsistent",
        depth=INCONSISTENCY_DEPTH,
    )
    return np.asarray(labels, dtype=np.int32)


def _stat_file(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
        return int(st.st_mtime_ns), int(st.st_size)
    except OSError:
        return None


def _cache_json_path(npz_path: Path) -> Path:
    return npz_path.with_suffix(".json")


def load_embedding_cache(npz_path: Path) -> dict | None:
    json_path = _cache_json_path(npz_path)
    if not npz_path.is_file() or not json_path.is_file():
        return None
    try:
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        with np.load(npz_path) as data:
            features = np.array(data["features"], dtype=np.float32)
            mtimes = np.array(data["mtimes"], dtype=np.int64)
            sizes = np.array(data["sizes"], dtype=np.int64)
    except (OSError, json.JSONDecodeError, KeyError, ValueError):
        return None
    if meta.get("version") != CACHE_VERSION or meta.get("embedder") != EMBEDDER_ID:
        return None
    stems = meta.get("stems") or []
    if not (len(stems) == features.shape[0] == mtimes.shape[0] == sizes.shape[0]):
        return None
    by_stem = {
        str(stem): (features[i], int(mtimes[i]), int(sizes[i]))
        for i, stem in enumerate(stems)
    }
    skipped: dict[str, tuple[int, int] | None] = {}
    for item in meta.get("skipped") or []:
        if isinstance(item, dict) and item.get("stem"):
            skipped[str(item["stem"])] = (
                int(item.get("mtime_ns") or 0),
                int(item.get("size") or 0),
            )
        elif isinstance(item, str):
            skipped[item] = None
    return {"by_stem": by_stem, "skipped": skipped, "features": features}


def save_embedding_cache(
    npz_path: Path,
    stems: list[str],
    features: np.ndarray,
    mtimes: list[int],
    sizes: list[int],
    skipped: list[dict],
) -> None:
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = _cache_json_path(npz_path)
    n = len(stems)
    if n:
        feats = np.asarray(features, dtype=np.float32)
        if feats.ndim != 2 or feats.shape[0] != n:
            raise ValueError("embedding cache feature rows must match stems")
    else:
        feats = np.zeros((0, 1), dtype=np.float32)
    mtimes_arr = np.asarray(mtimes, dtype=np.int64)
    sizes_arr = np.asarray(sizes, dtype=np.int64)
    npz_tmp = npz_path.with_name(npz_path.name + ".tmp")
    json_tmp = json_path.with_name(json_path.name + ".tmp")
    with open(npz_tmp, "wb") as fh:
        np.savez(fh, features=feats, mtimes=mtimes_arr, sizes=sizes_arr)
    meta = {
        "version": CACHE_VERSION,
        "embedder": EMBEDDER_ID,
        "stems": list(stems),
        "skipped": skipped,
    }
    json_tmp.write_text(json.dumps(meta) + "\n", encoding="utf-8")
    npz_tmp.replace(npz_path)
    json_tmp.replace(json_path)


def _bins_from_features(
    stems: list[str],
    features: np.ndarray,
    skipped: list[str],
    log: Optional[LogFn] = None,
) -> list[dict]:
    bins: list[dict] = []
    if stems:
        matrix = l2_normalize(features)
        labels = cluster_features(matrix, log)
        order: list[int] = []
        for lab in labels:
            lab_i = int(lab)
            if lab_i not in order:
                order.append(lab_i)
        for idx, lab in enumerate(order):
            member_stems = [stems[j] for j, L in enumerate(labels) if int(L) == lab]
            bins.append({
                "id": f"bin-{idx}",
                "name": "",
                "unsorted": False,
                "stems": member_stems,
            })
        _log(log, f"Found {len(bins)} group(s).")
    if skipped:
        bins.append({
            "id": "unsorted",
            "name": "Unsorted",
            "unsorted": True,
            "stems": skipped,
        })
    return bins


def run_sort(
    files: list[dict],
    images_dir: Path,
    detect_model=None,
    hub_dir: Path | None = None,
    cache_path: Path | None = None,
    log: Optional[LogFn] = None,
    progress: Optional[ProgressFn] = None,
) -> dict:
    """
    files: [{name, stem}] from list_image_files().
    Original files are not moved. detect_model is unused (kept for the job API).
    Embeddings are cached at cache_path so later sorts only re-cluster.
    """
    del detect_model
    total = len(files)
    empty = {
        "bins": [],
        "embedder": EMBEDDER_ID,
        "no_fruit": [],
        "embedded": 0,
        "from_cache": False,
        "cache_hits": 0,
    }
    if total == 0:
        return empty

    cached = load_embedding_cache(cache_path) if cache_path else None
    by_stem = (cached or {}).get("by_stem") or {}
    skipped_cached = (cached or {}).get("skipped") or {}

    feat_map: dict[str, np.ndarray] = {}
    fp_map: dict[str, tuple[int, int]] = {}
    skipped_meta: list[dict] = []
    need: list[tuple[dict, Path, int, int]] = []

    if progress:
        progress(0, total, "Matching cached embeddings…" if cached else "Reading folder…")

    for item in files:
        path = images_dir / item["name"]
        stem = item["stem"]
        if path.suffix.lower() not in VALID_EXTENSIONS:
            skipped_meta.append({"stem": stem, "mtime_ns": 0, "size": 0})
            continue
        fp = _stat_file(path)
        if fp is None:
            skipped_meta.append({"stem": stem, "mtime_ns": 0, "size": 0})
            continue
        mtime_ns, size = fp
        hit = by_stem.get(stem)
        if hit is not None and hit[1] == mtime_ns and hit[2] == size:
            feat_map[stem] = hit[0]
            fp_map[stem] = (mtime_ns, size)
            continue
        skip_hit = skipped_cached.get(stem)
        if skip_hit is not None and skip_hit == (mtime_ns, size):
            skipped_meta.append({"stem": stem, "mtime_ns": mtime_ns, "size": size})
            continue
        need.append((item, path, mtime_ns, size))

    cache_hits = len(feat_map)
    if need:
        cache = hub_dir or (images_dir / ".sort_hub")
        extractor = ResNetEmbedder(cache)
        extractor.load(log)
        transform = _transform()
        tensors: list = []
        pending: list[tuple[str, int, int]] = []
        for item, path, mtime_ns, size in need:
            tensor = load_tensor(path, transform)
            if tensor is None:
                skipped_meta.append({"stem": item["stem"], "mtime_ns": mtime_ns, "size": size})
                continue
            tensors.append(tensor)
            pending.append((item["stem"], mtime_ns, size))
        new_feats: list[np.ndarray] = []
        for start in range(0, len(tensors), BATCH_SIZE):
            chunk = tensors[start:start + BATCH_SIZE]
            if progress:
                progress(
                    start,
                    max(len(tensors), 1),
                    f"Embedding {start + 1}–{min(start + BATCH_SIZE, len(tensors))} / {len(tensors)}",
                )
            new_feats.append(extractor.embed_batch(chunk))
        tensors.clear()
        if new_feats:
            stacked = np.vstack(new_feats)
            for row, (stem, mtime_ns, size) in zip(stacked, pending):
                feat_map[stem] = row
                fp_map[stem] = (mtime_ns, size)
    elif cache_hits:
        _log(log, f"Using {cache_hits} cached embedding(s)…")

    stems = [item["stem"] for item in files if item["stem"] in feat_map]
    skipped_stems = [item["stem"] for item in skipped_meta]
    if stems:
        matrix = np.stack([feat_map[stem] for stem in stems], axis=0)
        mtimes = [fp_map[stem][0] for stem in stems]
        sizes = [fp_map[stem][1] for stem in stems]
    else:
        matrix = np.zeros((0, 1), dtype=np.float32)
        mtimes = []
        sizes = []

    if cache_path and (need or cached is None):
        save_embedding_cache(cache_path, stems, matrix, mtimes, sizes, skipped_meta)

    if progress:
        progress(total, total, "Grouping…")

    bins = _bins_from_features(stems, matrix, skipped_stems, log)
    return {
        "bins": bins,
        "embedder": EMBEDDER_ID,
        "no_fruit": skipped_stems,
        "embedded": len(stems),
        "from_cache": not need and cached is not None,
        "cache_hits": cache_hits,
    }
