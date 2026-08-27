"""
Pydantic schemas for API requests/responses.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class ClassInfo(BaseModel):
    id: int
    name: str


class DatasetConfig(BaseModel):
    """
    Form-driven dataset definition. The backend generates data.yaml
    from this rather than requiring the user to hand-write one.
    """
    dataset_root: str = ""
    classes: List[ClassInfo] = Field(default_factory=list)
    use_raw_yaml: bool = False
    raw_yaml_path: Optional[str] = None  # used only if use_raw_yaml=True


class AugmentationConfig(BaseModel):
    # Fixed-camera setups usually want these at 0
    degrees: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.0
    translate: float = 0.0
    scale: float = 0.0
    shear: float = 0.0
    perspective: float = 0.0

    # Glare / lighting augmentation (useful for shiny fruit under flash)
    enable_glare_aug: bool = False
    hsv_h: float = 0.015
    hsv_s: float = 0.5
    hsv_v: float = 0.4
    mosaic: float = 0.0
    mixup: float = 0.0
    copy_paste: float = 0.0


class TrainConfig(BaseModel):
    base_model_path: str
    dataset: DatasetConfig
    augmentation: AugmentationConfig = AugmentationConfig()

    epochs: int = 20
    imgsz: int = 640
    batch: int = 8
    workers: int = 0
    device: str = "0"  # "0" for GPU 0, "cpu" for CPU

    lr0: float = 0.003
    lrf: float = 0.0003
    optimizer: str = "auto"  # auto, SGD, Adam, AdamW

    patience: int = 0
    seed: int = 42

    project: str = "runs"
    name: str = "run"

    resume_from: Optional[str] = None  # path to last.pt to resume


class ProgressInfo(BaseModel):
    epoch: int
    total_epochs: int
    percent: float
    batch: Optional[int] = None
    total_batches: Optional[int] = None


class JobStatus(BaseModel):
    job_id: str
    state: str  # queued, running, completed, failed, stopped
    progress: Optional[ProgressInfo] = None
    output_dir: Optional[str] = None
    error: Optional[str] = None


class PresetSaveRequest(BaseModel):
    name: str
    config: TrainConfig


class ValidationIssue(BaseModel):
    level: str  # "error" or "warning"
    message: str


class ValidationResult(BaseModel):
    ok: bool
    issues: List[ValidationIssue] = []
    detected_class_ids: List[int] = []