"""Configuration dataclasses and serialization helpers."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import yaml


@dataclass
class DataConfig:
    dataset_root: str = ""
    train_manifest: str = ""
    validation_manifest: str = ""
    synthetic_pretraining_dataset_root: str = ""
    synthetic_pretraining_train_manifest: str = ""
    synthetic_pretraining_validation_manifest: str = ""
    real_finetuning_dataset_root: str = ""
    real_finetuning_train_manifest: str = ""
    real_finetuning_validation_manifest: str = ""
    input_length: int = 4096
    iq_fallback_sample_rate_hz: float | None = None
    num_workers: int = 0
    persistent_workers: bool = False
    prefetch_factor: int = 2
    pin_memory: bool = True


@dataclass
class TrainingConfig:
    seed: int = 42
    optimizer: str = "AdamW"
    learning_rate: float = 0.0005
    weight_decay: float = 0.0001
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    max_epochs: int = 40
    early_stopping_patience: int = 8
    focal_gamma: float = 2.0
    low_snr_qam_weight: float = 0.25
    amp: bool = True
    phase_rotation_augmentation: bool = True
    model_selection_priority: tuple[str, ...] = ("macro_f1", "recall_64QAM", "accuracy")
    output_dir: str = "research_qam_psk_fsk_model/results"
    gradient_clip_norm: float | None = 5.0
    profile_timings: bool = False
    gpu_utilization_logging: bool = False


@dataclass
class TransferConfig:
    scaler_policy: str = "fit_train"
    encoder_freeze_epochs: int = 0
    selective_unfreeze_epoch: int | None = None
    selective_unfreeze_modules: tuple[str, ...] = ()
    encoder_learning_rate: float | None = None
    fusion_head_learning_rate: float | None = None


@dataclass
class ArchitectureConfig:
    num_classes: int = 7
    feature_count: int = 22
    iq_embedding_dim: int = 128
    auxiliary_embedding_dim: int = 64
    fused_embedding_dim: int = 128
    psd_bins: int = 2048
    dropout: float = 0.25


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    transfer: TransferConfig = field(default_factory=TransferConfig)
    stage: str = "real_from_scratch"
    init_checkpoint: str = ""
    repository_version: str = "unversioned"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return ExperimentConfig(
        data=DataConfig(**raw.get("data", {})),
        training=TrainingConfig(**raw.get("training", {})),
        architecture=ArchitectureConfig(**raw.get("architecture", {})),
        transfer=TransferConfig(**raw.get("transfer", {})),
        stage=raw.get("stage", "real_from_scratch"),
        init_checkpoint=raw.get("init_checkpoint", ""),
        repository_version=raw.get("repository_version", "unversioned"),
    )
