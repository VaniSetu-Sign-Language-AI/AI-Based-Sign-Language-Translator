"""Centralised configuration loader (enterprise pattern: single source of truth)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "config.yaml"


@dataclass(frozen=True)
class AppConfig:
    raw_csv: str
    processed_dir: str
    image_dir: str
    sequence_length: int
    stride: int
    test_size: float
    random_state: int
    min_detection_confidence: float
    min_tracking_confidence: float


@dataclass(frozen=True)
class ModelConfig:
    conv1_filters: int
    conv2_filters: int
    lstm1_units: int
    lstm2_units: int
    embedding_dim: int
    dropout: float
    learning_rate: float


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    epochs: int
    early_stopping_patience: int
    reduce_lr_patience: int
    model_dir: str
    model_name: str
    rf_name: str
    encoder_name: str
    scaler_name: str
    rf_n_estimators: int
    rf_max_depth: int | None
    rf_n_jobs: int
    open_set_name: str
    open_set_margin: float


@dataclass(frozen=True)
class InferConfig:
    sequence_length: int
    confidence_threshold: float
    smoothing_window: int
    unknown_threshold: float
    margin_threshold: float


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: str | os.PathLike | None = None) -> tuple[AppConfig, ModelConfig, TrainConfig, InferConfig, dict]:
    raw = _load_yaml(Path(path) if path else CONFIG_PATH)
    d, m, t, i = raw["data"], raw["model"], raw["training"], raw["inference"]
    app = AppConfig(**{k: d[k] for k in AppConfig.__dataclass_fields__})
    model = ModelConfig(**{k: m[k] for k in ModelConfig.__dataclass_fields__})
    train = TrainConfig(**{k: t[k] for k in TrainConfig.__dataclass_fields__})
    infer = InferConfig(**{k: i[k] for k in InferConfig.__dataclass_fields__})
    return app, model, train, infer, raw
