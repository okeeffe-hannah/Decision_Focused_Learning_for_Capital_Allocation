"""Shared neural-network training configuration shared by PTO/BB/DFL"""

from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 200
    patience: int = 30
    val_frac: float = 0.2
    seed: int = 42


DEFAULT_MODEL_CONFIG = ModelConfig()
