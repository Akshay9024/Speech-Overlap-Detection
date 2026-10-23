from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import yaml


@dataclass(frozen=True)
class Config:
    segmentation: str
    min_overlap_duration: float
    min_gap_duration: float
    timeline_resolution: float
    device: str

    def resolved_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)

    cfg = Config(
        segmentation=str(raw["segmentation"]),
        min_overlap_duration=float(raw["min_overlap_duration"]),
        min_gap_duration=float(raw["min_gap_duration"]),
        timeline_resolution=float(raw["timeline_resolution"]),
        device=str(raw["device"]),
    )

    if cfg.min_overlap_duration < 0 or cfg.min_gap_duration < 0:
        raise ValueError("min_overlap_duration and min_gap_duration must be >= 0")
    if cfg.timeline_resolution <= 0:
        raise ValueError("timeline_resolution must be > 0")
    if cfg.device not in ("auto", "cpu", "cuda"):
        raise ValueError(f"device must be auto|cpu|cuda, got {cfg.device!r}")
    return cfg
