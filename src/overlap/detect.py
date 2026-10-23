from __future__ import annotations

import os

import torch
from pyannote.audio.pipelines import OverlappedSpeechDetection

from .config import Config


def build_pipeline(cfg: Config) -> OverlappedSpeechDetection:
    """Build and instantiate the overlapped speech detection pipeline.

    The HF token is read from HF_TOKEN; if unset, the cached huggingface-cli
    login is used. For the powerset segmentation-3.0 model the pipeline fixes
    onset/offset=0.5 internally, so only the two duration controls are set.
    """
    token = os.environ.get("HF_TOKEN", True)
    pipeline = OverlappedSpeechDetection(
        segmentation=cfg.segmentation, use_auth_token=token
    )
    pipeline.instantiate(
        {
            "min_duration_on": cfg.min_overlap_duration,
            "min_duration_off": cfg.min_gap_duration,
        }
    )
    pipeline.to(torch.device(cfg.resolved_device()))
    return pipeline


def detect_overlap(
    pipeline: OverlappedSpeechDetection, waveform: torch.Tensor, sample_rate: int
) -> list[tuple[float, float]]:
    """Run the pipeline and return sorted, non-overlapping [start, end) intervals."""
    annotation = pipeline({"waveform": waveform, "sample_rate": sample_rate})
    support = annotation.get_timeline().support()
    return [(float(seg.start), float(seg.end)) for seg in support]


def overlap_scores(
    pipeline: OverlappedSpeechDetection, waveform: torch.Tensor, sample_rate: int
) -> tuple["np.ndarray", "np.ndarray"]:
    """Per-frame overlap posterior (frame_start_times, scores in [0, 1]) for AP.

    Reads the pipeline's internal segmentation inference, whose pre-aggregation
    hook already reduces each frame to the 2nd-largest speaker activation — i.e.
    the probability that >= 2 speakers are active. Couples to pipeline internals.
    """
    import numpy as np

    swf = pipeline._segmentation({"waveform": waveform, "sample_rate": sample_rate})
    data = np.asarray(swf.data).reshape(len(swf.data), -1)
    scores = data[:, 0]
    sw = swf.sliding_window
    times = sw.start + np.arange(len(scores)) * sw.step
    return times, scores
