from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import torch

TARGET_SR = 16000  # pyannote/segmentation-3.0 operates on 16 kHz mono


def load_audio(path: str | Path) -> tuple[torch.Tensor, int, float]:
    """Load, validate and normalise audio to mono 16 kHz (Design §4.1).

    Multi-channel input is downmixed by averaging; other sample rates are
    resampled. The original duration and time origin are preserved. Returns
    (waveform [1, num_samples] float32, sample_rate, duration_seconds).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"audio file not found: {path}")

    try:
        signal, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    except Exception as exc:  # unreadable / corrupt / unsupported format
        raise ValueError(f"could not read audio file {path}: {exc}") from exc

    if signal.size == 0:
        raise ValueError(f"audio file is empty: {path}")

    waveform = torch.from_numpy(np.ascontiguousarray(signal, dtype=np.float32)).unsqueeze(0)
    duration = waveform.shape[1] / TARGET_SR
    return waveform, TARGET_SR, duration
