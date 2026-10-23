from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyannote.audio

from .audio import TARGET_SR, load_audio
from .config import load_config
from .detect import build_pipeline, detect_overlap
from .plot import plot_timeline
from .timeline import intervals_to_mask


def run(audio_path: str, config_path: str, out_dir: str) -> dict:
    cfg = load_config(config_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(audio_path).stem

    waveform, sample_rate, duration = load_audio(audio_path)
    pipeline = build_pipeline(cfg)
    intervals = detect_overlap(pipeline, waveform, sample_rate)
    times, mask = intervals_to_mask(intervals, duration, cfg.timeline_resolution)

    result = {
        "audio": str(audio_path),
        "duration": duration,
        "intervals": [[s, e] for s, e in intervals],
        "metadata": {
            "checkpoint": cfg.segmentation,
            "pyannote_audio_version": pyannote.audio.__version__,
            "target_sample_rate": TARGET_SR,
            "timeline_resolution": cfg.timeline_resolution,
            "min_overlap_duration": cfg.min_overlap_duration,
            "min_gap_duration": cfg.min_gap_duration,
            "device": cfg.resolved_device(),
        },
    }

    (out / f"{stem}_intervals.json").write_text(json.dumps(result, indent=2))
    np.savez(out / f"{stem}_timeline.npz", times=times, mask=mask)
    plot_timeline(times, mask, duration, out / f"{stem}_overlap.png", title=stem)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlapped speech detection")
    parser.add_argument("audio", help="path to the input audio file")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    result = run(args.audio, args.config, args.out_dir)
    print(json.dumps({"intervals": result["intervals"], "metadata": result["metadata"]}, indent=2))


if __name__ == "__main__":
    main()
