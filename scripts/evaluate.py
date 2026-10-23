from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyannote.audio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from overlap.audio import load_audio  # noqa: E402
from overlap.config import load_config  # noqa: E402
from overlap.detect import build_pipeline, detect_overlap, overlap_scores  # noqa: E402
from overlap.eval.metrics import (  # noqa: E402
    average_precision,
    boundary_errors,
    collar_scored_mask,
    frame_metrics,
    metrics_from_counts,
)
from overlap.eval.reference import (  # noqa: E402
    load_rttm_annotation,
    reference_overlap_intervals,
    sample_intervals_at,
)
from overlap.plot import plot_timeline  # noqa: E402
from overlap.timeline import intervals_to_mask  # noqa: E402


def evaluate(manifest_path: str, config_path: str, out_dir: str, collar: float, plots: bool, limit: int | None = None) -> dict:
    manifest = json.loads(Path(manifest_path).read_text())
    cfg = load_config(config_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = cfg.timeline_resolution

    pipeline = build_pipeline(cfg)

    entries = manifest["files"][:limit] if limit else manifest["files"]
    per_file = []
    agg_strict = {"tp": 0, "fp": 0, "fn": 0}
    agg_collar = {"tp": 0, "fp": 0, "fn": 0}
    aps = []

    for i, entry in enumerate(entries, 1):
        uri = entry["uri"]
        print(f"[{i}/{len(entries)}] {uri}", file=sys.stderr, flush=True)
        waveform, sr, duration = load_audio(entry["audio"])

        annotation = load_rttm_annotation(entry["rttm"], uri=entry.get("rttm_uri"))
        ref_intervals = reference_overlap_intervals(annotation)
        hyp_intervals = detect_overlap(pipeline, waveform, sr)

        _, ref_mask = intervals_to_mask(ref_intervals, duration, res)
        times, hyp_mask = intervals_to_mask(hyp_intervals, duration, res)

        strict = frame_metrics(ref_mask, hyp_mask)
        scored = collar_scored_mask(ref_mask, res, collar)
        lenient = frame_metrics(ref_mask, hyp_mask, scored_mask=scored)

        frame_times, scores = overlap_scores(pipeline, waveform, sr)
        ap = average_precision(sample_intervals_at(frame_times, ref_intervals), scores)
        aps.append(ap)

        bnd = boundary_errors(ref_intervals, hyp_intervals)

        for k in agg_strict:
            agg_strict[k] += strict[k]
            agg_collar[k] += lenient[k]

        per_file.append({
            "uri": uri,
            "condition": entry.get("condition"),
            "duration": duration,
            "overlap_ratio": float(ref_mask.mean()) if len(ref_mask) else 0.0,
            "strict": strict,
            "collar": lenient,
            "average_precision": ap,
            "boundary_errors": bnd,
        })

        if plots:
            plot_timeline(times, hyp_mask, duration, out / f"{uri}_overlap.png",
                          reference_mask=ref_mask, title=uri)

    report = {
        "dataset": manifest.get("dataset"),
        "seen_domain": manifest.get("seen_domain"),
        "metadata": {
            "checkpoint": cfg.segmentation,
            "pyannote_audio_version": pyannote.audio.__version__,
            "timeline_resolution": res,
            "min_overlap_duration": cfg.min_overlap_duration,
            "min_gap_duration": cfg.min_gap_duration,
            "collar": collar,
            "device": cfg.resolved_device(),
        },
        "corpus_strict": metrics_from_counts(**agg_strict),
        "corpus_collar": metrics_from_counts(**agg_collar),
        "mean_average_precision": float(np.nanmean(aps)) if aps else float("nan"),
        "per_file": per_file,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate overlap detection on a dataset manifest")
    parser.add_argument("manifest", help="JSON manifest with a 'files' list of {uri, audio, rttm}")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out-dir", default="outputs/eval")
    parser.add_argument("--collar", type=float, default=0.25, help="lenient collar in seconds (Design §8)")
    parser.add_argument("--plots", action="store_true", help="save per-file overlap plots")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N files (smoke test)")
    args = parser.parse_args()

    report = evaluate(args.manifest, args.config, args.out_dir, args.collar, args.plots, args.limit)
    summary = {k: report[k] for k in ("dataset", "seen_domain", "corpus_strict", "corpus_collar", "mean_average_precision", "metadata")}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
