from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def collar_scored_mask(ref_mask: np.ndarray, resolution: float, collar: float) -> np.ndarray:
    """Boolean mask of frames to score: all frames, minus a +/-collar forgiveness
    region around every reference boundary (Design §8). collar=0 scores everything."""
    scored = np.ones(len(ref_mask), dtype=bool)
    if collar <= 0 or len(ref_mask) == 0:
        return scored
    cf = int(round(collar / resolution))
    boundaries = np.flatnonzero(np.diff(ref_mask.astype(np.int8)) != 0) + 1
    for b in boundaries:
        scored[max(0, b - cf) : b + cf] = False
    return scored


def frame_metrics(
    ref_mask: np.ndarray, hyp_mask: np.ndarray, scored_mask: np.ndarray | None = None
) -> dict:
    """Frame-level (time-weighted) precision/recall/F1 on the overlap class."""
    ref = ref_mask.astype(bool)
    hyp = hyp_mask.astype(bool)
    if scored_mask is not None:
        ref, hyp = ref[scored_mask], hyp[scored_mask]
    tp = int(np.sum(ref & hyp))
    fp = int(np.sum(~ref & hyp))
    fn = int(np.sum(ref & ~hyp))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "iou": iou}


def metrics_from_counts(tp: int, fp: int, fn: int) -> dict:
    """Derive precision/recall/F1/IoU from aggregated counts (corpus micro-average)."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "iou": iou}


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Threshold-independent AP (area under PR curve) on the overlap class."""
    labels = np.asarray(labels)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    return float(average_precision_score(labels, np.asarray(scores)))


def boundary_errors(
    ref_intervals: list[tuple[float, float]], hyp_intervals: list[tuple[float, float]]
) -> dict:
    """Onset/offset absolute timing error (s) between each reference overlap
    region and the detected region it overlaps most (diagnostic, Design §8)."""
    onset_err: list[float] = []
    offset_err: list[float] = []
    for rs, re in ref_intervals:
        best, best_ov = None, 0.0
        for hs, he in hyp_intervals:
            ov = max(0.0, min(re, he) - max(rs, hs))
            if ov > best_ov:
                best, best_ov = (hs, he), ov
        if best is not None:
            onset_err.append(abs(rs - best[0]))
            offset_err.append(abs(re - best[1]))
    n = len(onset_err)
    return {
        "onset_mae": float(np.mean(onset_err)) if n else float("nan"),
        "offset_mae": float(np.mean(offset_err)) if n else float("nan"),
        "n_matched": n,
        "n_reference": len(ref_intervals),
    }


def _demo() -> None:
    ref = np.array([1, 1, 1, 0, 0], dtype=np.int8)
    hyp = np.array([0, 1, 1, 1, 0], dtype=np.int8)
    m = frame_metrics(ref, hyp)
    assert m["tp"] == 2 and m["fp"] == 1 and m["fn"] == 1, m
    assert abs(m["precision"] - 2 / 3) < 1e-9 and abs(m["recall"] - 2 / 3) < 1e-9
    assert abs(m["f1"] - 2 / 3) < 1e-9 and abs(m["iou"] - 0.5) < 1e-9

    # collar removes +/-1 frame around each reference boundary (at idx 3)
    scored = collar_scored_mask(ref, resolution=1.0, collar=1.0)
    assert scored.tolist() == [True, True, False, False, True]

    assert abs(average_precision(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.3, 0.9])) - 1.0) < 1e-9
    assert np.isnan(average_precision(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3])))

    be = boundary_errors([(1.0, 2.0)], [(1.1, 1.9)])
    assert abs(be["onset_mae"] - 0.1) < 1e-9 and abs(be["offset_mae"] - 0.1) < 1e-9 and be["n_matched"] == 1

    assert metrics_from_counts(2, 1, 1)["f1"] == frame_metrics(ref, hyp)["f1"]
    print("metrics self-check OK")


if __name__ == "__main__":
    _demo()
