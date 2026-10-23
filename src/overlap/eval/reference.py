from __future__ import annotations

from pathlib import Path

import numpy as np
from pyannote.core import Annotation, Segment
from pyannote.database.util import load_rttm


def build_annotation(segments: list[tuple[str, float, float]], uri: str | None = None) -> Annotation:
    """Build an Annotation from (speaker, start, end) tuples."""
    annotation = Annotation(uri=uri)
    for i, (speaker, start, end) in enumerate(segments):
        if end > start:
            annotation[Segment(start, end), i] = speaker
    return annotation


def load_rttm_annotation(rttm_path: str | Path, uri: str | None = None) -> Annotation:
    """Load a single-file RTTM into one Annotation.

    If `uri` is given, return that file's annotation; otherwise return the only
    one present (error if the RTTM holds several and none is selected).
    """
    annotations = load_rttm(str(rttm_path))
    if uri is not None:
        return annotations[uri]
    if len(annotations) == 1:
        return next(iter(annotations.values()))
    raise ValueError(
        f"{rttm_path} contains {len(annotations)} files; pass uri to select one"
    )


def reference_overlap_intervals(annotation: Annotation) -> list[tuple[float, float]]:
    """Reference overlap = regions with >= 2 simultaneous speakers (Design §8)."""
    overlap = annotation.get_overlap().support()
    return [(float(seg.start), float(seg.end)) for seg in overlap]


def sample_intervals_at(times: np.ndarray, intervals: list[tuple[float, float]]) -> np.ndarray:
    """1 where a grid time falls inside any [start, end) interval, else 0."""
    labels = np.zeros(len(times), dtype=np.int8)
    for start, end in intervals:
        labels[(times >= start) & (times < end)] = 1
    return labels
