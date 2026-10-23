from __future__ import annotations

import math

import numpy as np


def intervals_to_mask(
    intervals: list[tuple[float, float]], duration: float, resolution: float
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterise [start, end) intervals onto a fixed grid (Design §6).

    Frame i spans [i*resolution, (i+1)*resolution) and is 1 if it intersects
    any interval. Returns (frame_left_edges, mask[int{0,1}]) covering [0, duration].
    The same function rasterises detection and reference, keeping scoring on a
    common grid (Design §8).
    """
    if resolution <= 0:
        raise ValueError("resolution must be > 0")
    n = max(0, math.ceil(duration / resolution - 1e-9))
    times = np.arange(n) * resolution
    mask = np.zeros(n, dtype=np.int8)
    for start, end in intervals:
        if end <= start:
            continue
        i0 = max(0, math.floor(start / resolution + 1e-9))
        i1 = min(n, math.ceil(end / resolution - 1e-9))
        if i1 > i0:
            mask[i0:i1] = 1
    return times, mask


def _demo() -> None:
    # overlap 3.0-4.0 and 8.2-9.1 on a 12 s file at 10 ms grid (Design §6 example)
    times, mask = intervals_to_mask([(3.0, 4.0), (8.2, 9.1)], 12.0, 0.01)
    assert len(times) == len(mask) == 1200
    assert mask[:300].sum() == 0
    assert mask[300:400].sum() == 100  # [3.0, 4.0)
    assert mask[400:820].sum() == 0
    assert mask[820:910].sum() == 90   # [8.2, 9.1)
    assert mask[910:].sum() == 0
    # empty / degenerate inputs
    assert intervals_to_mask([], 1.0, 0.01)[1].sum() == 0
    assert intervals_to_mask([(1.0, 1.0)], 2.0, 0.01)[1].sum() == 0
    print("timeline self-check OK")


if __name__ == "__main__":
    _demo()
