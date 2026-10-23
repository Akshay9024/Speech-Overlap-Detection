from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_timeline(
    times: np.ndarray,
    mask: np.ndarray,
    duration: float,
    out_path: str | Path,
    reference_mask: np.ndarray | None = None,
    title: str | None = None,
) -> None:
    """Draw the 0/1 overlap step plot (Design §6). Reference overlap, if given,
    is shaded behind the detected signal."""
    fig, ax = plt.subplots(figsize=(12, 2.5))

    if reference_mask is not None:
        ax.fill_between(
            np.append(times, duration),
            np.append(reference_mask, reference_mask[-1] if len(reference_mask) else 0),
            step="post",
            color="tab:orange",
            alpha=0.25,
            label="reference overlap",
        )

    x = np.append(times, duration)
    y = np.append(mask, mask[-1] if len(mask) else 0)
    ax.step(x, y, where="post", color="tab:blue", label="detected overlap")

    ax.set_xlim(0, duration)
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Overlap")
    if title:
        ax.set_title(title)
    if reference_mask is not None:
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
