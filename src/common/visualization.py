from __future__ import annotations

from matplotlib import pyplot as plt
import numpy as np
from typing import Optional


def plot_voxels(volumes: list, titles: Optional[list] = None, keypoints_list: Optional[list] = None):
    """
    Display one or more voxel grids side-by-side.

    Parameters
    ----------
    volumes : list of ndarray, each (D, H, W) bool/float
    titles  : optional list of subplot titles
    keypoints_list : optional list of (N, 3) arrays; pass None per entry to skip
    """
    n = len(volumes)
    fig = plt.figure(figsize=(5 * n, 5))
    for i, vol in enumerate(volumes):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")

        mask = vol.astype(bool)
        ax.voxels(mask, facecolors="steelblue", edgecolors=None, alpha=0.5)

        if keypoints_list is not None and keypoints_list[i] is not None:
            kp = np.asarray(keypoints_list[i])
            if kp.ndim == 2 and kp.shape[0] > 0:
                ax.scatter(kp[:, 0], kp[:, 1], kp[:, 2], c="red", s=50, zorder=5)

        if titles is not None and i < len(titles):
            ax.set_title(titles[i])

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    plt.tight_layout()
    plt.show()


def plot_keypoints_voxel(volume, keypoints: np.ndarray):
    """Display a single voxel grid with keypoints marked in red."""
    plot_voxels([volume], keypoints_list=[keypoints])


def plot_keypoints_pc(points, keypoints):
    # Color keypoints on the point cloud
    pass
