from __future__ import annotations

from typing import Any, Optional, cast

import numpy as np
from matplotlib import pyplot as plt


def _as_xyz_points(points_like) -> np.ndarray:
    points: Any = points_like
    if isinstance(points, np.ndarray) and points.ndim == 0:
        points = points.item()

    if not isinstance(points, np.ndarray) and hasattr(points, "points"):
        points = np.asarray(points.points)
    else:
        points = np.asarray(points)

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(
            "Point cloud input must have shape (N, 3) or be an Open3D point cloud"
        )

    return points[:, :3]


def plot_voxels(
    volumes: list, titles: Optional[list] = None, keypoints_list: Optional[list] = None
):
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
        ax = cast(Any, fig.add_subplot(1, n, i + 1, projection="3d"))

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


def plot_pointcloud(
    pts: list,
    titles: Optional[list] = None,
    keypoints_list: Optional[list] = None,
):
    n = len(pts)
    fig = plt.figure(figsize=(6 * n, 5))

    for i, pt in enumerate(pts):
        xyz = _as_xyz_points(pt)
        ax = cast(Any, fig.add_subplot(1, n, i + 1, projection="3d"))
        ax.scatter(
            xyz[::5, 0],
            xyz[::5, 1],
            xyz[::5, 2],
            s=1,
            c=xyz[::5, 2],
            cmap="viridis",
        )

        if keypoints_list is not None and keypoints_list[i] is not None:
            kp = _as_xyz_points(keypoints_list[i])
            ax.scatter(kp[:, 0], kp[:, 1], kp[:, 2], c="red", s=12)

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
