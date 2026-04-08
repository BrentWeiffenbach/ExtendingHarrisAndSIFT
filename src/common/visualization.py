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


def _shift_keypoints_to_surface_corners(
    volume_zyx: np.ndarray,
    keypoints_xyz: np.ndarray,
) -> np.ndarray:
    """Display occupied-voxel keypoints on nearby exterior voxel corners.

    This is a visualization-only transform to make boundary corners appear less
    inset in 3D voxel renders.
    """
    vol = np.asarray(volume_zyx).astype(bool)
    if keypoints_xyz.size == 0:
        return keypoints_xyz

    nz, ny, nx = vol.shape
    out = keypoints_xyz.astype(float).copy()
    cx, cy, cz = nx / 2.0, ny / 2.0, nz / 2.0

    def axis_pos(
        idx: int, n: int, neg_empty: bool, pos_empty: bool, center: float
    ) -> float:
        if pos_empty and not neg_empty:
            return float(idx + 1)
        if neg_empty and not pos_empty:
            return float(idx)
        if pos_empty and neg_empty:
            return float(idx if idx < center else idx + 1)
        return float(idx + 0.5)

    for i, p in enumerate(keypoints_xyz):
        x, y, z = (
            int(round(float(p[0]))),
            int(round(float(p[1]))),
            int(round(float(p[2]))),
        )
        if not (0 <= x < nx and 0 <= y < ny and 0 <= z < nz):
            continue
        if not vol[z, y, x]:
            # Keep empty-space maxima centered on the grid cell for readability.
            out[i, :] = np.array([x + 0.5, y + 0.5, z + 0.5], dtype=float)
            continue

        x_neg_empty = x == 0 or not vol[z, y, x - 1]
        x_pos_empty = x == nx - 1 or not vol[z, y, x + 1]
        y_neg_empty = y == 0 or not vol[z, y - 1, x]
        y_pos_empty = y == ny - 1 or not vol[z, y + 1, x]
        z_neg_empty = z == 0 or not vol[z - 1, y, x]
        z_pos_empty = z == nz - 1 or not vol[z + 1, y, x]

        out[i, 0] = axis_pos(x, nx, x_neg_empty, x_pos_empty, cx)
        out[i, 1] = axis_pos(y, ny, y_neg_empty, y_pos_empty, cy)
        out[i, 2] = axis_pos(z, nz, z_neg_empty, z_pos_empty, cz)

    return out


def plot_voxels(
    volumes: list,
    titles: Optional[list] = None,
    keypoints_list: Optional[list] = None,
    show: bool = True,
    save_path: Optional[str] = None,
    surface_snap_keypoints: bool = True,
):
    """
    Display one or more voxel grids side-by-side.

    Parameters
    ----------
    volumes : list of ndarray, each (D, H, W) bool/float
    titles  : optional list of subplot titles
    keypoints_list : optional list of (N, 3) arrays; pass None per entry to skip
    surface_snap_keypoints : if True, occupied-voxel keypoints are displayed on
        neighboring exterior voxel corners for readability. If False, raw
        detector coordinates are drawn.
    """
    n = len(volumes)
    fig = plt.figure(figsize=(5 * n, 5))
    for i, vol in enumerate(volumes):
        ax = cast(Any, fig.add_subplot(1, n, i + 1, projection="3d"))

        # Volumes are stored as (z, y, x), while matplotlib.voxels expects (x, y, z).
        mask_xyz = np.transpose(vol.astype(bool), (2, 1, 0))
        ax.voxels(mask_xyz, facecolors="steelblue", edgecolors=None, alpha=0.5)

        if keypoints_list is not None and keypoints_list[i] is not None:
            kp = np.asarray(keypoints_list[i])
            if kp.ndim == 2 and kp.shape[0] > 0:
                if surface_snap_keypoints:
                    kp_plot = _shift_keypoints_to_surface_corners(vol, kp[:, :3])
                else:
                    kp_plot = kp[:, :3].astype(float)
                ax.scatter(
                    kp_plot[:, 0],
                    kp_plot[:, 1],
                    kp_plot[:, 2],
                    c="red",
                    edgecolors="black",
                    linewidths=0.4,
                    s=44,
                    depthshade=False,
                    zorder=5,
                )

        if titles is not None and i < len(titles):
            ax.set_title(titles[i])

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_pointcloud(
    pts: list,
    titles: Optional[list] = None,
    keypoints_list: Optional[list] = None,
    show: bool = True,
    save_path: Optional[str] = None,
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
    if save_path is not None:
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_voxel_slices(
    volume: np.ndarray,
    keypoints: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None,
):
    vol = np.asarray(volume)
    if vol.ndim != 3:
        raise ValueError("volume must be 3D")

    nx = vol.shape[2]
    ny = vol.shape[1]
    nz = vol.shape[0]
    cx, cy, cz = nx // 2, ny // 2, nz // 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(vol[cz, :, :], cmap="gray", origin="lower")
    axes[0].set_title(f"z = {cz}")
    axes[1].imshow(vol[:, cy, :], cmap="gray", origin="lower")
    axes[1].set_title(f"y = {cy}")
    axes[2].imshow(vol[:, :, cx], cmap="gray", origin="lower")
    axes[2].set_title(f"x = {cx}")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    if keypoints is not None and len(keypoints) > 0:
        kp = np.asarray(keypoints)
        if kp.ndim == 2 and kp.shape[1] >= 3:
            x, y, z = kp[:, 0], kp[:, 1], kp[:, 2]
            on_z = np.abs(z - cz) <= 1
            on_y = np.abs(y - cy) <= 1
            on_x = np.abs(x - cx) <= 1
            axes[0].scatter(x[on_z], y[on_z], s=14, c="red")
            axes[1].scatter(x[on_y], z[on_y], s=14, c="red")
            axes[2].scatter(y[on_x], z[on_x], s=14, c="red")

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_keypoints_voxel(volume, keypoints: np.ndarray):
    """Display a voxel and keypoints in both volume and orthogonal slices."""
    plot_voxels([volume], keypoints_list=[keypoints])
    plot_voxel_slices(volume, keypoints=keypoints)


def plot_keypoints_pc(points, keypoints):
    plot_pointcloud([points], keypoints_list=[keypoints])
