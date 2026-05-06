from __future__ import annotations

import importlib
import inspect
from typing import Any, Optional, cast

import numpy as np
from matplotlib import cm, colors
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.widgets import Slider
from scipy.ndimage import map_coordinates
from skimage.measure import marching_cubes


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
    Display one or more voxel grids side-by-side using marching cubes for fast rendering.

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

        # Use marching cubes for fast surface mesh extraction then trisurf rendering
        vol_bool = np.asarray(vol).astype(bool)
        if vol_bool.any():
            # Pad volume so surface is closed at boundaries
            padded = np.pad(
                vol_bool.astype(np.float32), 1, mode="constant", constant_values=0
            )
            # Extract surface mesh; marching_cubes returns (verts, faces, normals, values)
            # in (z,y,x) space
            verts, faces, _, _ = marching_cubes(padded, level=0.5)
            # Undo padding offset; coords are now in original (z,y,x) space
            verts -= 1
            # marching_cubes returns (z,y,x); convert to (x,y,z) for matplotlib 3D
            # verts_zyx = verts, we want verts_xyz = [[x,y,z], ...]
            # so x = verts[:, 2], y = verts[:, 1], z = verts[:, 0]
            ax.plot_trisurf(
                verts[:, 2],
                verts[:, 1],
                faces,
                verts[:, 0],
                color="steelblue",
                alpha=0.5,
                linewidth=0,
            )

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
                    marker="+",
                    s=200,
                    linewidths=2.5,
                    depthshade=False,
                    zorder=10,
                )

        # Set consistent 45° isometric-ish view
        ax.view_init(elev=30, azim=45)

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
    keypoint_scores_list: Optional[list] = None,
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
            ax.scatter(
                kp[:, 0],
                kp[:, 1],
                kp[:, 2],
                c="red",
                marker="+",
                s=300,
                linewidths=2.5,
                depthshade=False,
                zorder=10,
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


def plot_gaussian_scale_space(
    gaussian_pyramid: list[list[np.ndarray]],
    sigma_pyramid: list[list[float]],
):
    rows = len(gaussian_pyramid)
    cols = max((len(o) for o in gaussian_pyramid), default=0)
    if rows == 0 or cols == 0:
        return

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.5 * rows), squeeze=False)
    for octave_idx, octave_imgs in enumerate(gaussian_pyramid):
        for scale_idx in range(cols):
            ax = axes[octave_idx][scale_idx]
            ax.axis("off")
            if scale_idx < len(octave_imgs):
                sigma = sigma_pyramid[octave_idx][scale_idx]
                ax.imshow(octave_imgs[scale_idx], cmap="gray")
                ax.set_title(
                    f"O{octave_idx} S{scale_idx} sigma={sigma:.2f}", fontsize=9
                )
    fig.suptitle("Gaussian Scale Space", fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_dog_scale_space(dog_pyramid: list[list[np.ndarray]]):
    rows = len(dog_pyramid)
    cols = max((len(o) for o in dog_pyramid), default=0)
    if rows == 0 or cols == 0:
        return

    max_abs = 0.0
    for octave_imgs in dog_pyramid:
        for img in octave_imgs:
            max_abs = max(max_abs, float(np.max(np.abs(img))))
    if max_abs <= 0:
        max_abs = 1.0

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.5 * rows), squeeze=False)
    for octave_idx, octave_imgs in enumerate(dog_pyramid):
        for scale_idx in range(cols):
            ax = axes[octave_idx][scale_idx]
            ax.axis("off")
            if scale_idx < len(octave_imgs):
                ax.imshow(
                    octave_imgs[scale_idx], cmap="seismic", vmin=-max_abs, vmax=max_abs
                )
                ax.set_title(f"O{octave_idx} DoG {scale_idx}", fontsize=9)
    fig.suptitle("Difference of Gaussians", fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_gaussian_scale_space_3d(
    gaussian_pyramid: list[list[np.ndarray]],
    sigma_pyramid: list[list[float]],
    slice_axis: int = 0,
    slice_index: Optional[int] = None,
):
    rows = len(gaussian_pyramid)
    cols = max((len(o) for o in gaussian_pyramid), default=0)
    if rows == 0 or cols == 0:
        return

    axis = int(np.clip(slice_axis, 0, 2))
    fig, axes = plt.subplots(
        rows, cols, figsize=(3.4 * cols, 2.8 * rows), squeeze=False
    )
    for octave_idx, octave_vols in enumerate(gaussian_pyramid):
        for scale_idx in range(cols):
            ax = axes[octave_idx][scale_idx]
            ax.axis("off")
            if scale_idx >= len(octave_vols):
                continue

            vol = octave_vols[scale_idx]
            sigma = sigma_pyramid[octave_idx][scale_idx]
            if slice_index is None:
                idx = vol.shape[axis] // 2
            else:
                idx = int(np.clip(slice_index, 0, vol.shape[axis] - 1))

            if axis == 0:
                slice_img = vol[idx, :, :]
            elif axis == 1:
                slice_img = vol[:, idx, :]
            else:
                slice_img = vol[:, :, idx]

            ax.imshow(slice_img, cmap="gray")
            ax.set_title(
                f"O{octave_idx} S{scale_idx} sigma={sigma:.2f}\n{vol.shape}",
                fontsize=8,
            )

    fig.suptitle(f"3D Gaussian Pyramid Slices (axis={axis})", fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_gaussian_scale_space_3d_interactive(
    gaussian_pyramid: list[list[np.ndarray]],
    sigma_pyramid: list[list[float]],
    slice_axis: int = 0,
):
    rows = len(gaussian_pyramid)
    cols = max((len(o) for o in gaussian_pyramid), default=0)
    if rows == 0 or cols == 0:
        return

    axis = int(np.clip(slice_axis, 0, 2))

    def get_slice(vol: np.ndarray, idx: int) -> np.ndarray:
        if axis == 0:
            return vol[idx, :, :]
        if axis == 1:
            return vol[:, idx, :]
        return vol[:, :, idx]

    octave_idx = 0
    scale_idx = 0
    vol = gaussian_pyramid[octave_idx][scale_idx]
    max_slice = vol.shape[axis] - 1
    slice_idx = max_slice // 2

    fig, ax = plt.subplots(figsize=(8, 7))
    plt.subplots_adjust(bottom=0.28)
    image_artist = ax.imshow(get_slice(vol, slice_idx), cmap="gray")
    ax.axis("off")

    ax_octave = fig.add_axes((0.15, 0.18, 0.7, 0.03))
    ax_scale = fig.add_axes((0.15, 0.13, 0.7, 0.03))
    ax_slice = fig.add_axes((0.15, 0.08, 0.7, 0.03))

    octave_slider = Slider(
        ax=ax_octave,
        label="Octave",
        valmin=0,
        valmax=max(0, rows - 1),
        valinit=0,
        valstep=1,
    )
    scale_slider = Slider(
        ax=ax_scale,
        label="Sigma Level",
        valmin=0,
        valmax=max(0, cols - 1),
        valinit=0,
        valstep=1,
    )
    slice_slider = Slider(
        ax=ax_slice,
        label=f"Slice (axis={axis})",
        valmin=0,
        valmax=max(0, max_slice),
        valinit=slice_idx,
        valstep=1,
    )

    def update(_val):
        o_idx = int(octave_slider.val)
        s_idx = int(scale_slider.val)
        s_idx = min(s_idx, len(gaussian_pyramid[o_idx]) - 1)

        current_vol = gaussian_pyramid[o_idx][s_idx]
        current_sigma = sigma_pyramid[o_idx][s_idx]
        current_max_slice = current_vol.shape[axis] - 1

        slice_slider.valmax = max(0, current_max_slice)
        slice_slider.ax.set_xlim(slice_slider.valmin, slice_slider.valmax)
        c_idx = int(min(slice_slider.val, current_max_slice))
        if c_idx != int(slice_slider.val):
            slice_slider.set_val(c_idx)
            return

        current_slice = get_slice(current_vol, c_idx)
        image_artist.set_data(current_slice)
        image_artist.set_clim(
            float(np.min(current_slice)), float(np.max(current_slice))
        )
        ax.set_title(
            f"Octave {o_idx}, Sigma Level {s_idx}, sigma={current_sigma:.3f}, "
            f"slice={c_idx}, vol={current_vol.shape}"
        )
        fig.canvas.draw_idle()

    octave_slider.on_changed(update)
    scale_slider.on_changed(update)
    slice_slider.on_changed(update)
    update(None)
    plt.show()


def _pad_center_to_shape(
    volume: np.ndarray, target_shape: tuple[int, int, int]
) -> np.ndarray:
    padded = np.zeros(target_shape, dtype=np.float32)
    z, y, x = volume.shape
    tz, ty, tx = target_shape
    z0 = max(0, (tz - z) // 2)
    y0 = max(0, (ty - y) // 2)
    x0 = max(0, (tx - x) // 2)
    padded[z0 : z0 + z, y0 : y0 + y, x0 : x0 + x] = volume.astype(np.float32)
    return padded


def view_gaussian_scale_space_3d_napari(
    gaussian_pyramid: list[list[np.ndarray]],
    sigma_pyramid: list[list[float]],
):
    try:
        napari = importlib.import_module("napari")
    except ImportError as exc:
        raise ImportError(
            "napari is not installed. Install with: uv pip install napari pyqt5"
        ) from exc

    num_octaves = len(gaussian_pyramid)
    max_scales = max((len(octave) for octave in gaussian_pyramid), default=0)
    if num_octaves == 0 or max_scales == 0:
        return

    max_z = max(vol.shape[0] for octave in gaussian_pyramid for vol in octave)
    max_y = max(vol.shape[1] for octave in gaussian_pyramid for vol in octave)
    max_x = max(vol.shape[2] for octave in gaussian_pyramid for vol in octave)

    stacked = np.zeros(
        (num_octaves, max_scales, max_z, max_y, max_x),
        dtype=np.float32,
    )

    for o_idx, octave in enumerate(gaussian_pyramid):
        for s_idx, vol in enumerate(octave):
            stacked[o_idx, s_idx] = _pad_center_to_shape(vol, (max_z, max_y, max_x))

    viewer = napari.Viewer(ndisplay=3)
    layer = viewer.add_image(
        stacked,
        name="SIFT3D Gaussian Pyramid",
        rendering="mip",
        colormap="gray",
        depiction="volume",
    )

    layer.metadata["sigma_pyramid"] = sigma_pyramid
    viewer.dims.axis_labels = ("octave", "sigma_level", "z", "y", "x")
    viewer.scale_bar.visible = True
    viewer.text_overlay.visible = True
    viewer.text_overlay.text = (
        "Use sliders for octave/sigma_level. Rotate/zoom for true volumetric 3D."
    )
    napari.run()


def plot_dog_scale_space_3d(
    dog_pyramid: list[list[np.ndarray]],
    dog_sigma_pairs: list[list[tuple[float, float, float]]],
    slice_axis: int = 0,
    slice_index: Optional[int] = None,
):
    rows = len(dog_pyramid)
    cols = max((len(o) for o in dog_pyramid), default=0)
    if rows == 0 or cols == 0:
        return

    axis = int(np.clip(slice_axis, 0, 2))
    max_abs = 0.0
    for octave in dog_pyramid:
        for vol in octave:
            max_abs = max(max_abs, float(np.max(np.abs(vol))))
    if max_abs <= 0.0:
        max_abs = 1.0

    fig, axes = plt.subplots(
        rows, cols, figsize=(3.4 * cols, 2.8 * rows), squeeze=False
    )
    for octave_idx, octave_dogs in enumerate(dog_pyramid):
        for dog_idx in range(cols):
            ax = axes[octave_idx][dog_idx]
            ax.axis("off")
            if dog_idx >= len(octave_dogs):
                continue

            vol = octave_dogs[dog_idx]
            sigma_low, sigma_high, _delta = dog_sigma_pairs[octave_idx][dog_idx]
            if slice_index is None:
                idx = vol.shape[axis] // 2
            else:
                idx = int(np.clip(slice_index, 0, vol.shape[axis] - 1))

            if axis == 0:
                slice_img = vol[idx, :, :]
            elif axis == 1:
                slice_img = vol[:, idx, :]
            else:
                slice_img = vol[:, :, idx]

            ax.imshow(slice_img, cmap="seismic", vmin=-max_abs, vmax=max_abs)
            ax.set_title(
                f"O{octave_idx} DoG{dog_idx}\n{sigma_low:.2f}->{sigma_high:.2f}",
                fontsize=8,
            )

    fig.suptitle(f"3D DoG Pyramid Slices (axis={axis})", fontsize=12)
    plt.tight_layout()
    plt.show()


def view_dog_scale_space_3d_napari(
    dog_pyramid: list[list[np.ndarray]],
    dog_sigma_pairs: list[list[tuple[float, float, float]]],
):
    try:
        napari = importlib.import_module("napari")
    except ImportError as exc:
        raise ImportError(
            "napari is not installed. Install with: uv pip install napari pyqt5"
        ) from exc

    num_octaves = len(dog_pyramid)
    max_dog_levels = max((len(octave) for octave in dog_pyramid), default=0)
    if num_octaves == 0 or max_dog_levels == 0:
        return

    max_z = max(vol.shape[0] for octave in dog_pyramid for vol in octave)
    max_y = max(vol.shape[1] for octave in dog_pyramid for vol in octave)
    max_x = max(vol.shape[2] for octave in dog_pyramid for vol in octave)

    stacked = np.zeros(
        (num_octaves, max_dog_levels, max_z, max_y, max_x),
        dtype=np.float32,
    )

    for o_idx, octave in enumerate(dog_pyramid):
        for d_idx, vol in enumerate(octave):
            stacked[o_idx, d_idx] = _pad_center_to_shape(vol, (max_z, max_y, max_x))

    max_abs = float(np.percentile(np.abs(stacked), 99.5))
    if max_abs <= 0.0:
        max_abs = 1.0

    viewer = napari.Viewer(ndisplay=3)
    layer = viewer.add_image(
        stacked,
        name="SIFT3D DoG Pyramid",
        rendering="mip",
        colormap="gray",
        depiction="volume",
        contrast_limits=(-max_abs, max_abs),
    )
    layer.metadata["dog_sigma_pairs"] = dog_sigma_pairs
    viewer.dims.axis_labels = ("octave", "dog_level", "z", "y", "x")
    viewer.scale_bar.visible = True
    viewer.text_overlay.visible = True
    viewer.text_overlay.text = (
        "Use sliders for octave/dog_level. Contrast is symmetric around zero."
    )
    napari.run()


def rasterize_extrema_blobs_3d(
    volume_shape: tuple[int, int, int],
    extrema_global: np.ndarray,
    radius_factor: float = 1.0,
    max_blobs: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.zeros(volume_shape, dtype=np.int32)
    if extrema_global.size == 0:
        return labels, np.empty((0, 3), dtype=np.float32)

    data = extrema_global
    if max_blobs is not None and data.shape[0] > max_blobs:
        idx = np.argsort(np.abs(data[:, 4]))[::-1][:max_blobs]
        data = data[idx]

    centers: list[tuple[float, float, float]] = []
    z_max, y_max, x_max = volume_shape
    for i, row in enumerate(data, start=1):
        zc, yc, xc = float(row[0]), float(row[1]), float(row[2])
        sigma = float(row[3])
        radius = max(1, int(round(radius_factor * np.sqrt(2.0) * sigma)))

        z0 = max(0, int(np.floor(zc - radius)))
        z1 = min(z_max - 1, int(np.ceil(zc + radius)))
        y0 = max(0, int(np.floor(yc - radius)))
        y1 = min(y_max - 1, int(np.ceil(yc + radius)))
        x0 = max(0, int(np.floor(xc - radius)))
        x1 = min(x_max - 1, int(np.ceil(xc + radius)))

        zz, yy, xx = np.ogrid[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1]
        mask = (zz - zc) ** 2 + (yy - yc) ** 2 + (xx - xc) ** 2 <= radius**2
        labels[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1][mask] = i
        centers.append((zc, yc, xc))

    return labels, np.asarray(centers, dtype=np.float32)


def _compute_extrema_gradient_vectors_3d(
    original_volume: np.ndarray,
    zyx: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    gradient = np.gradient(original_volume.astype(np.float32))
    grad_z = np.asarray(gradient[0], dtype=np.float32)
    grad_y = np.asarray(gradient[1], dtype=np.float32)
    grad_x = np.asarray(gradient[2], dtype=np.float32)
    center_grad = np.column_stack(
        [
            map_coordinates(
                grad_z,
                [zyx[:, 0], zyx[:, 1], zyx[:, 2]],
                order=1,
                mode="nearest",
            ),
            map_coordinates(
                grad_y,
                [zyx[:, 0], zyx[:, 1], zyx[:, 2]],
                order=1,
                mode="nearest",
            ),
            map_coordinates(
                grad_x,
                [zyx[:, 0], zyx[:, 1], zyx[:, 2]],
                order=1,
                mode="nearest",
            ),
        ]
    ).astype(np.float32)

    vectors = np.zeros_like(center_grad, dtype=np.float32)
    for i, (row, sig) in enumerate(zip(zyx, sigma)):
        direction = center_grad[i]
        mag = float(np.linalg.norm(direction))

        if mag <= 1e-6:
            zc, yc, xc = (
                int(round(float(row[0]))),
                int(round(float(row[1]))),
                int(round(float(row[2]))),
            )
            search_radius = max(1, int(round(np.sqrt(2.0) * float(sig))))

            z0 = max(0, zc - search_radius)
            z1 = min(original_volume.shape[0] - 1, zc + search_radius)
            y0 = max(0, yc - search_radius)
            y1 = min(original_volume.shape[1] - 1, yc + search_radius)
            x0 = max(0, xc - search_radius)
            x1 = min(original_volume.shape[2] - 1, xc + search_radius)

            gz_patch = grad_z[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1]
            gy_patch = grad_y[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1]
            gx_patch = grad_x[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1]
            gmag_patch = np.sqrt(gz_patch**2 + gy_patch**2 + gx_patch**2)

            if gmag_patch.size > 0:
                best_idx = np.unravel_index(
                    int(np.argmax(gmag_patch)), gmag_patch.shape
                )
                direction = np.array(
                    [
                        float(gz_patch[best_idx]),
                        float(gy_patch[best_idx]),
                        float(gx_patch[best_idx]),
                    ],
                    dtype=np.float32,
                )
                mag = float(np.linalg.norm(direction))

        if mag <= 1e-6:
            direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            mag = 1.0

        direction = direction / mag
        arrow_length = max(1.5, np.sqrt(2.0) * float(sig) * 0.75)
        vectors[i] = direction * arrow_length

    return vectors


def plot_extrema_blobs_voxel(
    original_volume: np.ndarray,
    blob_labels: np.ndarray,
):
    fig = plt.figure(figsize=(10, 5))
    ax = cast(Any, fig.add_subplot(111, projection="3d"))

    base = original_volume.astype(bool)
    blobs = blob_labels > 0

    ax.voxels(base, facecolors="steelblue", edgecolors=None, alpha=0.25)
    ax.voxels(blobs, facecolors="orangered", edgecolors=None, alpha=0.85)
    ax.set_title("3D Extrema Blobs on Target Shape")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show()


def plot_extrema_circles_3d(
    original_volume: np.ndarray,
    extrema_global: np.ndarray,
    sigma_scale: float = 1.4142,
) -> None:
    """Show keypoints as scale-proportional circle outlines on the three
    orthogonal centre slices, matching the 2-D SIFT overlay style.

    extrema_global columns: z, y, x, sigma_char, response, octave, rs
    """
    D, H, W = original_volume.shape
    # Each view: (axis_col_in_zyx, slice_idx, horiz_col, vert_col, axis_label, h_label, v_label)
    views = [
        (0, D // 2, 2, 1, "Z", "X", "Y"),
        (1, H // 2, 2, 0, "Y", "X", "Z"),
        (2, W // 2, 1, 0, "X", "Y", "Z"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Keypoints — circle radius ∝ σ", fontsize=11)

    has_data = extrema_global.size > 0
    if has_data:
        data = np.asarray(extrema_global, dtype=np.float32)
        zyx = data[:, :3]
        global_sigma = data[:, 3] * (2.0 ** data[:, 5])
        radii = sigma_scale * global_sigma
        responses = data[:, 4]
        max_abs = float(np.max(np.abs(responses))) or 1e-6
        norm = colors.Normalize(vmin=-max_abs, vmax=max_abs)
        cmap = cm.get_cmap("RdBu_r")

    for ax, (ax_col, sl, h_col, v_col, ax_name, h_label, v_label) in zip(axes, views):
        if ax_col == 0:
            img = original_volume[sl, :, :]
        elif ax_col == 1:
            img = original_volume[:, sl, :]
        else:
            img = original_volume[:, :, sl]

        ax.imshow(img, cmap="gray", origin="upper", aspect="equal")
        ax.set_title(f"{ax_name} = {sl}", fontsize=9)
        ax.set_xlabel(h_label)
        ax.set_ylabel(v_label)

        if has_data:
            near = np.abs(zyx[:, ax_col] - sl) <= global_sigma
            for h, v, r, resp in zip(
                zyx[near, h_col], zyx[near, v_col], radii[near], responses[near]
            ):
                ax.add_patch(
                    Circle(
                        (float(h), float(v)),
                        radius=float(r),
                        fill=False,
                        edgecolor=cmap(norm(float(resp))),
                        linewidth=1.2,
                        alpha=0.85,
                    )
                )

    if has_data:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=axes[-1], shrink=0.75, label="DoG response")

    plt.tight_layout()
    plt.show()


def plot_extrema_gradient_overlay_3d(
    original_volume: np.ndarray,
    extrema_global: np.ndarray,
    radius_factor: float = 1.0,
    max_blobs: Optional[int] = None,
):
    fig = plt.figure(figsize=(11, 6))
    ax = cast(Any, fig.add_subplot(111, projection="3d"))

    base = original_volume.astype(bool)
    ax.voxels(base, facecolors="steelblue", edgecolors=None, alpha=0.2)

    if extrema_global.size == 0:
        ax.set_title("3D Extrema Gradient Overlay")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.tight_layout()
        plt.show()
        return

    data = np.asarray(extrema_global, dtype=np.float32)
    if max_blobs is not None and data.shape[0] > max_blobs:
        idx = np.argsort(np.abs(data[:, 4]))[::-1][:max_blobs]
        data = data[idx]

    blob_labels, _ = rasterize_extrema_blobs_3d(
        original_volume.shape,
        data,
        radius_factor=radius_factor,
        max_blobs=None,
    )

    zyx = data[:, :3]
    sigma = data[:, 3]
    response = data[:, 4]

    ax.voxels(blob_labels > 0, facecolors="orangered", edgecolors=None, alpha=0.45)

    vectors = _compute_extrema_gradient_vectors_3d(original_volume, zyx, sigma)

    max_abs = float(np.max(np.abs(response))) if response.size else 1.0
    if max_abs <= 0.0:
        max_abs = 1.0
    cmap = cm.get_cmap("plasma")
    norm = colors.Normalize(vmin=-max_abs, vmax=max_abs)

    for row, resp, vector in zip(zyx, response, vectors):
        zc, yc, xc = float(row[0]), float(row[1]), float(row[2])
        color = cmap(norm(float(resp)))

        dz, dy, dx = float(vector[0]), float(vector[1]), float(vector[2])
        ax.quiver(
            xc,
            yc,
            zc,
            dx,
            dy,
            dz,
            color=color,
            linewidth=1.2,
            arrow_length_ratio=0.2,
            alpha=0.95,
        )

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.72, label="DoG response")
    ax.set_title("3D Extrema Blobs with Gradient Arrows")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show()


def view_extrema_blobs_3d_napari(
    original_volume: np.ndarray,
    blob_labels: np.ndarray,
    centers: np.ndarray,
    extrema_global: Optional[np.ndarray] = None,
):
    try:
        napari = importlib.import_module("napari")
    except ImportError as exc:
        raise ImportError(
            "napari is not installed. Install with: uv pip install napari pyqt5"
        ) from exc

    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(
        original_volume.astype(np.float32),
        name="Target Volume",
        rendering="mip",
        colormap="gray",
        depiction="volume",
    )
    viewer.add_labels(blob_labels, name="Extrema Blobs", opacity=0.55)
    if centers.size > 0:
        points_kwargs: dict[str, Any] = {
            "data": centers,
            "name": "Extrema Centers",
            "size": 4,
            "face_color": "yellow",
        }
        add_points_params = inspect.signature(viewer.add_points).parameters
        if "edge_color" in add_points_params:
            points_kwargs["edge_color"] = "black"
        elif "border_color" in add_points_params:
            points_kwargs["border_color"] = "black"
        viewer.add_points(**points_kwargs)

    if extrema_global is not None and extrema_global.size > 0:
        extrema = np.asarray(extrema_global, dtype=np.float32)
        zyx = extrema[:, :3]
        sigma = extrema[:, 3]
        vectors = _compute_extrema_gradient_vectors_3d(original_volume, zyx, sigma)
        vector_data = np.stack([zyx, vectors], axis=1)

        vector_kwargs: dict[str, Any] = {
            "data": vector_data,
            "name": "Gradient Arrows",
        }
        add_vectors_params = inspect.signature(viewer.add_vectors).parameters
        if "edge_width" in add_vectors_params:
            vector_kwargs["edge_width"] = 1.1
        if "edge_color" in add_vectors_params:
            vector_kwargs["edge_color"] = "cyan"
        viewer.add_vectors(**vector_kwargs)

    viewer.scale_bar.visible = True
    viewer.text_overlay.visible = True
    viewer.text_overlay.text = "Blob overlays represent scale-dependent extrema support; arrows show local gradient direction."
    napari.run()


def plot_extrema_scale_space_3d(
    extrema_local: list[np.ndarray],
    max_points: int = 4000,
):
    points = [ext for ext in extrema_local if ext.size > 0]
    if not points:
        return

    data = np.vstack(points)
    if data.shape[0] > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(data.shape[0], size=max_points, replace=False)
        data = data[idx]

    y = data[:, 0]
    x = data[:, 1]
    s = data[:, 2] + data[:, 4] * 10.0
    response = data[:, 3]

    fig = plt.figure(figsize=(8, 6))
    ax = cast(Any, fig.add_subplot(111, projection="3d"))
    sc = ax.scatter(x, y, s, c=response, cmap="coolwarm", s=8, alpha=0.85)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Scale Index (octave-adjusted)")
    ax.set_title("Extrema in Scale Space")
    fig.colorbar(sc, ax=ax, shrink=0.7, label="DoG response")
    plt.tight_layout()
    plt.show()


def plot_extrema_overlay(
    original_image: np.ndarray,
    extrema_global: np.ndarray,
    base_sigma: float = 1.6,
    scales_per_octave: int = 5,
):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(original_image, cmap="gray")
    ax.set_title("Extrema Overlay on Original Image")
    ax.axis("off")

    if extrema_global.size > 0:
        y = extrema_global[:, 0]
        x = extrema_global[:, 1]
        dog_scale_idx = extrema_global[:, 2]
        octave_idx = extrema_global[:, 4]
        sigma = base_sigma * (2.0 ** (octave_idx + (dog_scale_idx / scales_per_octave)))
        radii = np.sqrt(2.0) * sigma

        norm = colors.Normalize(
            vmin=float(np.min(dog_scale_idx)), vmax=float(np.max(dog_scale_idx)) + 1e-8
        )
        cmap = cm.get_cmap("plasma")
        for xi, yi, ri, si in zip(x, y, radii, dog_scale_idx):
            color = cmap(norm(float(si)))
            ax.add_patch(
                Circle(
                    (float(xi), float(yi)),
                    radius=float(ri),
                    fill=False,
                    edgecolor=color,
                    linewidth=1.1,
                    alpha=0.9,
                )
            )

        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.75, label="DoG scale index")

        marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]
        unique_octaves = np.unique(octave_idx.astype(int))
        legend_handles: list[Line2D] = []
        for i, octave in enumerate(unique_octaves):
            marker = marker_cycle[i % len(marker_cycle)]
            mask = octave_idx.astype(int) == octave
            ax.scatter(
                x[mask],
                y[mask],
                c=dog_scale_idx[mask],
                cmap=cmap,
                norm=norm,
                marker=marker,
                s=16,
                alpha=0.9,
                edgecolors="black",
                linewidths=0.25,
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker,
                    color="none",
                    markerfacecolor="lightgray",
                    markeredgecolor="black",
                    markersize=6,
                    label=f"Octave {octave}",
                )
            )

        if legend_handles:
            ax.legend(handles=legend_handles, title="Octave (shape)", loc="upper right")

    plt.tight_layout()
    plt.show()


def plot_extrema_gradient_overlay(
    original_image: np.ndarray,
    orientation_signatures,
    base_sigma: float = 1.6,
    scales_per_octave: int = 5,
):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(original_image, cmap="gray")
    ax.set_title("Keypoint Gradient Overlay on Original Image")
    ax.axis("off")

    signatures = [sig for sig in orientation_signatures if sig is not None]
    if signatures:
        scale_indices = np.array(
            [float(sig.scale_index) for sig in signatures], dtype=np.float32
        )
        norm = colors.Normalize(
            vmin=float(np.min(scale_indices)), vmax=float(np.max(scale_indices)) + 1e-8
        )
        cmap = cm.get_cmap("plasma")

        for signature in signatures:
            octave = float(signature.octave_index)
            scale_index = float(signature.scale_index)
            center_yx = np.asarray(signature.center_yx, dtype=np.float32) * (
                2.0**octave
            )
            y = float(center_yx[0])
            x = float(center_yx[1])

            sigma = float(signature.sigma) * (2.0**octave)
            radius = max(1.0, np.sqrt(2.0) * sigma)
            arrow_length = max(3.0, 0.9 * np.sqrt(2.0) * sigma)
            theta = float(signature.dominant_orientation)
            dx = float(np.cos(theta) * arrow_length)
            dy = float(np.sin(theta) * arrow_length)
            color = cmap(norm(scale_index))

            ax.add_patch(
                Circle(
                    (x, y),
                    radius=radius,
                    fill=False,
                    edgecolor=color,
                    linewidth=1.2,
                    alpha=0.85,
                )
            )

            # ax.quiver(
            #     x,
            #     y,
            #     dx,
            #     dy,
            #     color=[color],
            #     angles="xy",
            #     scale_units="xy",
            #     scale=1.0,
            #     width=0.0045,
            #     headwidth=4.5,
            #     headlength=6.0,
            #     headaxislength=5.2,
            #     alpha=0.95,
            # )

        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.75, label="DoG scale index")

        marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]
        unique_octaves = np.unique([int(sig.octave_index) for sig in signatures])
        legend_handles: list[Line2D] = []
        for i, octave in enumerate(unique_octaves):
            marker = marker_cycle[i % len(marker_cycle)]
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker,
                    color="none",
                    markerfacecolor="lightgray",
                    markeredgecolor="black",
                    markersize=6,
                    label=f"Octave {octave}",
                )
            )

        if legend_handles:
            ax.legend(handles=legend_handles, title="Octave (shape)", loc="upper right")

    plt.tight_layout()
    plt.show()


def plot_sift2d_orientation_views(original_image: np.ndarray, signature) -> None:
    image = np.asarray(original_image, dtype=np.float32)
    if image.ndim != 2 or image.size == 0:
        return

    center_y = float(signature.center_yx[0])
    center_x = float(signature.center_yx[1])

    sample_points = np.asarray(signature.sample_points, dtype=np.float32)
    gradient_vectors = np.asarray(signature.gradient_vectors, dtype=np.float32)
    magnitudes = np.asarray(signature.magnitudes, dtype=np.float32)
    histogram = np.asarray(signature.histogram, dtype=np.float32)
    bin_edges = np.asarray(signature.bin_edges, dtype=np.float32)

    fig_grad, ax_grad = plt.subplots(figsize=(7, 6))
    ax_grad.imshow(image, cmap="gray", origin="upper")
    ax_grad.set_title(
        f"Gradient samples on image for keypoint (o={signature.octave_index}, s={signature.scale_index})"
    )
    ax_grad.set_xlabel("x")
    ax_grad.set_ylabel("y")

    window_radius = max(1, int(round(3.0 * float(signature.sigma))))
    x0 = max(0, int(round(center_x)) - window_radius)
    x1 = min(image.shape[1] - 1, int(round(center_x)) + window_radius)
    y0 = max(0, int(round(center_y)) - window_radius)
    y1 = min(image.shape[0] - 1, int(round(center_y)) + window_radius)
    ax_grad.set_xlim(x0 - 0.5, x1 + 0.5)
    ax_grad.set_ylim(y1 + 0.5, y0 - 0.5)

    if sample_points.size > 0 and gradient_vectors.size > 0:
        point_x = sample_points[:, 1]
        point_y = sample_points[:, 0]
        max_mag = float(np.max(magnitudes)) if magnitudes.size else 1.0
        max_mag = max(max_mag, 1e-6)
        vec_x = gradient_vectors[:, 1] / max_mag * 0.85
        vec_y = gradient_vectors[:, 0] / max_mag * 0.85
        ax_grad.quiver(
            point_x,
            point_y,
            vec_x,
            vec_y,
            magnitudes,
            cmap="viridis",
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.004,
            alpha=0.9,
        )

    ax_grad.scatter(
        [center_x],
        [center_y],
        c="red",
        s=40,
        marker="x",
        label="keypoint",
    )
    ax_grad.legend(loc="upper right")

    fig_hist, ax_hist = plt.subplots(figsize=(7, 5))
    if histogram.size > 0 and bin_edges.size == histogram.size + 1:
        widths = np.diff(bin_edges)
        centers = bin_edges[:-1] + widths / 2.0
        centers_deg = np.degrees(centers)
        widths_deg = np.degrees(widths)
        ax_hist.bar(
            centers_deg,
            histogram,
            width=widths_deg,
            align="center",
            color="#4c78a8",
            edgecolor="black",
            alpha=0.9,
        )
        dominant_deg = float(np.degrees(signature.dominant_orientation))
        ax_hist.axvline(dominant_deg, color="crimson", linestyle="--", linewidth=1.5)
        ax_hist.text(
            dominant_deg,
            float(np.max(histogram)) if histogram.size else 0.0,
            f"  dominant={dominant_deg:.1f}°",
            color="crimson",
            va="bottom",
        )
    ax_hist.set_xlim(0.0, 360.0)
    ax_hist.set_xlabel("Orientation (degrees)")
    ax_hist.set_ylabel("Weighted gradient magnitude")
    ax_hist.set_title("SIFT2D orientation histogram signature")
    ax_hist.grid(alpha=0.2)

    fig_grad.suptitle("Local gradient samples", fontsize=12)
    fig_hist.suptitle("Orientation histogram", fontsize=12)
    plt.show()


def _rasterize_pc_values(
    pts: np.ndarray,
    values: np.ndarray,
    grid_shape: tuple[int, int, int],
    min_corner: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    """Splat per-point scalar values onto a (D, H, W) voxel grid by averaging."""
    vol = np.zeros(grid_shape, dtype=np.float32)
    count = np.zeros(grid_shape, dtype=np.int32)
    D, H, W = grid_shape
    idx = np.floor((pts - min_corner) / voxel_size).astype(int)
    ix = np.clip(idx[:, 0], 0, W - 1)
    iy = np.clip(idx[:, 1], 0, H - 1)
    iz = np.clip(idx[:, 2], 0, D - 1)
    np.add.at(vol, (iz, iy, ix), values)
    np.add.at(count, (iz, iy, ix), 1)
    nz = count > 0
    vol[nz] /= count[nz]
    return vol


def view_pc_radii_napari(
    points: np.ndarray,
    points_per_octave: list[np.ndarray],
    density_pyramid: list[list[np.ndarray]],
    radii_pyramid: list[list[float]],
    dog_pyramid: list[list[np.ndarray]],
    keypoints: Optional[np.ndarray] = None,
    signal_name: str = "KDE density",
) -> None:
    """Interactive napari viewer for the radii-based SIFT point cloud pipeline.

    Adds one Points layer per (octave, scale) for the scalar field heatmap and one
    per (octave, DoG index) for the DoG heatmap — each colored by its scalar value —
    plus a raw point cloud layer and a keypoints layer.  Toggle layers in the napari
    panel to compare octaves/scales.

    Parameters
    ----------
    points : np.ndarray
        (N, 3) original normalized point cloud.
    points_per_octave : list[np.ndarray]
        FPS-subsampled point sets per octave.
    density_pyramid : list[list[np.ndarray]]
        Per-octave, per-scale scalar field arrays (KDE density or geometry measure).
    radii_pyramid : list[list[float]]
        Corresponding query radii.
    dog_pyramid : list[list[np.ndarray]]
        Per-octave DoG arrays (one fewer than density per octave).
    keypoints : np.ndarray | None
        (K, 5) keypoints: x, y, z, radius, response.
    signal_name : str
        Human-readable name shown in layer labels and the overlay text (default:
        ``"KDE density"``).  Pass e.g. ``"λ_min geometry"`` for the geometric variant.
    """
    try:
        napari = importlib.import_module("napari")
    except ImportError as exc:
        raise ImportError(
            "napari is not installed. Install with: uv pip install napari pyqt5"
        ) from exc

    pts = np.asarray(points, dtype=np.float32)
    # napari Points use (z, y, x) ordering
    pts_zyx = pts[:, [2, 1, 0]].astype(np.float32)

    viewer = napari.Viewer(ndisplay=3)

    # Raw point cloud — small white dots for spatial context
    viewer.add_points(
        pts_zyx,
        name="Raw Point Cloud",
        size=0.008,
        face_color="white",
        opacity=0.2,
    )

    # Scalar field scale-space: one Points layer per (octave, scale)
    # Only the first layer is visible by default; toggle others in the layer panel.
    first_density = True
    short_sig = signal_name.replace(" ", "_")
    for o_idx, (octave_densities, octave_radii) in enumerate(
        zip(density_pyramid, radii_pyramid)
    ):
        oct_pts = np.asarray(points_per_octave[o_idx], dtype=np.float32)
        oct_zyx = oct_pts[:, [2, 1, 0]]
        for s_idx, (dens, r) in enumerate(zip(octave_densities, octave_radii)):
            dens = np.asarray(dens, dtype=np.float32)
            d_min, d_max = float(dens.min()), float(dens.max())
            if d_max <= d_min:
                d_max = d_min + 1.0
            viewer.add_points(
                oct_zyx,
                features={"signal": dens},
                face_color="signal",
                face_colormap="plasma",
                face_contrast_limits=(d_min, d_max),
                name=f"{short_sig} o={o_idx} s={s_idx} r={r:.3f}",
                size=0.015,
                opacity=0.85,
                visible=first_density,
            )
            first_density = False

    # DoG pyramid: one Points layer per (octave, dog index) — all hidden by default
    for o_idx, octave_dogs in enumerate(dog_pyramid):
        oct_pts = np.asarray(points_per_octave[o_idx], dtype=np.float32)
        oct_zyx = oct_pts[:, [2, 1, 0]]
        for d_idx, dog in enumerate(octave_dogs):
            dog = np.asarray(dog, dtype=np.float32)
            vmax = float(np.percentile(np.abs(dog), 99)) or 1.0
            viewer.add_points(
                oct_zyx,
                features={"dog": dog},
                face_color="dog",
                face_colormap="bwr",
                face_contrast_limits=(-vmax, vmax),
                name=f"DoG o={o_idx} d={d_idx}",
                size=0.015,
                opacity=0.85,
                visible=False,
            )

    # Keypoints — sized by detected scale radius
    if keypoints is not None and keypoints.shape[0] > 0:
        kp = np.asarray(keypoints, dtype=np.float32)
        kp_zyx = kp[:, [2, 1, 0]]
        # radius col is physical units; scale to a visible napari point size
        sizes = np.clip(kp[:, 3] * 3.0, 0.01, 0.2)

        kp_kwargs: dict[str, Any] = {
            "data": kp_zyx,
            "name": "Keypoints",
            "size": sizes,
            "face_color": "red",
            "opacity": 0.95,
        }
        add_pts_params = inspect.signature(viewer.add_points).parameters
        if "edge_color" in add_pts_params:
            kp_kwargs["edge_color"] = "white"
        elif "border_color" in add_pts_params:
            kp_kwargs["border_color"] = "white"
        viewer.add_points(**kp_kwargs)

    viewer.scale_bar.visible = True
    viewer.text_overlay.visible = True
    viewer.text_overlay.text = (
        f"{signal_name} layers: scalar field at each octave/scale (plasma = low→high).\n"
        "DoG layers: difference of Gaussians (bwr = negative→positive).\n"
        "Toggle layers in the panel to compare scales. Red = keypoints."
    )
    napari.run()


def plot_voxel_storage_layout(
    volume: np.ndarray,
    max_points: int = 3000,
):
    if volume.ndim != 3:
        raise ValueError("Expected a 3D voxel volume with shape (Z, Y, X)")

    z_dim, y_dim, x_dim = volume.shape
    z_mid = z_dim // 2
    y_mid = y_dim // 2
    x_mid = x_dim // 2

    zyx = np.argwhere(volume > 0)
    if zyx.shape[0] > max_points:
        rng = np.random.default_rng(0)
        keep = rng.choice(zyx.shape[0], size=max_points, replace=False)
        zyx = zyx[keep]

    linear_idx = np.ravel_multi_index(
        (zyx[:, 0], zyx[:, 1], zyx[:, 2]),
        dims=volume.shape,
    )

    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(volume[z_mid, :, :], cmap="gray")
    ax1.set_title(f"XY slice at z={z_mid} (volume[z, y, x])")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(volume[:, y_mid, :], cmap="gray")
    ax2.set_title(f"XZ slice at y={y_mid} (volume[z, y, x])")
    ax2.set_xlabel("x")
    ax2.set_ylabel("z")

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(volume[:, :, x_mid], cmap="gray")
    ax3.set_title(f"YZ slice at x={x_mid} (volume[z, y, x])")
    ax3.set_xlabel("y")
    ax3.set_ylabel("z")

    ax4 = cast(Any, fig.add_subplot(2, 2, 4, projection="3d"))
    if zyx.shape[0] > 0:
        sc = ax4.scatter(
            zyx[:, 2],
            zyx[:, 1],
            zyx[:, 0],
            c=linear_idx,
            cmap="viridis",
            s=10,
            alpha=0.9,
        )
        fig.colorbar(sc, ax=ax4, shrink=0.7, label="Linear memory index")
    ax4.set_title("Occupied voxels (color = flattened index)")
    ax4.set_xlabel("x")
    ax4.set_ylabel("y")
    ax4.set_zlabel("z")

    fig.suptitle(
        "Voxel Storage Layout: axis order is (z, y, x), flattened in row-major order",
        fontsize=13,
    )
    plt.tight_layout()
    plt.show()
