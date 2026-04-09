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


def view_extrema_blobs_3d_napari(
    original_volume: np.ndarray,
    blob_labels: np.ndarray,
    centers: np.ndarray,
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

    viewer.scale_bar.visible = True
    viewer.text_overlay.visible = True
    viewer.text_overlay.text = (
        "Blob overlays represent scale-dependent extrema support."
    )
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
