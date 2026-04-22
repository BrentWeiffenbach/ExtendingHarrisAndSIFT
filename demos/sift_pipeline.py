from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from src.common.io import ModelNetLoader, SyntheticVoxelLoader, load_pointcloud
from src.common.visualization import (
    plot_dog_scale_space,
    plot_dog_scale_space_3d,
    plot_extrema_gradient_overlay,
    plot_extrema_gradient_overlay_3d,
    plot_extrema_scale_space_3d,
    plot_gaussian_scale_space,
    plot_gaussian_scale_space_3d,
    plot_gaussian_scale_space_3d_interactive,
    plot_pointcloud,
    plot_sift2d_orientation_views,
    plot_voxel_storage_layout,
    plot_voxels,
    rasterize_extrema_blobs_3d,
    view_dog_scale_space_3d_napari,
    view_extrema_blobs_3d_napari,
    view_gaussian_scale_space_3d_napari,
    view_pc_radii_napari,
)
from src.pointcloud.params import SIFTGeomPCParams, SIFTRadiiPCParams, SIFTVoxelPCParams
from src.pointcloud.sift_pc import SIFTGeomPC, SIFTRadiiPC, SIFTVoxelPC
from src.voxel.params import SIFT2DParams, SIFT3DParams
from src.voxel.sift2d import SIFT2D
from src.voxel.sift3d import SIFT3DVoxel

PC_SYNTHETIC_ROOT = "data/Pointcloud/synthetic"
PC_SYNTHETIC_SHAPES = [
    "cone",
    "cube",
    "cuboid",
    "cylinder",
    "pyramid",
    "sphere",
    "torus",
]


def _load_synthetic_pc(name: str | None) -> tuple[np.ndarray, str]:
    shape = name if name in PC_SYNTHETIC_SHAPES else "sphere"
    path = Path(PC_SYNTHETIC_ROOT) / f"{shape}.ply"
    pcd = load_pointcloud(str(path))
    pts = np.asarray(pcd.points, dtype=np.float32)
    # Normalize to [0, 1] bounding box so default params work regardless of scale
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    rng = hi - lo
    rng[rng == 0] = 1.0
    pts = (pts - lo) / rng
    return pts, shape


def _plot_pc_scale_space(
    points_per_octave: list[np.ndarray],
    density_pyramid: list[list[np.ndarray]],
    radii_pyramid: list[list[float]],
    max_octaves: int = 3,
    max_scales: int = 4,
) -> None:
    n_oct = min(len(density_pyramid), max_octaves)
    n_sc = min(len(density_pyramid[0]) if density_pyramid else 0, max_scales)
    if n_oct == 0 or n_sc == 0:
        return

    fig, axes = plt.subplots(n_oct, n_sc, figsize=(3.5 * n_sc, 3 * n_oct))
    if n_oct == 1:
        axes = [axes]
    if n_sc == 1:
        axes = [[row] for row in axes]

    for o in range(n_oct):
        pts = points_per_octave[o]
        for s in range(n_sc):
            ax = axes[o][s]
            density = density_pyramid[o][s]
            r = radii_pyramid[o][s]
            sc = ax.scatter(pts[:, 0], pts[:, 1], c=density, s=2, cmap="plasma")
            plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.04)
            ax.set_title(f"Oct {o}, Scale {s}\nr={r:.3f}", fontsize=8)
            ax.set_aspect("equal")
            ax.axis("off")

    fig.suptitle("Point Cloud Density Scale-Space (Gaussian KDE)", fontsize=11)
    plt.tight_layout()
    plt.show()


def _plot_pc_dog(
    points_per_octave: list[np.ndarray],
    dog_pyramid: list[list[np.ndarray]],
    dog_radius_pairs: list[list[tuple[float, float]]],
    max_octaves: int = 3,
    max_dogs: int = 3,
) -> None:
    n_oct = min(len(dog_pyramid), max_octaves)
    n_dog = min(len(dog_pyramid[0]) if dog_pyramid else 0, max_dogs)
    if n_oct == 0 or n_dog == 0:
        return

    fig, axes = plt.subplots(n_oct, n_dog, figsize=(3.5 * n_dog, 3 * n_oct))
    if n_oct == 1:
        axes = [axes]
    if n_dog == 1:
        axes = [[row] for row in axes]

    for o in range(n_oct):
        pts = points_per_octave[o]
        for d in range(min(n_dog, len(dog_pyramid[o]))):
            ax = axes[o][d]
            dog = dog_pyramid[o][d]
            r_lo, r_hi = dog_radius_pairs[o][d]
            vmax = float(np.abs(dog).max()) or 1.0
            sc = ax.scatter(
                pts[:, 0], pts[:, 1], c=dog, s=2, cmap="RdBu_r", vmin=-vmax, vmax=vmax
            )
            plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.04)
            ax.set_title(f"Oct {o}, DoG {d}\nr=[{r_lo:.3f},{r_hi:.3f}]", fontsize=8)
            ax.set_aspect("equal")
            ax.axis("off")

    fig.suptitle("Point Cloud DoG Scale-Space", fontsize=11)
    plt.tight_layout()
    plt.show()


def _resolve_3d_demo_volume(
    default_synthetic_name: str,
    synthetic_name: str | None = None,
    modelnet_index: int | None = None,
) -> np.ndarray:
    """Resolve a 3D demo volume from synthetic or ModelNet sources."""
    if synthetic_name is not None and modelnet_index is not None:
        raise ValueError("Provide only one of synthetic_name or modelnet_index")

    if modelnet_index is not None:
        loader = ModelNetLoader("data/Voxel/real/ModelNet10-dataset/modelnet10.npy.gz")
        return loader.load_by_index(modelnet_index)

    shape_name = synthetic_name or default_synthetic_name
    return SyntheticVoxelLoader().load_by_name(shape_name)


def run_2d_demo(
    synthetic_name: str | None = None,
    modelnet_index: int | None = None,
) -> None:
    image_path = "data/2d/image_0011.jpg"
    params = SIFT2DParams(
        num_octaves=4,
        scales_per_octave=5,
        base_sigma=1.6,
        contrast_threshold=0.0,
    )
    detector = SIFT2D(params)
    result = detector.run(image_path)

    plot_gaussian_scale_space(result.gaussian_pyramid, result.sigma_pyramid)
    plot_dog_scale_space(result.dog_pyramid)
    plot_extrema_scale_space_3d(result.extrema_local, max_points=params.max_plot_points)
    plot_extrema_gradient_overlay(
        result.original_image,
        result.orientation_signatures,
        base_sigma=params.base_sigma,
        scales_per_octave=params.scales_per_octave,
    )


def run_2d_signature_demo(
    synthetic_name: str | None = None,
    modelnet_index: int | None = None,
) -> None:
    image_path = "data/2d/image_0011.jpg"
    params = SIFT2DParams(
        num_octaves=4,
        scales_per_octave=5,
        base_sigma=1.6,
        contrast_threshold=0.0,
    )
    detector = SIFT2D(params)
    result = detector.run(image_path)

    signatures = sorted(
        result.orientation_signatures,
        key=lambda signature: abs(float(signature.keypoint[3])),
        reverse=True,
    )
    if not signatures:
        return

    signature_index = int(
        np.clip(params.orientation_visual_keypoint_index, 0, len(signatures) - 1)
    )
    plot_sift2d_orientation_views(result.original_image, signatures[signature_index])


def run_3d_gaussian_demo(
    synthetic_name: str | None = None,
    modelnet_index: int | None = None,
) -> None:
    volume = _resolve_3d_demo_volume("cube", synthetic_name, modelnet_index)
    params = SIFT3DParams(
        num_octaves=4,
        scales_per_octave=5,
        base_sigma=1.2,
        min_size=8,
        downsample_factor=2,
        slice_axis=0,
    )
    detector = SIFT3DVoxel(params)
    result = detector.run(volume)
    plot_gaussian_scale_space_3d(
        result.gaussian_pyramid,
        result.sigma_pyramid,
        slice_axis=params.slice_axis,
    )


def run_3d_gaussian_interactive_demo(
    synthetic_name: str | None = None,
    modelnet_index: int | None = None,
) -> None:
    volume = _resolve_3d_demo_volume("cube", synthetic_name, modelnet_index)
    params = SIFT3DParams()
    detector = SIFT3DVoxel(params)
    result = detector.run(volume)
    view_gaussian_scale_space_3d_napari(result.gaussian_pyramid, result.sigma_pyramid)


def run_3d_gaussian_slider_demo(
    synthetic_name: str | None = None,
    modelnet_index: int | None = None,
) -> None:
    volume = _resolve_3d_demo_volume("cube", synthetic_name, modelnet_index)
    params = SIFT3DParams()
    detector = SIFT3DVoxel(params)
    result = detector.run(volume)
    plot_gaussian_scale_space_3d_interactive(
        result.gaussian_pyramid,
        result.sigma_pyramid,
        slice_axis=params.slice_axis,
    )


def run_3d_dog_demo(
    synthetic_name: str | None = None,
    modelnet_index: int | None = None,
) -> None:
    volume = _resolve_3d_demo_volume("pyramid", synthetic_name, modelnet_index)
    params = SIFT3DParams()
    detector = SIFT3DVoxel(params)
    result = detector.run(volume)
    plot_dog_scale_space_3d(
        result.dog_pyramid,
        result.dog_sigma_pairs,
        slice_axis=params.slice_axis,
    )


def run_3d_dog_napari_demo(
    synthetic_name: str | None = None,
    modelnet_index: int | None = None,
) -> None:
    volume = _resolve_3d_demo_volume("pyramid", synthetic_name, modelnet_index)
    params = SIFT3DParams(
        num_octaves=4,
        scales_per_octave=5,
        base_sigma=1.2,
        min_size=8,
        downsample_factor=2,
        slice_axis=0,
    )
    detector = SIFT3DVoxel(params)
    result = detector.run(volume)
    view_dog_scale_space_3d_napari(result.dog_pyramid, result.dog_sigma_pairs)


def run_3d_extrema_demo(
    synthetic_name: str | None = None,
    modelnet_index: int | None = None,
) -> None:
    volume = _resolve_3d_demo_volume("pyramid", synthetic_name, modelnet_index)
    params = SIFT3DParams()
    detector = SIFT3DVoxel(params)
    result = detector.run(volume)
    plot_extrema_gradient_overlay_3d(
        result.original_volume,
        result.extrema_global,
        radius_factor=params.blob_radius_factor,
        max_blobs=params.max_blob_keypoints,
    )


def run_3d_extrema_napari_demo(
    synthetic_name: str | None = None,
    modelnet_index: int | None = None,
) -> None:
    volume = _resolve_3d_demo_volume("pyramid", synthetic_name, modelnet_index)
    params = SIFT3DParams()
    detector = SIFT3DVoxel(params)
    result = detector.run(volume)
    blob_labels, centers = rasterize_extrema_blobs_3d(
        result.original_volume.shape,
        result.extrema_global,
        radius_factor=params.blob_radius_factor,
        max_blobs=params.max_blob_keypoints,
    )
    view_extrema_blobs_3d_napari(
        result.original_volume,
        blob_labels,
        centers,
        result.extrema_global,
    )


def run_voxel_storage_demo(
    synthetic_name: str | None = None,
    modelnet_index: int | None = None,
) -> None:
    volume = _resolve_3d_demo_volume("pyramid", synthetic_name, modelnet_index)
    volume = np.asarray(volume, dtype=np.float32)
    plot_voxel_storage_layout(volume)


def run_pc_radii_demo(
    synthetic_name: str | None = None,
    modelnet_index: int | None = None,
) -> None:
    """Demo: radii-based SIFT on a point cloud.

    Shows the Gaussian KDE density scale-space, the DoG pyramid, and detected keypoints.
    Use --synthetic-name to pick a shape (cone/cube/cuboid/cylinder/pyramid/sphere/torus).
    """
    pts, name = _load_synthetic_pc(synthetic_name)
    print(f"Loaded '{name}' point cloud: {pts.shape[0]} points")

    params = SIFTRadiiPCParams(
        num_octaves=3,
        scales_per_octave=4,
        base_radius=0.08,
        contrast_threshold=0.0005,
    )
    detector = SIFTRadiiPC(params)
    result = detector.run(pts)
    print(f"Detected {result.keypoints.shape[0]} keypoints")

    _plot_pc_scale_space(
        result.points_per_octave,
        result.density_pyramid,
        result.radii_pyramid,
    )

    dog_radius_pairs: list[list[tuple[float, float]]] = []
    for o, octave_dogs in enumerate(result.dog_pyramid):
        pairs: list[tuple[float, float]] = []
        radii = result.radii_pyramid[o]
        for i in range(len(octave_dogs)):
            pairs.append((radii[i], radii[i + 1]))
        dog_radius_pairs.append(pairs)

    _plot_pc_dog(
        result.points_per_octave,
        result.dog_pyramid,
        dog_radius_pairs,
    )

    kp = result.keypoints[:, :3] if result.keypoints.shape[0] > 0 else None
    plot_pointcloud(
        [pts], titles=[f"{name} — SIFTRadiiPC keypoints"], keypoints_list=[kp]
    )


def run_pc_radii_napari_demo(
    synthetic_name: str | None = None,
    modelnet_index: int | None = None,
) -> None:
    """Demo: radii-based SIFT on a point cloud — interactive napari viewer.

    Shows the Gaussian KDE density scale-space and DoG pyramid as 5D volumetric
    layers (octave × scale × z × y × x) with napari sliders, plus detected
    keypoints as 3D spheres sized by their scale radius.
    Use --synthetic-name to pick a shape (cone/cube/cuboid/cylinder/pyramid/sphere/torus).
    """
    pts, name = _load_synthetic_pc(synthetic_name)
    print(f"Loaded '{name}' point cloud: {pts.shape[0]} points")

    params = SIFTRadiiPCParams(
        num_octaves=3,
        scales_per_octave=4,
        base_radius=0.02,
        contrast_threshold=0.0005,
    )
    detector = SIFTRadiiPC(params)
    result = detector.run(pts)
    print(f"Detected {result.keypoints.shape[0]} keypoints")

    kp = result.keypoints if result.keypoints.shape[0] > 0 else None
    view_pc_radii_napari(
        points=pts,
        points_per_octave=result.points_per_octave,
        density_pyramid=result.density_pyramid,
        radii_pyramid=result.radii_pyramid,
        dog_pyramid=result.dog_pyramid,
        keypoints=kp,
    )


def run_pc_geom_demo(
    synthetic_name: str | None = None,
    modelnet_index: int | None = None,
) -> None:
    """Demo: geometry-based scale-space SIFT on a point cloud (matplotlib).

    Uses the smallest eigenvalue of the Gaussian-weighted local covariance (normalised
    by r²) as the scalar field.  Non-trivial DoG responses arise from 3-D shape
    complexity (corners, edges) rather than point density.
    Use --synthetic-name to pick a shape — try 'cube' or 'pyramid' for clear results.
    """
    pts, name = _load_synthetic_pc(synthetic_name)
    print(f"Loaded '{name}' point cloud: {pts.shape[0]} points")

    params = SIFTGeomPCParams(
        num_octaves=3,
        scales_per_octave=4,
        base_radius=0.5,
        contrast_threshold=1e-4,
    )
    detector = SIFTGeomPC(params)
    result = detector.run(pts)
    print(f"Detected {result.keypoints.shape[0]} keypoints")

    _plot_pc_scale_space(
        result.points_per_octave,
        result.density_pyramid,
        result.radii_pyramid,
    )

    dog_radius_pairs: list[list[tuple[float, float]]] = []
    for o, octave_dogs in enumerate(result.dog_pyramid):
        pairs: list[tuple[float, float]] = []
        radii = result.radii_pyramid[o]
        for i in range(len(octave_dogs)):
            pairs.append((radii[i], radii[i + 1]))
        dog_radius_pairs.append(pairs)

    _plot_pc_dog(result.points_per_octave, result.dog_pyramid, dog_radius_pairs)

    kp = result.keypoints[:, :3] if result.keypoints.shape[0] > 0 else None
    plot_pointcloud(
        [pts], titles=[f"{name} — SIFTGeomPC keypoints"], keypoints_list=[kp]
    )


def run_pc_geom_napari_demo(
    synthetic_name: str | None = None,
    modelnet_index: int | None = None,
) -> None:
    """Demo: geometry-based scale-space SIFT on a point cloud — interactive napari.

    Each (octave, scale) layer is coloured by λ_min of the local covariance (plasma
    colourmap); the DoG layers use bwr.  Compare with pc-radii-napari to see how the
    geometric signal concentrates on edges/corners while KDE density is uniform.
    Use --synthetic-name to pick a shape — try 'cube' or 'pyramid' for clear results.
    """
    pts, name = _load_synthetic_pc(synthetic_name)
    print(f"Loaded '{name}' point cloud: {pts.shape[0]} points")

    params = SIFTGeomPCParams(
        num_octaves=3,
        scales_per_octave=4,
        base_radius=1,
        contrast_threshold=1e-4,
    )
    detector = SIFTGeomPC(params)
    result = detector.run(pts)
    print(f"Detected {result.keypoints.shape[0]} keypoints")

    kp = result.keypoints if result.keypoints.shape[0] > 0 else None
    view_pc_radii_napari(
        points=pts,
        points_per_octave=result.points_per_octave,
        density_pyramid=result.density_pyramid,
        radii_pyramid=result.radii_pyramid,
        dog_pyramid=result.dog_pyramid,
        keypoints=kp,
        signal_name="λ_min geometry",
    )


def run_pc_voxel_demo(
    synthetic_name: str | None = None,
    modelnet_index: int | None = None,
) -> None:
    """Demo: voxelization-based SIFT on a point cloud.

    Voxelizes the point cloud, runs SIFT3DVoxel, then projects keypoints back to
    physical coordinates and shows both the voxel volume and the point cloud.
    Use --synthetic-name to pick a shape (cone/cube/cuboid/cylinder/pyramid/sphere/torus).
    """
    pts, name = _load_synthetic_pc(synthetic_name)
    print(f"Loaded '{name}' point cloud: {pts.shape[0]} points")

    params = SIFTVoxelPCParams(voxel_size=0.05)
    detector = SIFTVoxelPC(params)
    run_result = detector.run(pts)

    volume = run_result["volume"]
    kp_physical = run_result["keypoints"]
    sift3d_result = run_result["sift3d_result"]
    print(f"Voxelized to {volume.shape}, detected {kp_physical.shape[0]} keypoints")

    # Show DoG scale-space of the voxelized volume
    plot_dog_scale_space_3d(
        sift3d_result.dog_pyramid,
        sift3d_result.dog_sigma_pairs,
        slice_axis=0,
    )

    # Show voxelized volume (keypoints in voxel coords for overlay)
    if sift3d_result.extrema_global.shape[0] > 0:
        kp_voxel_xyz = sift3d_result.extrema_global[:, [2, 1, 0]].astype(np.int32)
    else:
        kp_voxel_xyz = np.empty((0, 3), dtype=np.int32)
    plot_voxels(
        [volume],
        titles=[f"{name} voxelized (voxel_size={params.voxel_size})"],
        keypoints_list=[kp_voxel_xyz],
    )

    # Show point cloud with physical-space keypoints
    kp_plot = kp_physical if kp_physical.shape[0] > 0 else None
    plot_pointcloud(
        [pts],
        titles=[f"{name} — SIFTVoxelPC keypoints"],
        keypoints_list=[kp_plot],
    )


DEMO_REGISTRY = {
    "2d": run_2d_demo,
    "2d-signature": run_2d_signature_demo,
    "3d-gaussian": run_3d_gaussian_demo,
    "3d-gaussian-interactive": run_3d_gaussian_interactive_demo,
    "3d-gaussian-slider": run_3d_gaussian_slider_demo,
    "3d-dog": run_3d_dog_demo,
    "3d-dog-napari": run_3d_dog_napari_demo,
    "3d-extrema": run_3d_extrema_demo,
    "3d-extrema-napari": run_3d_extrema_napari_demo,
    "voxel-storage": run_voxel_storage_demo,
    "pc-radii": run_pc_radii_demo,
    "pc-radii-napari": run_pc_radii_napari_demo,
    "pc-geom": run_pc_geom_demo,
    "pc-geom-napari": run_pc_geom_napari_demo,
    "pc-voxel": run_pc_voxel_demo,
}


def run_demo(
    mode: str,
    synthetic_name: str | None = None,
    modelnet_index: int | None = None,
) -> None:
    if mode not in DEMO_REGISTRY:
        raise ValueError(f"Unknown demo mode: {mode}")
    DEMO_REGISTRY[mode](
        synthetic_name=synthetic_name,
        modelnet_index=modelnet_index,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=list(DEMO_REGISTRY.keys()),
        default="2d",
        help="Choose which demo pipeline to run.",
    )
    parser.add_argument(
        "--synthetic-name",
        type=str,
        default=None,
        help="Optional synthetic shape name for 3D demos",
    )
    parser.add_argument(
        "--modelnet-index",
        type=int,
        default=None,
        help="Optional ModelNet sample index for 3D demos",
    )
    args = parser.parse_args()
    run_demo(
        args.mode,
        synthetic_name=args.synthetic_name,
        modelnet_index=args.modelnet_index,
    )


if __name__ == "__main__":
    main()
