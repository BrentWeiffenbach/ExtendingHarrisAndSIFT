from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from demos.sift3d_walkthrough import run_walkthrough
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
from src.pointcloud.params import SIFTRadiiPCParams, SIFTVoxelPCParams
from src.pointcloud.sift_pc import SIFTRadiiPC, SIFTVoxelPC
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
    smoothed_pyramid: list[list[np.ndarray]],
    radii_pyramid: list[list[float]],
    max_octaves: int = 3,
    max_scales: int = 4,
) -> None:
    n_oct = min(len(smoothed_pyramid), max_octaves)
    n_sc = min(len(smoothed_pyramid[0]) if smoothed_pyramid else 0, max_scales)
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
            signal = smoothed_pyramid[o][s]
            # (N, 3) smoothed positions → display displacement magnitude from original
            if signal.ndim == 2:
                signal = np.linalg.norm(signal - pts, axis=1)
            r = radii_pyramid[o][s]
            sc = ax.scatter(pts[:, 0], pts[:, 1], c=signal, s=2, cmap="plasma")
            plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.04)
            ax.set_title(f"Oct {o}, Scale {s}\nr={r:.3f}", fontsize=8)
            ax.set_aspect("equal")
            ax.axis("off")

    fig.suptitle("Point Cloud Scale-Space", fontsize=11)
    plt.tight_layout()
    plt.show()


def _plot_pc_dog(
    points_per_octave: list[np.ndarray],
    dog_pyramid: list[list[np.ndarray]],
    dog_radii: list[list[float]],
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
            r = dog_radii[o][d]
            vmax = float(np.abs(dog).max()) or 1.0
            sc = ax.scatter(
                pts[:, 0], pts[:, 1], c=dog, s=2, cmap="RdBu_r", vmin=-vmax, vmax=vmax
            )
            plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.04)
            ax.set_title(f"Oct {o}, DoG {d}\nr={r:.3f}", fontsize=8)
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

    params = SIFTRadiiPCParams()
    detector = SIFTRadiiPC(params)
    result = detector.run(pts)
    print(f"Detected {result.keypoints.shape[0]} keypoints")

    _plot_pc_scale_space(
        result.points_per_octave,
        result.smoothed_pyramid,
        result.radii_pyramid,
    )

    _plot_pc_dog(
        result.points_per_octave,
        result.dog_pyramid,
        result.radii_pyramid,
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
        radii=[0.02, 0.04, 0.08, 0.16],
        contrast_threshold=0.0005,
    )
    detector = SIFTRadiiPC(params)
    result = detector.run(pts)
    print(f"Detected {result.keypoints.shape[0]} keypoints")

    kp = result.keypoints if result.keypoints.shape[0] > 0 else None
    # Convert (N,3) smoothed positions to displacement magnitude for napari display
    disp_pyramid = [
        [
            np.linalg.norm(s - result.points_per_octave[o], axis=1).astype(np.float32)
            for s in octave
        ]
        for o, octave in enumerate(result.smoothed_pyramid)
    ]
    view_pc_radii_napari(
        points=pts,
        points_per_octave=result.points_per_octave,
        density_pyramid=disp_pyramid,
        radii_pyramid=result.radii_pyramid,
        dog_pyramid=result.dog_pyramid,
        keypoints=kp,
        signal_name="position drift",
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


def run_pc_sift_response_demo(
    synthetic_name: str | None = None,
    modelnet_index: int | None = None,
) -> None:
    """Visualize 3D SIFT keypoint responses across all three point-cloud variants.

    Produces three sequential figures:
      1. Per-method 3D scatter — keypoints colored by |DoG response|, sized by scale radius
      2. Response analysis — distribution histogram, response vs scale, scale histogram
      3. Multi-shape bar charts — keypoint count and mean |response| for SIFTGeomPC on all shapes

    Use --synthetic-name to choose the shape for Figs 1–2 (default: cube).
    """
    pts, name = _load_synthetic_pc(synthetic_name)
    print(f"Loaded '{name}': {pts.shape[0]} points")

    radii_params = SIFTRadiiPCParams(
        num_octaves=3,
        radii=[0.05, 0.08, 0.13, 0.2],
        contrast_threshold=0.0005,
    )
    voxel_params = SIFTVoxelPCParams(voxel_size=0.05)

    print("Running SIFTRadiiPC...")
    radii_result = SIFTRadiiPC(radii_params).run(pts)
    print("Running SIFTVoxelPC...")
    voxel_run = SIFTVoxelPC(voxel_params).run(pts)

    voxel_kp = voxel_run["keypoints"]  # (N, 3) physical xyz
    sift3d_eg = voxel_run[
        "sift3d_result"
    ].extrema_global  # (N, 7): z,y,x,sigma,response,octave,rs
    voxel_responses = (
        np.abs(sift3d_eg[:, 4])
        if sift3d_eg.shape[0] > 0
        else np.array([], dtype=np.float32)
    )
    voxel_sigmas = (
        sift3d_eg[:, 3] * voxel_params.voxel_size
        if sift3d_eg.shape[0] > 0
        else np.array([], dtype=np.float32)
    )

    print(
        f"Keypoints — Radii: {radii_result.keypoints.shape[0]}, "
        f"Voxel: {voxel_kp.shape[0]}"
    )

    rng = np.random.default_rng(0)
    disp_idx = rng.choice(len(pts), min(5000, len(pts)), replace=False)
    pts_d = pts[disp_idx]
    elev, azim = 25, 45

    # ---- Figure 1: Per-method 3D scatter colored by |response| ---------------
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(
        f"3D SIFT Keypoint Responses — '{name}'\n"
        "Color = |DoG response|; marker size ∝ scale radius",
        fontsize=11,
    )

    method_specs: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = [
        (
            "SIFTRadiiPC",
            radii_result.keypoints[:, :3]
            if radii_result.keypoints.shape[0] > 0
            else np.empty((0, 3)),
            np.abs(radii_result.keypoints[:, 4])
            if radii_result.keypoints.shape[0] > 0
            else np.array([]),
            np.clip(radii_result.keypoints[:, 3] * 1000, 30, 300)
            if radii_result.keypoints.shape[0] > 0
            else np.array([]),
        ),
        ("SIFTVoxelPC", voxel_kp, voxel_responses, np.full(len(voxel_kp), 60)),
    ]

    for col, (method_name, kp_xyz, responses, sizes) in enumerate(method_specs):
        ax = fig.add_subplot(1, 3, col + 1, projection="3d")
        ax.scatter(
            pts_d[:, 0],
            pts_d[:, 1],
            pts_d[:, 2],
            c="lightsteelblue",
            s=1,
            alpha=0.2,
        )
        if kp_xyz.shape[0] > 0 and len(responses) > 0:
            vmax = float(np.percentile(responses, 95)) or 1e-6
            sc = ax.scatter(
                kp_xyz[:, 0],
                kp_xyz[:, 1],
                kp_xyz[:, 2],
                c=responses,
                s=sizes,
                cmap="hot",
                vmin=0,
                vmax=vmax,
                edgecolors="red",
                linewidths=0.5,
                alpha=0.9,
                zorder=5,
            )
            plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.12, label="|response|")
        ax.set_title(f"{method_name}\n{kp_xyz.shape[0]} keypoints", fontsize=9)
        ax.view_init(elev=elev, azim=azim)
        ax.tick_params(labelsize=6)

    plt.tight_layout()
    plt.show()

    # ---- Figure 2: Response analysis ------------------------------------------
    radii_kp = radii_result.keypoints

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"SIFT Response Analysis — '{name}'", fontsize=11)

    # |response| histogram
    ax = axes[0]
    if radii_kp.shape[0] > 0:
        ax.hist(
            np.abs(radii_kp[:, 4]),
            bins=30,
            alpha=0.6,
            label=f"Radii ({radii_kp.shape[0]})",
            color="steelblue",
            density=True,
        )
    if len(voxel_responses) > 0:
        ax.hist(
            voxel_responses,
            bins=30,
            alpha=0.6,
            label=f"Voxel ({voxel_kp.shape[0]})",
            color="seagreen",
            density=True,
        )
    ax.set_xlabel("|DoG response|")
    ax.set_ylabel("Density")
    ax.set_title("Response Magnitude Distribution")
    ax.legend(fontsize=8)

    # Response vs scale
    ax = axes[1]
    if radii_kp.shape[0] > 0:
        ax.scatter(
            radii_kp[:, 3],
            np.abs(radii_kp[:, 4]),
            s=15,
            alpha=0.6,
            label="Radii",
            color="steelblue",
        )
    if len(voxel_sigmas) > 0 and len(voxel_responses) > 0:
        ax.scatter(
            voxel_sigmas,
            voxel_responses,
            s=15,
            alpha=0.6,
            label="Voxel (σ·voxel_size)",
            color="seagreen",
        )
    ax.set_xlabel("Scale (radius or σ·voxel_size)")
    ax.set_ylabel("|DoG response|")
    ax.set_title("Response vs Scale")
    ax.legend(fontsize=8)

    # Scale distribution
    ax = axes[2]
    if radii_kp.shape[0] > 0:
        ax.hist(
            radii_kp[:, 3],
            bins=25,
            alpha=0.6,
            label="Radii",
            color="steelblue",
            density=True,
        )
    if len(voxel_sigmas) > 0:
        ax.hist(
            voxel_sigmas,
            bins=25,
            alpha=0.6,
            label="Voxel",
            color="seagreen",
            density=True,
        )
    ax.set_xlabel("Scale")
    ax.set_ylabel("Density")
    ax.set_title("Scale Distribution")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


DEMO_REGISTRY = {
    "3d-walkthrough": lambda synthetic_name=None, modelnet_index=None: run_walkthrough(
        shape=synthetic_name, modelnet_index=modelnet_index
    ),
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
    "pc-voxel": run_pc_voxel_demo,
    "pc-sift-response": run_pc_sift_response_demo,
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
