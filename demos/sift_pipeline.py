from __future__ import annotations

import argparse

import numpy as np

from src.common.io import ModelNetLoader, SyntheticVoxelLoader
from src.common.visualization import (
    plot_dog_scale_space,
    plot_dog_scale_space_3d,
    plot_extrema_gradient_overlay,
    plot_extrema_gradient_overlay_3d,
    plot_extrema_scale_space_3d,
    plot_gaussian_scale_space,
    plot_gaussian_scale_space_3d,
    plot_gaussian_scale_space_3d_interactive,
    plot_sift2d_orientation_views,
    plot_voxel_storage_layout,
    rasterize_extrema_blobs_3d,
    view_dog_scale_space_3d_napari,
    view_extrema_blobs_3d_napari,
    view_gaussian_scale_space_3d_napari,
)
from src.voxel.params import SIFT2DParams, SIFT3DParams
from src.voxel.sift2d import SIFT2D
from src.voxel.sift3d import SIFT3DVoxel


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
