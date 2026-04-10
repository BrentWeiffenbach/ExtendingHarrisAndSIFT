import argparse

import numpy as np

from src.common.visualization import (
    plot_dog_scale_space_3d,
    plot_dog_scale_space,
    plot_extrema_gradient_overlay_3d,
    plot_extrema_scale_space_3d,
    plot_gaussian_scale_space,
    plot_gaussian_scale_space_3d,
    plot_gaussian_scale_space_3d_interactive,
    plot_extrema_gradient_overlay,
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


def run_2d_demo():
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


def run_2d_signature_demo():
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
        np.clip(
            params.orientation_visual_keypoint_index,
            0,
            len(signatures) - 1,
        )
    )
    plot_sift2d_orientation_views(result.original_image, signatures[signature_index])


def run_3d_gaussian_demo():
    volume_path = "data/Voxel/synthetic/cube.npy"
    params = SIFT3DParams(
        num_octaves=4,
        scales_per_octave=5,
        base_sigma=1.2,
        min_size=8,
        downsample_factor=2,
        slice_axis=0,
    )
    detector = SIFT3DVoxel(params)
    result = detector.run(volume_path)
    plot_gaussian_scale_space_3d(
        result.gaussian_pyramid,
        result.sigma_pyramid,
        slice_axis=params.slice_axis,
    )


def run_3d_gaussian_interactive_demo():
    volume_path = "data/Voxel/synthetic/cube.npy"
    params = SIFT3DParams(
        num_octaves=4,
        scales_per_octave=5,
        base_sigma=1.2,
        min_size=8,
        downsample_factor=2,
        slice_axis=0,
    )
    detector = SIFT3DVoxel(params)
    result = detector.run(volume_path)
    plot_gaussian_scale_space_3d_interactive(
        result.gaussian_pyramid,
        result.sigma_pyramid,
        slice_axis=params.slice_axis,
    )


def run_3d_gaussian_napari_demo():
    volume_path = "data/Voxel/synthetic/cube.npy"
    params = SIFT3DParams(
        num_octaves=4,
        scales_per_octave=5,
        base_sigma=1.2,
        min_size=8,
        downsample_factor=2,
        slice_axis=0,
    )
    detector = SIFT3DVoxel(params)
    result = detector.run(volume_path)
    view_gaussian_scale_space_3d_napari(
        result.gaussian_pyramid,
        result.sigma_pyramid,
    )


def run_3d_dog_demo():
    volume_path = "data/Voxel/synthetic/pyramid.npy"
    params = SIFT3DParams(
        num_octaves=4,
        scales_per_octave=5,
        base_sigma=1.2,
        min_size=8,
        downsample_factor=2,
        slice_axis=0,
    )
    detector = SIFT3DVoxel(params)
    result = detector.run(volume_path)
    plot_dog_scale_space_3d(
        result.dog_pyramid,
        result.dog_sigma_pairs,
        slice_axis=params.slice_axis,
    )


def run_3d_dog_napari_demo():
    volume_path = "data/Voxel/synthetic/pyramid.npy"
    params = SIFT3DParams(
        num_octaves=4,
        scales_per_octave=5,
        base_sigma=1.2,
        min_size=8,
        downsample_factor=2,
        slice_axis=0,
    )
    detector = SIFT3DVoxel(params)
    result = detector.run(volume_path)
    view_dog_scale_space_3d_napari(
        result.dog_pyramid,
        result.dog_sigma_pairs,
    )


def run_3d_extrema_demo():
    volume_path = "data/Voxel/synthetic/pyramid.npy"
    params = SIFT3DParams(
        num_octaves=4,
        scales_per_octave=5,
        base_sigma=1.2,
        min_size=8,
        downsample_factor=2,
        slice_axis=0,
        extrema_contrast_threshold=0.01,
        extrema_border=1,
        blob_radius_factor=1.0,
        max_blob_keypoints=2000,
    )
    detector = SIFT3DVoxel(params)
    result = detector.run(volume_path)
    plot_extrema_gradient_overlay_3d(
        result.original_volume,
        result.extrema_global,
        radius_factor=params.blob_radius_factor,
        max_blobs=params.max_blob_keypoints,
    )


def run_3d_extrema_napari_demo():
    volume_path = "data/Voxel/synthetic/pyramid.npy"
    params = SIFT3DParams(
        num_octaves=4,
        scales_per_octave=5,
        base_sigma=1.2,
        min_size=8,
        downsample_factor=2,
        slice_axis=0,
        extrema_contrast_threshold=0.00,
        extrema_border=1,
        blob_radius_factor=1.0,
        max_blob_keypoints=2000,
    )
    detector = SIFT3DVoxel(params)
    result = detector.run(volume_path)
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


def run_voxel_storage_demo():
    volume = np.load("data/Voxel/synthetic/pyramid.npy").astype(np.float32)
    plot_voxel_storage_layout(volume)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[
            "2d",
            "2d-signature",
            "3d-gaussian",
            "3d-gaussian-interactive",
            "3d-gaussian-napari",
            "3d-dog",
            "3d-dog-napari",
            "3d-extrema",
            "3d-extrema-napari",
            "voxel-storage",
        ],
        default="2d",
        help="Choose which demo pipeline to run.",
    )
    args = parser.parse_args()

    if args.mode == "2d":
        run_2d_demo()
    elif args.mode == "2d-signature":
        run_2d_signature_demo()
    elif args.mode == "3d-gaussian":
        run_3d_gaussian_demo()
    elif args.mode == "3d-gaussian-interactive":
        run_3d_gaussian_interactive_demo()
    elif args.mode == "3d-gaussian-napari":
        run_3d_gaussian_napari_demo()
    elif args.mode == "3d-dog":
        run_3d_dog_demo()
    elif args.mode == "3d-dog-napari":
        run_3d_dog_napari_demo()
    elif args.mode == "3d-extrema":
        run_3d_extrema_demo()
    elif args.mode == "3d-extrema-napari":
        run_3d_extrema_napari_demo()
    elif args.mode == "voxel-storage":
        run_voxel_storage_demo()
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
