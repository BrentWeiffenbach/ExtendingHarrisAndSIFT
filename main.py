"""Main CLI for running Harris/SIFT detectors or SIFT pipeline demos.

Examples:
  # Run a SIFT pipeline demo
  python main.py --demo 3d-extrema

  # Run Harris on a synthetic cube
  python main.py --detector harris --synthetic-name cube --show

  # Run SIFT on ModelNet10 sample 0, save result
  python main.py --detector sift --modelnet-index 0 --no-show --output-dir /tmp

  # Run SIFT on a 2D image
  python main.py --dimension 2d --detector sift --show
"""

import argparse
import importlib
import inspect
import os
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from demos.sift_pipeline import DEMO_REGISTRY, run_demo
from src.common.io import SyntheticVoxelLoader, ModelNetLoader, load_image
from src.common.visualization import (
    plot_voxels,
    rasterize_extrema_blobs_3d,
    view_extrema_blobs_3d_napari,
)
from src.voxel.harris3d import Harris3DVoxel
from src.voxel.params import default_harris3d_params, SIFT3DParams, SIFT2DParams
from src.voxel.sift2d import SIFT2D
from src.voxel.sift3d import SIFT3DVoxel


def _show_3d_napari(
    volume: np.ndarray,
    keypoints_xyz: np.ndarray,
    name: str,
    detector_name: str,
    sift_extrema_global: np.ndarray | None = None,
    sift_params: SIFT3DParams | None = None,
) -> None:
    """Show 3D volume and keypoints in napari.

    Parameters
    ----------
    volume : np.ndarray
        Volume in (z, y, x) order.
    keypoints_xyz : np.ndarray
        Keypoints in (x, y, z) order.
    name : str
        Layer title prefix.
    detector_name : str
        Detector name ('harris' or 'sift').
    sift_extrema_global : np.ndarray | None
        SIFT extrema in global (z, y, x, sigma, response, octave, dog_idx).
    sift_params : SIFT3DParams | None
        SIFT parameters for blob radius/limits.
    """
    if detector_name == "sift" and sift_extrema_global is not None:
        params = sift_params or SIFT3DParams()
        blob_labels, centers = rasterize_extrema_blobs_3d(
            volume.shape,
            sift_extrema_global,
            radius_factor=params.blob_radius_factor,
            max_blobs=params.max_blob_keypoints,
        )
        view_extrema_blobs_3d_napari(
            volume,
            blob_labels,
            centers,
            sift_extrema_global,
        )
        return

    try:
        napari = importlib.import_module("napari")
    except ImportError as exc:
        raise ImportError(
            "napari is not installed. Install with: uv pip install napari pyqt5"
        ) from exc

    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(
        volume.astype(np.float32),
        name=f"{name} volume",
        rendering="mip",
        colormap="gray",
        depiction="volume",
    )

    if keypoints_xyz.size > 0:
        # napari points are expected in (z, y, x)
        keypoints_zyx = keypoints_xyz[:, [2, 1, 0]].astype(np.float32)
        points_kwargs = {
            "data": keypoints_zyx,
            "name": f"{name} keypoints",
            "size": 4,
            "face_color": "red",
        }
        add_points_params = inspect.signature(viewer.add_points).parameters
        if "edge_color" in add_points_params:
            points_kwargs["edge_color"] = "white"
        elif "border_color" in add_points_params:
            points_kwargs["border_color"] = "white"
        viewer.add_points(**points_kwargs)

    viewer.scale_bar.visible = True
    viewer.text_overlay.visible = True
    viewer.text_overlay.text = "3D detector output in napari"
    napari.run()


def _run_3d_detector(
    detector_name: str,
    vol: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None, SIFT3DParams | None]:
    """Run a 3D detector on a voxel volume.

    Parameters
    ----------
    detector_name : str
        'harris' or 'sift'
    vol : np.ndarray
        Voxel volume (32, 32, 32) bool

    Returns
    -------
    tuple[np.ndarray, np.ndarray | None, SIFT3DParams | None]
        (keypoints_xyz, sift_extrema_global_or_none, sift_params_or_none)
    """
    if detector_name == "harris":
        params = default_harris3d_params()
        detector = Harris3DVoxel(params)
        keypoints = detector.detect(vol)
        return keypoints, None, None
    elif detector_name == "sift":
        params = SIFT3DParams()
        detector = SIFT3DVoxel(params)
        result = detector.run(vol)
        # Convert z,y,x -> x,y,z
        if result.extrema_global.shape[0] > 0:
            keypoints = result.extrema_global[:, [2, 1, 0]].astype(np.int32)
        else:
            keypoints = np.empty((0, 3), dtype=np.int32)
        return keypoints, result.extrema_global.astype(np.float32), params
    else:
        raise ValueError(f"Unknown 3D detector: {detector_name}")


def _run_2d_detector(detector_name: str, img: np.ndarray) -> np.ndarray:
    """Run a 2D detector on an image.

    Parameters
    ----------
    detector_name : str
        'harris' or 'sift'
    img : np.ndarray
        Grayscale image (H, W) float32 [0, 1]

    Returns
    -------
    np.ndarray
        Keypoints (N, 2) in (y, x) format
    """
    if detector_name == "harris":
        raise NotImplementedError("2D Harris detector not yet implemented")
    elif detector_name == "sift":
        params = SIFT2DParams()
        detector = SIFT2D(params)
        result = detector.run(img)
        keypoints = (
            result.extrema_global[:, :2]
            if result.extrema_global.shape[0] > 0
            else np.empty((0, 2))
        )
    else:
        raise ValueError(f"Unknown 2D detector: {detector_name}")
    return keypoints


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run Harris/SIFT detectors or visualize SIFT pipeline demos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Demo mode
    parser.add_argument(
        "--demo",
        type=str,
        default=None,
        choices=list(DEMO_REGISTRY.keys()),
        help="Run a SIFT pipeline visualization demo (mutually exclusive with detector mode)",
    )

    # Detector mode
    parser.add_argument(
        "--detector",
        type=str,
        default="harris",
        choices=["harris", "sift"],
        help="Detector to run (default: harris)",
    )

    parser.add_argument(
        "--dimension",
        type=str,
        default="3d",
        choices=["2d", "3d"],
        help="Data dimension (default: 3d)",
    )

    # Data source (one of: --synthetic-name, --modelnet-index, none for default)
    parser.add_argument(
        "--synthetic-name",
        type=str,
        default=None,
        help="Synthetic shape name (cone, cube, cuboid, cylinder, pyramid, sphere, torus)",
    )

    parser.add_argument(
        "--modelnet-index",
        type=int,
        default=None,
        help="ModelNet10 sample index (0-based)",
    )

    # Output options
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show plots (default: True)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/figures",
        help="Output directory for saved plots (default: outputs/figures)",
    )

    args = parser.parse_args()

    # Route to demo mode if requested
    if args.demo is not None:
        print(f"Running demo: {args.demo}")
        run_demo(
            args.demo,
            synthetic_name=args.synthetic_name,
            modelnet_index=args.modelnet_index,
        )
        return

    # Detector mode: load data and run
    if args.dimension == "3d":
        # Load 3D data
        if args.synthetic_name:
            vol = SyntheticVoxelLoader().load_by_name(args.synthetic_name)
            name = args.synthetic_name
        elif args.modelnet_index is not None:
            loader = ModelNetLoader(
                "data/Voxel/real/ModelNet10-dataset/modelnet10.npy.gz"
            )
            vol = loader.load_by_index(args.modelnet_index)
            name = f"modelnet_{args.modelnet_index}"
        else:
            # Default: load first synthetic shape
            shapes = SyntheticVoxelLoader().load_all()
            if not shapes:
                print("Error: No synthetic shapes found")
                sys.exit(1)
            name, vol = shapes[0]
            print(f"Loaded default: {name}")

        # Run detector
        keypoints, sift_extrema_global, sift_params = _run_3d_detector(
            args.detector, vol
        )
        print(f"Detected {len(keypoints)} keypoints")

        # Save/show
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, f"{name}.png")
        plot_voxels(
            [vol],
            titles=[f"{name} ({args.detector.upper()})"],
            keypoints_list=[keypoints],
            show=False,
            save_path=save_path,
        )

        if args.show:
            try:
                _show_3d_napari(
                    vol,
                    keypoints.astype(np.float32),
                    name,
                    args.detector,
                    sift_extrema_global=sift_extrema_global,
                    sift_params=sift_params,
                )
            except ImportError as exc:
                print(f"Warning: {exc}")
                print("Falling back to matplotlib 3D viewer")
                plot_voxels(
                    [vol],
                    titles=[f"{name} ({args.detector.upper()})"],
                    keypoints_list=[keypoints],
                    show=True,
                    save_path=None,
                )
        print(f"Saved to: {save_path}")

    elif args.dimension == "2d":
        # Load 2D data
        if args.synthetic_name or args.modelnet_index is not None:
            print(
                "Warning: --synthetic-name and --modelnet-index are ignored for 2D mode"
            )

        # Load first image
        image_paths = sorted(Path("data/2d").glob("*.jpg"))
        if not image_paths:
            print("Error: No images found in data/2d/")
            sys.exit(1)

        img_path = image_paths[0]
        img = load_image(str(img_path))
        name = img_path.stem
        print(f"Loaded image: {name}")

        # Run detector
        keypoints = _run_2d_detector(args.detector, img)
        print(f"Detected {len(keypoints)} keypoints")

        # Save/show
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, f"{name}.png")

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(img, cmap="gray")
        if len(keypoints) > 0:
            ax.scatter(keypoints[:, 1], keypoints[:, 0], s=20, c="red", linewidths=0.4)
        ax.set_title(f"{name} ({args.detector.upper()})")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if args.show:
            plt.show()
        else:
            plt.close(fig)
        print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()
