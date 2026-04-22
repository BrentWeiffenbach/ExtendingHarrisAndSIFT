"""Batch runner for all detectors on all datasets.

Runs Harris3D and SIFT3D on voxels (synthetic + real ModelNet10),
SIFT-geom/radii/voxel on synthetic point clouds, and SIFT2D on images.
Saves all results to outputs/run_all/.
"""

import argparse
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from src.common.io import SyntheticVoxelLoader, ModelNetLoader, load_image, load_pointcloud
from src.common.visualization import plot_voxels, plot_pointcloud
from src.pointcloud.params import SIFTGeomPCParams, SIFTRadiiPCParams, SIFTVoxelPCParams
from src.pointcloud.sift_pc import SIFTGeomPC, SIFTRadiiPC, SIFTVoxelPC
from src.voxel.harris3d import Harris3DVoxel
from src.voxel.params import default_harris3d_params, SIFT3DParams, SIFT2DParams
from src.voxel.sift2d import SIFT2D
from src.voxel.sift3d import SIFT3DVoxel

_PC_SHAPES = ["cone", "cube", "cuboid", "cylinder", "pyramid", "sphere", "torus"]


def _print_banner() -> None:
    print("=" * 70)
    print("BATCH RUNNER: 3D & 2D Detector Evaluation")
    print("=" * 70)


def _load_voxel_datasets() -> list[tuple[str, np.ndarray]]:
    """Load synthetic voxels and a ModelNet subset for 3D voxel runs."""
    synthetic_loader = SyntheticVoxelLoader()
    synthetic_data = synthetic_loader.load_all()
    print(f"  Loaded {len(synthetic_data)} synthetic shapes")

    modelnet_loader = ModelNetLoader(
        "data/Voxel/real/ModelNet10-dataset/modelnet10.npy.gz"
    )
    modelnet_data = [
        (f"modelnet_{i}", vol) for i, vol in modelnet_loader.load_range(5, 10)
    ]
    print(f"  Loaded {len(modelnet_data)} ModelNet10 samples")

    return synthetic_data + modelnet_data


def _load_image_dataset() -> list[tuple[str, np.ndarray]]:
    """Load all 2D images for detector runs."""
    image_data: list[tuple[str, np.ndarray]] = []
    for img_path in sorted(Path("data/2d").glob("*.jpg")):
        try:
            img = load_image(str(img_path))
            image_data.append((img_path.stem, img))
        except Exception as e:
            print(f"  [WARN] Failed to load image {img_path}: {e}")
    print(f"  Loaded {len(image_data)} images")
    return image_data


def _normalize_pointcloud(pts: np.ndarray) -> np.ndarray:
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    rng = hi - lo
    rng[rng == 0] = 1.0
    return (pts - lo) / rng


def _run_3d_voxel_batch(
    voxel_data: list[tuple[str, np.ndarray]],
    out_root: Path,
    detectors: list[str],
) -> None:
    """Run selected 3D voxel detectors over voxel datasets."""
    print("\n[2/4] Running 3D voxel detectors...")
    for name, vol in voxel_data:
        for detector in detectors:
            try:
                run_3d_voxel(name, vol, out_root, detector)
            except Exception as e:
                print(f"  [ERROR] 3D {detector} {name}: {e}")


def _run_3d_pc_batch(out_root: Path, pc_detector: str) -> None:
    """Run point-cloud detector over all synthetic PLY shapes."""
    print(f"\n[3/4] Running 3D point-cloud detector ({pc_detector})...")
    for shape in _PC_SHAPES:
        try:
            run_3d_pc(shape, out_root, pc_detector)
        except Exception as e:
            print(f"  [ERROR] PC {pc_detector} {shape}: {e}")


def _run_2d_batch(
    image_data: list[tuple[str, np.ndarray]],
    out_root: Path,
    detectors: list[str],
) -> None:
    """Run selected 2D detectors over image datasets."""
    print("\n[4/4] Running 2D image detectors...")
    for name, img in image_data:
        for detector in detectors:
            try:
                run_2d_image(name, img, out_root, detector)
            except Exception as e:
                print(f"  [ERROR] 2D {detector} {name}: {e}")


def _print_footer(out_root: Path) -> None:
    print("\n" + "=" * 70)
    print(f"Results saved to: {out_root}")
    print("=" * 70)


def run_3d_voxel(
    name: str, vol: np.ndarray, out_root: Path, detector_name: str
) -> None:
    """Run 3D voxel detector on a volume and save result.

    Parameters
    ----------
    name : str
        Sample name (e.g., 'cube', 'modelnet_0')
    vol : np.ndarray
        3D voxel volume (32, 32, 32) bool
    out_root : Path
        Output root directory
    detector_name : str
        Detector name: 'harris' or 'sift'
    """
    out_path = out_root / "3d" / detector_name / "voxel" / f"{name}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if detector_name == "harris":
        params = default_harris3d_params()
        detector = Harris3DVoxel(params)
        keypoints = detector.detect(vol)
    elif detector_name == "sift":
        params = SIFT3DParams()
        detector = SIFT3DVoxel(params)
        result = detector.run(vol)
        if result.extrema_global.shape[0] > 0:
            keypoints = result.extrema_global[:, [2, 1, 0]].astype(np.int32)
        else:
            keypoints = np.empty((0, 3), dtype=np.int32)
    else:
        raise ValueError(f"Unknown voxel detector: {detector_name}")

    print(f"  {detector_name:>6} {name:>15} | keypoints: {len(keypoints):>4}")

    plot_voxels(
        [vol],
        titles=[f"{name} ({detector_name.upper()})"],
        keypoints_list=[keypoints],
        show=False,
        save_path=str(out_path),
    )


def run_3d_pc(name: str, out_root: Path, pc_detector: str) -> None:
    """Run point-cloud SIFT detector on a synthetic shape and save result.

    Parameters
    ----------
    name : str
        Shape name matching data/Pointcloud/synthetic/<name>.ply
    out_root : Path
        Output root directory
    pc_detector : str
        One of 'geom' (default), 'radii', 'voxel'
    """
    ply_path = f"data/Pointcloud/synthetic/{name}.ply"
    pcd = load_pointcloud(ply_path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    pts = _normalize_pointcloud(pts)

    if pc_detector == "geom":
        params = SIFTGeomPCParams(
            num_octaves=3,
            scales_per_octave=4,
            base_radius=0.08,
            contrast_threshold=1e-4,
        )
        keypoints = SIFTGeomPC(params).detect(pts)
    elif pc_detector == "radii":
        params = SIFTRadiiPCParams(
            num_octaves=3,
            scales_per_octave=4,
            base_radius=0.08,
            contrast_threshold=0.0005,
        )
        keypoints = SIFTRadiiPC(params).detect(pts)
    elif pc_detector == "voxel":
        params = SIFTVoxelPCParams(voxel_size=0.05)
        keypoints = SIFTVoxelPC(params).detect(pts)
    else:
        raise ValueError(f"Unknown PC detector: {pc_detector}")

    print(f"  sift-{pc_detector:>5} {name:>15} | keypoints: {len(keypoints):>4}")

    out_path = out_root / "3d" / f"sift-{pc_detector}" / "pointcloud" / f"{name}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kp_plot = keypoints if keypoints.shape[0] > 0 else None
    plot_pointcloud(
        [pts],
        titles=[f"{name} (sift-{pc_detector})"],
        keypoints_list=[kp_plot],
        show=False,
        save_path=str(out_path),
    )


def run_2d_image(
    name: str, img: np.ndarray, out_root: Path, detector_name: str
) -> None:
    """Run 2D detector on image and save result.

    Parameters
    ----------
    name : str
        Image name (e.g., 'image_0011')
    img : np.ndarray
        Grayscale image (H, W) float32 [0, 1]
    out_root : Path
        Output root directory
    detector_name : str
        Detector name: 'harris' or 'sift'
    """
    out_path = out_root / "2d" / detector_name / "image" / f"{name}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if detector_name == "harris":
        print(f"  [SKIP] 2D Harris not implemented: {name}")
        return

    if detector_name == "sift":
        params = SIFT2DParams()
        detector = SIFT2D(params)
        result = detector.run(img)
        kps = (
            result.extrema_global[:, :2]
            if result.extrema_global.shape[0] > 0
            else np.empty((0, 2))
        )

        print(f"  {detector_name:>6} {name:>15} | keypoints: {len(kps):>4}")

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(img, cmap="gray")
        if len(kps) > 0:
            ax.scatter(kps[:, 1], kps[:, 0], s=20, c="red", linewidths=0.4)
        ax.set_title(f"{name} ({detector_name.upper()})")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        raise ValueError(f"Unknown detector: {detector_name}")


def main() -> None:
    """Run all detectors on all datasets."""
    parser = argparse.ArgumentParser(
        description="Run 2D/3D detector batches with optional filters"
    )
    parser.add_argument(
        "--dimension",
        choices=["2d", "3d", "all"],
        default="all",
        help="Filter runs by dimension (default: all)",
    )
    parser.add_argument(
        "--detector",
        choices=["harris", "sift", "all"],
        default="all",
        help="Filter voxel detector runs (default: all)",
    )
    parser.add_argument(
        "--pc-detector",
        choices=["geom", "radii", "voxel"],
        default="geom",
        help="Point-cloud SIFT variant to run when dimension includes 3d (default: geom)",
    )
    args = parser.parse_args()

    out_root = Path("outputs/run_all")
    _print_banner()
    print(f"Filters: dimension={args.dimension}, detector={args.detector}, pc-detector={args.pc_detector}")

    run_3d = args.dimension in {"3d", "all"}
    run_2d = args.dimension in {"2d", "all"}

    detectors = ["harris", "sift"] if args.detector == "all" else [args.detector]

    # Load datasets
    print("\n[1/4] Loading datasets...")
    voxel_data: list[tuple[str, np.ndarray]] = []
    image_data: list[tuple[str, np.ndarray]] = []

    if run_3d:
        voxel_data = _load_voxel_datasets()
    if run_2d:
        image_data = _load_image_dataset()

    # Voxel detectors
    if run_3d:
        _run_3d_voxel_batch(voxel_data, out_root, detectors)
    else:
        print("\n[2/4] Running 3D voxel detectors...")
        print("  [SKIP] 3D filtered out by --dimension")

    # Point-cloud detectors
    if run_3d:
        _run_3d_pc_batch(out_root, args.pc_detector)
    else:
        print(f"\n[3/4] Running 3D point-cloud detector ({args.pc_detector})...")
        print("  [SKIP] 3D filtered out by --dimension")

    # 2D image detectors
    if run_2d:
        _run_2d_batch(image_data, out_root, detectors)
    else:
        print("\n[4/4] Running 2D image detectors...")
        print("  [SKIP] 2D filtered out by --dimension")

    _print_footer(out_root)


if __name__ == "__main__":
    main()
