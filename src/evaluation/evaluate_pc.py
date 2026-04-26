from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from src.common.io import RealPointCloudLoader, SyntheticPointCloudLoader
from src.common.visualization import plot_pointcloud
from src.pointcloud.harris_pc import HarrisPC
from src.pointcloud.params import HarrisPCParams, default_harris_pc_params

PointCloudEntry = tuple[str, np.ndarray]


def load_synthetic_pc_dataset(
    root: str = "data/Pointcloud/synthetic",
) -> List[PointCloudEntry]:
    """Load all synthetic point cloud shapes. Delegates to SyntheticPointCloudLoader."""
    return SyntheticPointCloudLoader(root).load_all()


def load_real_pc_dataset(
    root: str = "data/Pointcloud/real",
) -> List[PointCloudEntry]:
    """Load all real point cloud samples. Delegates to RealPointCloudLoader."""
    return RealPointCloudLoader(root).load_all()


def run_harris_pc_synthetic(
    params: HarrisPCParams | None = None,
    out_dir: str = "outputs/harris_pc",
) -> List[dict]:
    cfg = params or default_harris_pc_params()
    detector = HarrisPC(cfg)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    for name, pts in load_synthetic_pc_dataset():
        kps = detector.detect(pts)
        save_path = out / f"{name}_pc.png"
        plot_pointcloud(
            [pts],
            titles=[f"{name}: Harris PC ({kps.shape[0]} kps)"],
            keypoints_list=[kps],
            show=False,
            save_path=str(save_path),
        )
        rows.append(
            {
                "shape": name,
                "num_keypoints": int(kps.shape[0]),
                "plot_path": str(save_path),
            }
        )
    return rows


def run_harris_pc_real(
    params: HarrisPCParams | None = None,
    out_dir: str = "outputs/harris_pc",
) -> List[dict]:
    cfg = params or default_harris_pc_params()
    detector = HarrisPC(cfg)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    for name, pts in load_real_pc_dataset():
        kps = detector.detect(pts)
        save_path = out / f"{name}_pc.png"
        plot_pointcloud(
            [pts],
            titles=[f"{name}: Harris PC ({kps.shape[0]} kps)"],
            keypoints_list=[kps],
            show=False,
            save_path=str(save_path),
        )
        rows.append(
            {
                "shape": name,
                "num_keypoints": int(kps.shape[0]),
                "plot_path": str(save_path),
            }
        )
    return rows


if __name__ == "__main__":
    print("=== Synthetic point clouds ===")
    for row in run_harris_pc_synthetic():
        print(f"  {row['shape']:>15} | keypoints={row['num_keypoints']:>4}")

    print("\n=== Real point clouds ===")
    for row in run_harris_pc_real():
        print(f"  {row['shape']:>15} | keypoints={row['num_keypoints']:>4}")
