from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.common.io import ModelNetLoader, SyntheticVoxelLoader
from src.common.visualization import plot_voxels
from src.voxel.harris3d import Harris3DVoxel
from src.voxel.params import Harris3DParams, default_harris3d_params

VolumeEntry = Tuple[str, np.ndarray]


def load_synthetic_voxel_dataset(
    root: str = "data/Voxel/synthetic",
) -> List[VolumeEntry]:
    """Load all synthetic voxel shapes. Delegates to SyntheticVoxelLoader."""
    return SyntheticVoxelLoader(root).load_all()


def run_harris3d_synthetic(
    params: Harris3DParams | None = None,
    out_dir: str = "outputs/harris3d",
) -> List[dict]:
    cfg = params or default_harris3d_params()
    detector = Harris3DVoxel(cfg)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []

    for name, vol in load_synthetic_voxel_dataset():
        kps = detector.detect(vol)
        volume_path = out / f"{name}_volume.png"
        plot_voxels(
            [vol],
            titles=[f"{name}: 3D Volume + Keypoints"],
            keypoints_list=[kps],
            show=False,
            save_path=str(volume_path),
        )

        rows.append(
            {
                "shape": name,
                "num_keypoints": int(kps.shape[0]),
                "volume_plot": str(volume_path),
            }
        )
    return rows


def run_harris3d_real_chair(
    params: Harris3DParams | None = None,
    out_dir: str = "outputs/harris3d",
) -> dict:
    cfg = params or default_harris3d_params()
    detector = Harris3DVoxel(cfg)

    modelnet_loader = ModelNetLoader(
        "data/Voxel/real/ModelNet10-dataset/modelnet10.npy.gz"
    )
    # Find first non-empty chair sample
    idx = -1
    chair = None
    for idx, chair in modelnet_loader.load_sequential():
        if chair.any():
            break

    keypoints = detector.detect(chair)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    volume_path = out / "chair_volume.png"

    plot_voxels(
        [chair],
        titles=[f"ModelNet10 Chair sample {idx}: 3D Volume + Keypoints"],
        keypoints_list=[keypoints],
        show=False,
        save_path=str(volume_path),
    )

    report = {
        "shape": "chair",
        "sample_index": int(idx),
        "num_keypoints": int(keypoints.shape[0]),
        "volume_plot": str(volume_path),
    }
    return report


def run_harris3d_random_modelnet_sample(
    params: Harris3DParams | None = None,
    sample_index: int | None = None,
) -> dict:
    cfg = params or default_harris3d_params()
    detector = Harris3DVoxel(cfg)

    modelnet_loader = ModelNetLoader(
        "data/Voxel/real/ModelNet10-dataset/modelnet10.npy.gz"
    )

    if sample_index is not None:
        idx, sample = sample_index, modelnet_loader.load_by_index(sample_index)
    else:
        idx, sample = modelnet_loader.load_random()

    keypoints = detector.detect(sample)

    return {
        "shape": "modelnet_random",
        "sample_index": int(idx),
        "num_keypoints": int(keypoints.shape[0]),
        "sample": sample,
        "keypoints": keypoints,
    }


if __name__ == "__main__":
    rows = run_harris3d_synthetic()
    for row in rows:
        print(f"{row['shape']:>8} | keypoints={row['num_keypoints']:>4}")

    chair = run_harris3d_real_chair()
    print(f"{chair['shape']:>8} | keypoints={chair['num_keypoints']:>4}")
