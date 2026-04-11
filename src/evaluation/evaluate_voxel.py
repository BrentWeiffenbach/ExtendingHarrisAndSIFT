from __future__ import annotations

import gzip
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.common.visualization import plot_voxels
from src.voxel.harris3d import Harris3DVoxel
from src.voxel.params import Harris3DParams, default_harris3d_params

VolumeEntry = Tuple[str, np.ndarray]


def load_synthetic_voxel_dataset(
    root: str = "data/Voxel/synthetic",
) -> List[VolumeEntry]:
    base = Path(root)
    volumes: List[VolumeEntry] = []
    for path in sorted(base.glob("*.npy")):
        volumes.append((path.stem, np.load(path).astype(bool)))
    return volumes


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

    with gzip.open("data/Voxel/real/ModelNet10-dataset/chair.npy.gz", "rb") as f:
        chairs = np.load(f)

    idx = next(i for i in range(len(chairs)) if chairs[i, 0].any())
    chair = chairs[idx, 0].astype(bool)
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

    modelnet_path = "data/Voxel/real/ModelNet10-dataset/modelnet10.npy.gz"
    with gzip.open(modelnet_path, "rb") as f:
        modelnet = np.load(f)

    idx = (
        sample_index
        if sample_index is not None
        else random.randint(0, modelnet.shape[0] - 1)
    )
    sample = modelnet[idx, 0].astype(bool)
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
