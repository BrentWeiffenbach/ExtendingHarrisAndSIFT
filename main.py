import gzip
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.common.visualization import plot_voxels
from src.evaluation.evaluate_voxel import (
    load_synthetic_voxel_dataset,
    run_harris3d_random_modelnet_sample,
    run_harris3d_real_chair,
)

SHOW_PLOTS = True
ONLY_REAL = False


def main():
    # Show all synthetic voxel shapes with detected keypoints
    from src.voxel.harris3d import Harris3DVoxel
    from src.voxel.params import default_harris3d_params

    synthetic_dataset = load_synthetic_voxel_dataset()
    detector = Harris3DVoxel(default_harris3d_params())
    synthetic_out = "outputs/figures/synthetic"
    real_out = "outputs/figures/real"
    os.makedirs(synthetic_out, exist_ok=True)
    os.makedirs(real_out, exist_ok=True)
    if not ONLY_REAL:
        for name, vol in synthetic_dataset:
            keypoints = detector.detect(vol)
            print(f"Showing: {name} | keypoints: {len(keypoints)}")
            if SHOW_PLOTS:
                plot_voxels(
                    [vol],
                    titles=[f"{name}: 3D Volume + Keypoints"],
                    keypoints_list=[keypoints],
                    show=True,
                    save_path=os.path.join(synthetic_out, f"{name}_volume.png"),
                )

    # Show real chair sample with detected keypoints
    real = run_harris3d_real_chair()
    with gzip.open("data/Voxel/real/ModelNet10-dataset/chair.npy.gz", "rb") as f:
        chairs = np.load(f)
    idx = real["sample_index"]
    chair = chairs[idx, 0].astype(bool)
    keypoints = detector.detect(chair)
    print(f"Showing: chair | keypoints: {len(keypoints)}")
    if SHOW_PLOTS:
        plot_voxels(
            [chair],
            titles=[f"ModelNet10 Chair sample {idx}: 3D Volume + Keypoints"],
            keypoints_list=[keypoints],
            show=True,
            save_path=os.path.join(real_out, "chair_volume.png"),
        )

    # Visualize a random sample from ModelNet10
    random_sample = run_harris3d_random_modelnet_sample(
        params=default_harris3d_params()
    )
    sample_idx = random_sample["sample_index"]
    sample = random_sample["sample"]
    keypoints = random_sample["keypoints"]
    print(f"Showing ModelNet10 sample {sample_idx} | keypoints: {len(keypoints)}")
    if SHOW_PLOTS:
        plot_voxels(
            [sample],
            titles=[f"ModelNet10 sample {sample_idx}: 3D Volume + Keypoints"],
            keypoints_list=[keypoints],
            show=True,
            save_path=os.path.join(
                real_out, f"modelnet_sample_{sample_idx}_volume.png"
            ),
        )


if __name__ == "__main__":
    main()
