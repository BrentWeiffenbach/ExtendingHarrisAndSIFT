from src.common.visualization import plot_voxels
import numpy as np
import gzip


def main():
    # Synthetic cube: (32, 32, 32) bool
    cube = np.load("data/Voxel/synthetic/pyramid.npy")

    # Real chair sample: (890, 1, 32, 32, 32) -> pick first sample, squeeze channel
    with gzip.open("data/Voxel/real/ModelNet10-dataset/chair.npy.gz", "rb") as f:
        chairs = np.load(f)
    # Some samples are empty; pick the first non-empty one
    idx = next(i for i in range(len(chairs)) if chairs[i, 0].any())
    chair = chairs[idx, 0].astype(bool)  # (32, 32, 32)

    plot_voxels(
        [cube, chair],
        titles=["Synthetic Cube", "Real Chair (ModelNet10)"],
    )

if __name__ == "__main__":
    main()
