import gzip

import numpy as np
import open3d as o3d

from src.common.visualization import plot_pointcloud, plot_voxels


def main():
    # Voxel Synthetic cube: (32, 32, 32) bool
    cube = np.load("data/Voxel/synthetic/pyramid.npy")
    # Pointcloud Synthetic cube
    pcd_cube = np.asarray(
        o3d.io.read_point_cloud("data/Pointcloud/synthetic/cube_noisy.ply").points
    )

    # Real chair sample: (890, 1, 32, 32, 32) -> pick first sample, squeeze channel
    with gzip.open("data/Voxel/real/ModelNet10-dataset/chair.npy.gz", "rb") as f:
        chairs = np.load(f)
    # Some samples are empty; pick the first non-empty one
    idx = next(i for i in range(len(chairs)) if chairs[i, 0].any())
    chair = chairs[idx, 0].astype(bool)  # (32, 32, 32)

    # Real bunny pointcloud
    pcd_bunny = np.asarray(
        o3d.io.read_point_cloud("data/Pointcloud/real/bunny.ply").points
    )

    plot_voxels(
        [cube, chair],
        titles=["Synthetic Cube", "Real Chair (ModelNet10)"],
    )

    plot_pointcloud(
        [pcd_cube, pcd_bunny],
        titles=["Synthetic Cube Pointcloud", "Stanford Bunny Pointcloud"],
    )


if __name__ == "__main__":
    main()
