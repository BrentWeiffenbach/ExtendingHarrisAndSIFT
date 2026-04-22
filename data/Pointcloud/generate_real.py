"""Download and save real point cloud samples to data/Pointcloud/real/.

Uses Open3D's built-in data module (downloads from GitHub on first run,
then cached locally). Produces four real-world samples alongside the
existing bunny.ply:

    fragment.ply  — indoor RGB-D scan fragment (~100k points)
    armadillo.ply — Stanford armadillo 3D scan (~100k points)
    eagle.ply     — eagle sculpture scan (~100k points, subsampled)

Run once from the repo root:
    python data/Pointcloud/generate_real.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

OUT = Path("data/Pointcloud/real")
MAX_POINTS = 8_000


def _maybe_subsample(pts: np.ndarray, max_pts: int = MAX_POINTS) -> np.ndarray:
    if pts.shape[0] > max_pts:
        rng = np.random.default_rng(42)
        idx = rng.choice(pts.shape[0], size=max_pts, replace=False)
        return pts[idx]
    return pts


def _save(name: str, pts: np.ndarray) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"{name}.ply"
    if out_path.exists():
        print(f"  [SKIP] {name}.ply already exists")
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.io.write_point_cloud(str(out_path), pcd)
    print(f"  Saved {name}.ply ({pts.shape[0]:,} points)")


def main() -> None:
    print("Generating real point cloud samples...")

    # Fragment: indoor RGB-D scan — already a point cloud
    frag = o3d.data.PLYPointCloud()
    pts = np.asarray(o3d.io.read_point_cloud(frag.path).points, dtype=np.float64)
    _save("fragment", _maybe_subsample(pts))

    # Armadillo: classic Stanford 3D scan mesh — extract vertices
    arm = o3d.data.ArmadilloMesh()
    pts = np.asarray(o3d.io.read_triangle_mesh(arm.path).vertices, dtype=np.float64)
    _save("armadillo", _maybe_subsample(pts))

    # Eagle: real point cloud scan — subsample to keep runtime fast
    eagle = o3d.data.EaglePointCloud()
    pts = np.asarray(o3d.io.read_point_cloud(eagle.path).points, dtype=np.float64)
    _save("eagle", _maybe_subsample(pts))

    print("Done. Files saved to", OUT)


if __name__ == "__main__":
    main()
