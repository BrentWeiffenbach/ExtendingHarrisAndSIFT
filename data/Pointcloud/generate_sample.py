"""
Generate sample point cloud data for testing Harris 3D and SIFT 3D detectors.

Synthetic data:
  - cube_noisy.ply   : noisy unit cube surface (8 ground-truth corners — ideal for Harris)
  - sphere_noisy.ply : noisy sphere surface (no corners — good negative test)

Real data (open3d built-in download):
  - real/bunny.ply   : Stanford Bunny (~40 k points, rich geometry)

Usage:
    python data/generate_sample.py
"""

import numpy as np
import open3d as o3d
from pathlib import Path

SYNTHETIC_DIR = Path(__file__).parent / "synthetic"
REAL_DIR = Path(__file__).parent / "real"
SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
REAL_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------

def _save_ply(points: np.ndarray, path: Path) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    o3d.io.write_point_cloud(str(path), pcd)
    print(f"Saved {len(points):,} points → {path}")


def generate_cube(n_per_face: int = 2000, noise_std: float = 0.01) -> np.ndarray:
    """Uniformly sample the 6 faces of a unit cube then add Gaussian noise."""
    faces = []
    for axis in range(3):
        for val in (0.0, 1.0):
            u = RNG.random((n_per_face, 2))
            pts = np.zeros((n_per_face, 3))
            other = [i for i in range(3) if i != axis]
            pts[:, other] = u
            pts[:, axis] = val
            faces.append(pts)
    pts = np.concatenate(faces, axis=0)
    pts += RNG.normal(0, noise_std, pts.shape)
    return pts


def generate_sphere(n_points: int = 12000, radius: float = 1.0,
                    noise_std: float = 0.01) -> np.ndarray:
    """Uniformly sample a sphere surface using Marsaglia method."""
    v = RNG.standard_normal((n_points, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    pts = radius * v
    pts += RNG.normal(0, noise_std, pts.shape)
    return pts


# ---------------------------------------------------------------------------
# Real data via open3d built-in downloader
# ---------------------------------------------------------------------------

def download_bunny() -> None:
    """Download the Stanford Bunny mesh, sample a point cloud, and save it."""
    dest = REAL_DIR / "bunny.ply"
    if dest.exists():
        print(f"Bunny already exists at {dest}, skipping download.")
        return

    print("Downloading Stanford Bunny via open3d datasets …")
    bunny = o3d.data.BunnyMesh()          # downloads on first call (~3 MB)
    mesh = o3d.io.read_triangle_mesh(bunny.path)
    mesh.compute_vertex_normals()

    # Sample 40 000 points from the mesh surface
    pcd = mesh.sample_points_uniformly(number_of_points=40000)
    o3d.io.write_point_cloud(str(dest), pcd)
    print(f"Saved bunny point cloud → {dest}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Synthetic datasets
    cube_pts = generate_cube(n_per_face=2000, noise_std=0.01)
    _save_ply(cube_pts, SYNTHETIC_DIR / "cube_noisy.ply")

    sphere_pts = generate_sphere(n_points=12000, noise_std=0.01)
    _save_ply(sphere_pts, SYNTHETIC_DIR / "sphere_noisy.ply")

    # Real data
    download_bunny()

    print("\nAll sample data ready.")
    print("  Synthetic → data/synthetic/")
    print("  Real      → data/real/")
