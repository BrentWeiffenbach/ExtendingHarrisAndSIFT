"""
Generate sample point cloud data for testing Harris 3D and SIFT 3D detectors.

Synthetic data (matching the 7 voxel shapes for direct comparison):
  - cone.ply     : cone surface (1 tip corner)
  - cube.ply     : unit cube surface (8 ground-truth corners)
  - cuboid.ply   : rectangular box surface (8 corners, elongated)
  - cylinder.ply : cylinder surface (2 circular edges, no true corners)
  - pyramid.ply  : square-base pyramid surface (5 corners: 4 base + 1 apex)
  - sphere.ply   : sphere surface (no corners — good negative test)
  - torus.ply    : torus surface (no corners — good negative test)

Real data (open3d built-in download):
  - real/bunny.ply : Stanford Bunny (~40 k points, rich geometry)

All shapes have Gaussian noise added (noise_std=0.005) to simulate
realistic scan artifacts and avoid degenerate flat-face geometry.

Usage:
    python data/Pointcloud/generate_synthetic.py
"""

from pathlib import Path

import numpy as np
import open3d as o3d

SYNTHETIC_DIR = Path(__file__).parent / "synthetic"
REAL_DIR = Path(__file__).parent / "real"
SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
REAL_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)
NOISE_STD = 0.005


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------


def _save_ply(points: np.ndarray, path: Path) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    o3d.io.write_point_cloud(str(path), pcd)
    print(f"Saved {len(points):,} points → {path}")


def _add_noise(pts: np.ndarray, std: float = NOISE_STD) -> np.ndarray:
    return pts + RNG.normal(0, std, pts.shape)


# ---- Cube ----


def generate_cube(n_per_face: int = 2000) -> np.ndarray:
    """Uniformly sample the 6 faces of a unit cube centered at (0.5, 0.5, 0.5)."""
    faces = []
    for axis in range(3):
        for val in (0.0, 1.0):
            u = RNG.random((n_per_face, 2))
            pts = np.zeros((n_per_face, 3))
            other = [i for i in range(3) if i != axis]
            pts[:, other] = u
            pts[:, axis] = val
            faces.append(pts)
    return _add_noise(np.concatenate(faces, axis=0))


# ---- Cuboid ----


def generate_cuboid(
    n_total: int = 12000, sx: float = 1.0, sy: float = 0.6, sz: float = 0.4
) -> np.ndarray:
    """Rectangular box with different side lengths, centered at origin."""
    dims = np.array([sx, sy, sz])
    # Sample proportional to face area
    face_areas = [sy * sz, sx * sz, sx * sy]  # x-face, y-face, z-face
    total_area = 2 * sum(face_areas)
    faces = []
    for axis in range(3):
        n_face = int(n_total * face_areas[axis] / total_area)
        for val in (-dims[axis] / 2, dims[axis] / 2):
            u = RNG.random((n_face, 2))
            pts = np.zeros((n_face, 3))
            other = [i for i in range(3) if i != axis]
            pts[:, other[0]] = u[:, 0] * dims[other[0]] - dims[other[0]] / 2
            pts[:, other[1]] = u[:, 1] * dims[other[1]] - dims[other[1]] / 2
            pts[:, axis] = val
            faces.append(pts)
    return _add_noise(np.concatenate(faces, axis=0))


# ---- Sphere ----


def generate_sphere(n_points: int = 12000, radius: float = 0.5) -> np.ndarray:
    """Uniformly sample a sphere surface using Marsaglia method."""
    v = RNG.standard_normal((n_points, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return _add_noise(radius * v)


# ---- Cylinder ----


def generate_cylinder(
    n_points: int = 12000, radius: float = 0.4, half_height: float = 0.5
) -> np.ndarray:
    """Cylinder aligned to z-axis: lateral surface + two disk caps."""
    area_lateral = 2 * np.pi * radius * 2 * half_height
    area_cap = np.pi * radius**2
    total = area_lateral + 2 * area_cap
    n_lateral = int(n_points * area_lateral / total)
    n_cap = (n_points - n_lateral) // 2

    # Lateral surface
    theta = RNG.uniform(0, 2 * np.pi, n_lateral)
    z = RNG.uniform(-half_height, half_height, n_lateral)
    lateral = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), z])

    # Top and bottom caps (uniform disk sampling)
    caps = []
    for sign in (-1, 1):
        r = radius * np.sqrt(RNG.random(n_cap))
        t = RNG.uniform(0, 2 * np.pi, n_cap)
        cap = np.column_stack(
            [r * np.cos(t), r * np.sin(t), np.full(n_cap, sign * half_height)]
        )
        caps.append(cap)

    return _add_noise(np.concatenate([lateral] + caps, axis=0))


# ---- Cone ----


def generate_cone(
    n_points: int = 12000, radius: float = 0.45, height: float = 1.0
) -> np.ndarray:
    """Cone with apex at (0, 0, height/2), base circle at z = -height/2."""
    slant = np.sqrt(radius**2 + height**2)
    area_lateral = np.pi * radius * slant
    area_base = np.pi * radius**2
    total = area_lateral + area_base
    n_lateral = int(n_points * area_lateral / total)
    n_base = n_points - n_lateral

    # Lateral surface: parametric s in [0, 1] from apex to base rim
    # Weight sampling by circumference at that height → sqrt for uniform area
    s = np.sqrt(RNG.random(n_lateral))
    theta = RNG.uniform(0, 2 * np.pi, n_lateral)
    r_at_s = radius * s
    z_at_s = height / 2 - height * s  # apex to base
    lateral = np.column_stack([r_at_s * np.cos(theta), r_at_s * np.sin(theta), z_at_s])

    # Base disk
    r_b = radius * np.sqrt(RNG.random(n_base))
    t_b = RNG.uniform(0, 2 * np.pi, n_base)
    base = np.column_stack(
        [r_b * np.cos(t_b), r_b * np.sin(t_b), np.full(n_base, -height / 2)]
    )

    return _add_noise(np.concatenate([lateral, base], axis=0))


# ---- Torus ----


def generate_torus(
    n_points: int = 12000, major_r: float = 0.35, minor_r: float = 0.15
) -> np.ndarray:
    """Torus in the xy-plane."""
    theta = RNG.uniform(0, 2 * np.pi, n_points)
    phi = RNG.uniform(0, 2 * np.pi, n_points)
    x = (major_r + minor_r * np.cos(phi)) * np.cos(theta)
    y = (major_r + minor_r * np.cos(phi)) * np.sin(theta)
    z = minor_r * np.sin(phi)
    return _add_noise(np.column_stack([x, y, z]))


# ---- Pyramid ----


def generate_pyramid(
    n_points: int = 12000, base_half: float = 0.5, height: float = 1.0
) -> np.ndarray:
    """Square-base pyramid, apex at (0, 0, height/2), base at z = -height/2.
    4 triangular faces + 1 square base."""
    # Compute face areas for proportional sampling
    slant_h = np.sqrt(base_half**2 + height**2)
    area_tri = 0.5 * 2 * base_half * slant_h  # one triangular face
    area_base = (2 * base_half) ** 2
    total = 4 * area_tri + area_base

    n_base = int(n_points * area_base / total)
    n_tri = (n_points - n_base) // 4
    apex = np.array([0.0, 0.0, height / 2])

    # Base: uniform square at z = -height/2
    base_pts = np.column_stack(
        [
            RNG.uniform(-base_half, base_half, n_base),
            RNG.uniform(-base_half, base_half, n_base),
            np.full(n_base, -height / 2),
        ]
    )

    # Four triangular faces: each connects two adjacent base corners to the apex
    base_corners = np.array(
        [
            [-base_half, -base_half, -height / 2],
            [+base_half, -base_half, -height / 2],
            [+base_half, +base_half, -height / 2],
            [-base_half, +base_half, -height / 2],
        ]
    )
    tri_faces = []
    for i in range(4):
        c0 = base_corners[i]
        c1 = base_corners[(i + 1) % 4]
        # Uniform sampling on triangle via barycentric coordinates
        u = RNG.random((n_tri, 2))
        mask = u.sum(axis=1) > 1
        u[mask] = 1 - u[mask]
        pts = (1 - u[:, 0:1] - u[:, 1:2]) * apex + u[:, 0:1] * c0 + u[:, 1:2] * c1
        tri_faces.append(pts)

    return _add_noise(np.concatenate([base_pts] + tri_faces, axis=0))


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
    bunny = o3d.data.BunnyMesh()  # downloads on first call (~3 MB)
    mesh = o3d.io.read_triangle_mesh(bunny.path)
    mesh.compute_vertex_normals()

    # Sample 40 000 points from the mesh surface
    pcd = mesh.sample_points_uniformly(number_of_points=40000)
    o3d.io.write_point_cloud(str(dest), pcd)
    print(f"Saved bunny point cloud → {dest}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SHAPES = {
    "cone": generate_cone,
    "cube": generate_cube,
    "cuboid": generate_cuboid,
    "cylinder": generate_cylinder,
    "pyramid": generate_pyramid,
    "sphere": generate_sphere,
    "torus": generate_torus,
}


if __name__ == "__main__":
    # Synthetic datasets
    for name, fn in SHAPES.items():
        pts = fn()
        _save_ply(pts, SYNTHETIC_DIR / f"{name}.ply")

    # Real data
    download_bunny()

    # Real data
    download_bunny()

    print("\nAll sample data ready.")
    print("  Synthetic → data/Pointcloud/synthetic/")
    print("  Real      → data/Pointcloud/real/")
