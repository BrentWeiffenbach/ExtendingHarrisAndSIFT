"""
Generate synthetic 3D voxel grids (.npy) for basic geometric shapes.

Each shape is a binary (bool) numpy array of shape (SIZE, SIZE, SIZE).
Saved to the same directory as: cube.npy, cuboid.npy, sphere.npy,
cylinder.npy, cone.npy, torus.npy, pyramid.npy
"""

import os
import numpy as np

SIZE = 32  # voxel grid resolution
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic")


def _grid():
    """Return coordinate arrays centered in [0, SIZE)."""
    coords = np.arange(SIZE)
    z, y, x = np.meshgrid(coords, coords, coords, indexing="ij")
    # Shift so the center of the grid is at origin
    cx = cy = cz = (SIZE - 1) / 2.0
    return x - cx, y - cy, z - cz


def make_cube(side=0.55):
    """Axis-aligned cube. `side` as a fraction of grid half-extent."""
    x, y, z = _grid()
    half = side * SIZE / 2.0
    return (np.abs(x) <= half) & (np.abs(y) <= half) & (np.abs(z) <= half)


def make_cuboid(sx=0.8, sy=0.45, sz=0.35):
    """Axis-aligned rectangular box. sx/sy/sz as fractions of half-extent."""
    x, y, z = _grid()
    hx, hy, hz = sx * SIZE / 2.0, sy * SIZE / 2.0, sz * SIZE / 2.0
    return (np.abs(x) <= hx) & (np.abs(y) <= hy) & (np.abs(z) <= hz)


def make_sphere(radius=0.45):
    """Filled sphere. `radius` as a fraction of half-extent."""
    x, y, z = _grid()
    r = radius * SIZE / 2.0
    return (x**2 + y**2 + z**2) <= r**2


def make_cylinder(radius=0.35, half_height=0.45, axis="z"):
    """Filled cylinder aligned to `axis` ('x', 'y', or 'z')."""
    x, y, z = _grid()
    r = radius * SIZE / 2.0
    h = half_height * SIZE / 2.0
    if axis == "z":
        return (x**2 + y**2 <= r**2) & (np.abs(z) <= h)
    elif axis == "y":
        return (x**2 + z**2 <= r**2) & (np.abs(y) <= h)
    else:
        return (y**2 + z**2 <= r**2) & (np.abs(x) <= h)


def make_cone(radius=0.45, height=0.9, tip_up=True):
    """Filled cone with apex pointing up (+z) or down."""
    x, y, z = _grid()
    h = height * SIZE / 2.0
    r = radius * SIZE / 2.0
    # Map z in [-h, h] to a local t in [0, 1] from base to apex
    if tip_up:
        t = (z + h) / (2.0 * h)           # 0 at bottom, 1 at top
    else:
        t = (h - z) / (2.0 * h)
    t = np.clip(t, 0, 1)
    r_at_z = r * (1.0 - t)                # radius shrinks toward apex
    return (x**2 + y**2 <= r_at_z**2) & (np.abs(z) <= h)


def make_torus(major_r=0.35, minor_r=0.15):
    """Solid torus in the xy-plane.
    major_r: distance from center to tube center (fraction of half-extent).
    minor_r: tube radius (fraction of half-extent).
    """
    x, y, z = _grid()
    R = major_r * SIZE / 2.0
    r = minor_r * SIZE / 2.0
    dist_to_ring = (np.sqrt(x**2 + y**2) - R)**2 + z**2
    return dist_to_ring <= r**2


def make_pyramid(base=0.7, height=0.8):
    """Square-base pyramid. `base` and `height` as fractions of extent."""
    x, y, z = _grid()
    h = height * SIZE / 2.0
    half_b = base * SIZE / 2.0
    # z runs from -h (base) to +h (apex)
    t = np.clip((z + h) / (2.0 * h), 0, 1)   # 0 at base, 1 at apex
    half_at_z = half_b * (1.0 - t)
    return (np.abs(x) <= half_at_z) & (np.abs(y) <= half_at_z) & (np.abs(z) <= h)


SHAPES = {
    "cube":     make_cube,
    "cuboid":   make_cuboid,
    "sphere":   make_sphere,
    "cylinder": make_cylinder,
    "cone":     make_cone,
    "torus":    make_torus,
    "pyramid":  make_pyramid,
}


def main():
    for name, fn in SHAPES.items():
        voxel = fn().astype(bool)
        path = os.path.join(OUT_DIR, f"{name}.npy")
        np.save(path, voxel)
        filled = voxel.sum()
        total = voxel.size
        print(f"Saved {path}  shape={voxel.shape}  filled={filled}/{total} ({100*filled/total:.1f}%)")


if __name__ == "__main__":
    main()
