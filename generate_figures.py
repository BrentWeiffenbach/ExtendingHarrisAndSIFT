"""
Generate publication-quality grid figures for all detectors and datasets.

Outputs (all in outputs/figures/):
  harris_voxel_synthetic.png   — Harris3D on 7 synthetic shapes (3×3 grid, last cell blank)
  harris_voxel_real.png        — Harris3D on ModelNet 5-10 (3×2 grid)
  harris_pc_synthetic.png      — HarrisPC on 7 synthetic shapes
  harris_pc_real.png           — HarrisPC on real point clouds
  harris_pc_robustness.png     — HarrisPC perturbation samples (rotation/noise/downsample)
  harris_voxel_robustness.png  — Harris3D perturbation samples
  sift_voxel_synthetic.png     — SIFT3D on 7 synthetic shapes
  sift_voxel_real.png          — SIFT3D on ModelNet 5-10
  sift_pc_synthetic.png        — SIFT-Geom PC on 7 synthetic shapes
  sift_pc_real.png             — SIFT-Geom PC on real point clouds
  sift_pc_robustness.png       — SIFT-Geom PC perturbation samples
  sift_voxel_robustness.png    — SIFT3D perturbation samples

Run from workspace root:
    python generate_figures.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.ndimage import rotate as ndimage_rotate
from scipy.ndimage import zoom
from skimage.measure import marching_cubes

from src.common.io import (
    ModelNetLoader,
    RealPointCloudLoader,
    SyntheticPointCloudLoader,
    SyntheticVoxelLoader,
)
from src.common.visualization import _shift_keypoints_to_surface_corners
from src.pointcloud.harris_pc import HarrisPC
from src.pointcloud.params import SIFTRadiiPCParams, default_harris_pc_params
from src.pointcloud.sift_pc import SIFTRadiiPC
from src.voxel.harris3d import Harris3DVoxel
from src.voxel.params import SIFT3DParams, default_harris3d_params
from src.voxel.sift3d import SIFT3DVoxel

OUT_DIR = Path("outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELNET_PATH = "data/Voxel/real/ModelNet10-dataset/modelnet10.npy.gz"
DPI = 150

# ─────────────────────────────────────────────────────────────────────────────
# Low-level drawing helpers
# ─────────────────────────────────────────────────────────────────────────────


def _draw_voxel(ax, vol: np.ndarray, kps: Optional[np.ndarray], title: str) -> None:
    """Render a single voxel volume with marching-cubes surface + cross keypoints."""
    vol_bool = np.asarray(vol).astype(bool)
    if vol_bool.any():
        padded = np.pad(vol_bool.astype(np.float32), 1, constant_values=0)
        verts, faces, _, _ = marching_cubes(padded, level=0.5)
        verts -= 1
        ax.plot_trisurf(
            verts[:, 2],
            verts[:, 1],
            faces,
            verts[:, 0],
            color="steelblue",
            alpha=0.45,
            linewidth=0,
        )

    if kps is not None and len(kps) > 0:
        kp = np.asarray(kps)
        kp_plot = _shift_keypoints_to_surface_corners(vol_bool, kp[:, :3])
        ax.scatter(
            kp_plot[:, 0],
            kp_plot[:, 1],
            kp_plot[:, 2],
            c="red",
            marker="+",
            s=220,
            linewidths=2.5,
            depthshade=False,
            zorder=10,
        )

    ax.view_init(elev=28, azim=45)
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_axis_off()


def _draw_pc(
    ax, pts: np.ndarray, kps: Optional[np.ndarray], title: str, stride: int = 6
) -> None:
    """Render a single point cloud with cross keypoints."""
    ax.scatter(
        pts[::stride, 0],
        pts[::stride, 1],
        pts[::stride, 2],
        s=1,
        c=pts[::stride, 2],
        cmap="viridis",
        alpha=0.5,
        depthshade=False,
    )
    if kps is not None and len(kps) > 0:
        kp = np.asarray(kps)
        ax.scatter(
            kp[:, 0],
            kp[:, 1],
            kp[:, 2],
            c="red",
            marker="+",
            s=280,
            linewidths=2.8,
            depthshade=False,
            zorder=10,
        )
    ax.view_init(elev=28, azim=45)
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_axis_off()


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path.relative_to(Path('.'))}")


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────


def _norm_pc(pts: np.ndarray) -> np.ndarray:
    lo, hi = pts.min(0), pts.max(0)
    rng = hi - lo
    rng[rng == 0] = 1.0
    return (pts - lo) / rng


def load_synthetic_voxels():
    return SyntheticVoxelLoader().load_all()


def load_modelnet(start=5, end=11):
    loader = ModelNetLoader(MODELNET_PATH)
    return [(f"modelnet_{i}", vol) for i, vol in loader.load_range(start, end)]


def load_synthetic_pc():
    return [(n, _norm_pc(p)) for n, p in SyntheticPointCloudLoader().load_all()]


def load_real_pc():
    return [(n, _norm_pc(p)) for n, p in RealPointCloudLoader().load_all()]


# ─────────────────────────────────────────────────────────────────────────────
# Grid builders
# ─────────────────────────────────────────────────────────────────────────────


def _voxel_grid(
    samples, detector_fn, title: str, rows: int, cols: int, out_path: Path
) -> None:
    """Build rows×cols voxel grid figure."""
    fig = plt.figure(figsize=(cols * 3.5, rows * 3.5))
    fig.suptitle(title, fontsize=12, y=1.01)
    for idx, (name, vol) in enumerate(samples):
        if idx >= rows * cols:
            break
        kps = detector_fn(vol)
        ax = fig.add_subplot(rows, cols, idx + 1, projection="3d")
        _draw_voxel(ax, vol, kps, name)
    # blank remaining cells
    for idx in range(len(samples), rows * cols):
        ax = fig.add_subplot(rows, cols, idx + 1)
        ax.set_visible(False)
    plt.tight_layout()
    _save(fig, out_path)


def _pc_grid(
    samples, detector_fn, title: str, rows: int, cols: int, out_path: Path
) -> None:
    """Build rows×cols PC grid figure."""
    fig = plt.figure(figsize=(cols * 3.5, rows * 3.5))
    fig.suptitle(title, fontsize=12, y=1.01)
    for idx, (name, pts) in enumerate(samples):
        if idx >= rows * cols:
            break
        kps = detector_fn(pts)
        ax = fig.add_subplot(rows, cols, idx + 1, projection="3d")
        _draw_pc(ax, pts, kps, name)
    for idx in range(len(samples), rows * cols):
        ax = fig.add_subplot(rows, cols, idx + 1)
        ax.set_visible(False)
    plt.tight_layout()
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Robustness grid helpers
# ─────────────────────────────────────────────────────────────────────────────


def _rotate_vol(vol: np.ndarray, angle: float) -> np.ndarray:
    return (
        ndimage_rotate(vol.astype(float), angle, axes=(1, 2), reshape=False, order=0)
        .astype(bool)
        .astype(float)
    )


def _noise_vol(vol: np.ndarray, flip_prob: float, rng) -> np.ndarray:
    mask = rng.random(vol.shape) < flip_prob
    return np.logical_xor(vol.astype(bool), mask).astype(float)


def _downsample_vol(vol: np.ndarray, factor: float) -> np.ndarray:
    small = zoom(vol.astype(float), factor, order=0)
    back = zoom(small, 1.0 / factor, order=0)
    # crop or pad to original shape
    out = np.zeros_like(vol, dtype=float)
    sl = tuple(slice(0, min(s, b)) for s, b in zip(vol.shape, back.shape))
    out[sl] = back[sl]
    return out


def _rotate_pc(pts: np.ndarray, angle: float) -> np.ndarray:
    rad = np.deg2rad(angle)
    c, s = np.cos(rad), np.sin(rad)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    ctr = pts.mean(0, keepdims=True)
    return (pts - ctr) @ R.T + ctr


def _noise_pc(pts: np.ndarray, sigma: float, rng) -> np.ndarray:
    return pts + rng.normal(0, sigma, pts.shape)


def _downsample_pc(pts: np.ndarray, keep: float, rng) -> np.ndarray:
    n = max(8, int(round(keep * len(pts))))
    idx = rng.choice(len(pts), size=min(n, len(pts)), replace=False)
    return pts[idx]


def _robustness_voxel(
    shape_name: str, vol: np.ndarray, detector_fn, rng, out_path: Path
) -> None:
    """3×4 grid: rows=perturbation type, cols=baseline+3 levels."""
    rot_angles = [15, 30, 45]
    noise_probs = [0.02, 0.05, 0.10]
    ds_factors = [0.85, 0.70, 0.50]

    rows_data = [
        (
            "Rotation",
            [vol] + [_rotate_vol(vol, a) for a in rot_angles],
            ["baseline"] + [f"{a}°" for a in rot_angles],
        ),
        (
            "Noise",
            [vol] + [_noise_vol(vol, p, rng) for p in noise_probs],
            ["baseline"] + [f"p={p}" for p in noise_probs],
        ),
        (
            "Downsample",
            [vol] + [_downsample_vol(vol, f) for f in ds_factors],
            ["baseline"] + [f"×{f}" for f in ds_factors],
        ),
    ]

    fig = plt.figure(figsize=(4 * 4, 3 * 3.5))
    fig.suptitle(f"Harris3D robustness — {shape_name}", fontsize=12, y=1.01)
    for row_i, (label, vols, subtitles) in enumerate(rows_data):
        for col_i, (v, st) in enumerate(zip(vols, subtitles)):
            ax = fig.add_subplot(3, 4, row_i * 4 + col_i + 1, projection="3d")
            kps = detector_fn(v)
            full_title = f"{label}\n{st}  n={len(kps)}"
            _draw_voxel(ax, v, kps, full_title)
    plt.tight_layout()
    _save(fig, out_path)


def _robustness_pc(
    shape_name: str, pts: np.ndarray, detector_fn, rng, out_path: Path
) -> None:
    """3×4 grid: rows=perturbation type, cols=baseline+3 levels."""
    rot_angles = [15, 30, 45]
    noise_sigmas = [0.01, 0.03, 0.06]
    ds_ratios = [0.85, 0.70, 0.50]

    rows_data = [
        (
            "Rotation",
            [pts] + [_rotate_pc(pts, a) for a in rot_angles],
            ["baseline"] + [f"{a}°" for a in rot_angles],
        ),
        (
            "Noise",
            [pts] + [_noise_pc(pts, s, rng) for s in noise_sigmas],
            ["baseline"] + [f"σ={s}" for s in noise_sigmas],
        ),
        (
            "Downsample",
            [pts] + [_downsample_pc(pts, r, rng) for r in ds_ratios],
            ["baseline"] + [f"×{r}" for r in ds_ratios],
        ),
    ]

    fig = plt.figure(figsize=(4 * 4, 3 * 3.5))
    fig.suptitle(f"Harris PC robustness — {shape_name}", fontsize=12, y=1.01)
    for row_i, (label, clouds, subtitles) in enumerate(rows_data):
        for col_i, (p, st) in enumerate(zip(clouds, subtitles)):
            ax = fig.add_subplot(3, 4, row_i * 4 + col_i + 1, projection="3d")
            kps = detector_fn(p)
            full_title = f"{label}\n{st}  n={len(kps)}"
            _draw_pc(ax, p, kps, full_title)
    plt.tight_layout()
    _save(fig, out_path)


def _robustness_sift3d(
    shape_name: str, vol: np.ndarray, detector_fn, rng, out_path: Path
) -> None:
    rot_angles = [15, 30, 45]
    noise_probs = [0.02, 0.05, 0.10]
    ds_factors = [0.85, 0.70, 0.50]

    rows_data = [
        (
            "Rotation",
            [vol] + [_rotate_vol(vol, a) for a in rot_angles],
            ["baseline"] + [f"{a}°" for a in rot_angles],
        ),
        (
            "Noise",
            [vol] + [_noise_vol(vol, p, rng) for p in noise_probs],
            ["baseline"] + [f"p={p}" for p in noise_probs],
        ),
        (
            "Downsample",
            [vol] + [_downsample_vol(vol, f) for f in ds_factors],
            ["baseline"] + [f"×{f}" for f in ds_factors],
        ),
    ]

    fig = plt.figure(figsize=(4 * 4, 3 * 3.5))
    fig.suptitle(f"SIFT3D robustness — {shape_name}", fontsize=12, y=1.01)
    for row_i, (label, vols, subtitles) in enumerate(rows_data):
        for col_i, (v, st) in enumerate(zip(vols, subtitles)):
            ax = fig.add_subplot(3, 4, row_i * 4 + col_i + 1, projection="3d")
            kps = detector_fn(v)
            full_title = f"{label}\n{st}  n={len(kps)}"
            _draw_voxel(ax, v, kps, full_title)
    plt.tight_layout()
    _save(fig, out_path)


def _robustness_sift_pc(
    shape_name: str, pts: np.ndarray, detector_fn, rng, out_path: Path
) -> None:
    rot_angles = [15, 30, 45]
    noise_sigmas = [0.01, 0.03, 0.06]
    ds_ratios = [0.85, 0.70, 0.50]

    rows_data = [
        (
            "Rotation",
            [pts] + [_rotate_pc(pts, a) for a in rot_angles],
            ["baseline"] + [f"{a}°" for a in rot_angles],
        ),
        (
            "Noise",
            [pts] + [_noise_pc(pts, s, rng) for s in noise_sigmas],
            ["baseline"] + [f"σ={s}" for s in noise_sigmas],
        ),
        (
            "Downsample",
            [pts] + [_downsample_pc(pts, r, rng) for r in ds_ratios],
            ["baseline"] + [f"×{r}" for r in ds_ratios],
        ),
    ]

    fig = plt.figure(figsize=(4 * 4, 3 * 3.5))
    fig.suptitle(f"SIFT Radii PC robustness — {shape_name}", fontsize=12, y=1.01)
    for row_i, (label, clouds, subtitles) in enumerate(rows_data):
        for col_i, (p, st) in enumerate(zip(clouds, subtitles)):
            ax = fig.add_subplot(3, 4, row_i * 4 + col_i + 1, projection="3d")
            kps = detector_fn(p)
            full_title = f"{label}\n{st}  n={len(kps)}"
            _draw_pc(ax, p, kps, full_title)
    plt.tight_layout()
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Detector wrappers
# ─────────────────────────────────────────────────────────────────────────────


def _harris3d_detect(vol):
    return Harris3DVoxel(default_harris3d_params()).detect(vol)


def _sift3d_detect(vol):
    result = SIFT3DVoxel(SIFT3DParams()).run(vol)
    if result.extrema_global.shape[0] > 0:
        return result.extrema_global[:, [2, 1, 0]].astype(np.int32)
    return np.empty((0, 3), dtype=np.int32)


def _harris_pc_detect(pts):
    return HarrisPC(default_harris_pc_params()).detect(pts)


def _sift_pc_detect(pts):
    return SIFTRadiiPC(SIFTRadiiPCParams()).detect(pts)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    rng = np.random.default_rng(42)

    print("Loading datasets…")
    syn_vox = load_synthetic_voxels()  # 7 shapes
    real_vox = load_modelnet(5, 11)  # 6 samples (indices 5–10)
    syn_pc = load_synthetic_pc()  # 7 shapes
    real_pc = load_real_pc()  # 4 shapes

    # ── Harris3D ──────────────────────────────────────────────────────────
    print("\n[1/8] Harris3D synthetic (3×3 grid)…")
    _voxel_grid(
        syn_vox,
        _harris3d_detect,
        "Harris3D — Synthetic Voxel Shapes",
        rows=3,
        cols=3,
        out_path=OUT_DIR / "harris_voxel_synthetic.png",
    )

    print("[2/8] Harris3D ModelNet (3×2 grid)…")
    _voxel_grid(
        real_vox,
        _harris3d_detect,
        "Harris3D — ModelNet10 Samples (indices 5–10)",
        rows=3,
        cols=2,
        out_path=OUT_DIR / "harris_voxel_real.png",
    )

    print("[3/8] Harris PC synthetic (3×3 grid)…")
    _pc_grid(
        syn_pc,
        _harris_pc_detect,
        "Harris PC — Synthetic Point Cloud Shapes",
        rows=3,
        cols=3,
        out_path=OUT_DIR / "harris_pc_synthetic.png",
    )

    print("[4/8] Harris PC real (2×2 grid)…")
    _pc_grid(
        real_pc,
        _harris_pc_detect,
        "Harris PC — Real Point Clouds",
        rows=2,
        cols=2,
        out_path=OUT_DIR / "harris_pc_real.png",
    )

    # ── SIFT3D ────────────────────────────────────────────────────────────
    print("[5/8] SIFT3D synthetic (3×3 grid)…")
    _voxel_grid(
        syn_vox,
        _sift3d_detect,
        "SIFT3D — Synthetic Voxel Shapes",
        rows=3,
        cols=3,
        out_path=OUT_DIR / "sift_voxel_synthetic.png",
    )

    print("[6/8] SIFT3D ModelNet (3×2 grid)…")
    _voxel_grid(
        real_vox,
        _sift3d_detect,
        "SIFT3D — ModelNet10 Samples (indices 5–10)",
        rows=3,
        cols=2,
        out_path=OUT_DIR / "sift_voxel_real.png",
    )

    print("[7/8] SIFT Radii PC synthetic (3×3 grid)…")
    _pc_grid(
        syn_pc,
        _sift_pc_detect,
        "SIFT Radii PC — Synthetic Point Cloud Shapes",
        rows=3,
        cols=3,
        out_path=OUT_DIR / "sift_pc_synthetic.png",
    )

    print("[8/8] SIFT Radii PC real (2×2 grid)…")
    _pc_grid(
        real_pc,
        _sift_pc_detect,
        "SIFT Radii PC — Real Point Clouds",
        rows=2,
        cols=2,
        out_path=OUT_DIR / "sift_pc_real.png",
    )

    # ── Robustness on cube (representative shape) ─────────────────────────
    print("\nGenerating robustness figures (cube shape)…")
    cube_vol = dict(syn_vox)["cube"].astype(float)
    cube_pts = dict(syn_pc)["cube"]

    print("  Harris3D robustness…")
    _robustness_voxel(
        "cube", cube_vol, _harris3d_detect, rng, OUT_DIR / "harris_voxel_robustness.png"
    )

    print("  Harris PC robustness…")
    _robustness_pc(
        "cube", cube_pts, _harris_pc_detect, rng, OUT_DIR / "harris_pc_robustness.png"
    )

    print("  SIFT3D robustness…")
    _robustness_sift3d(
        "cube", cube_vol, _sift3d_detect, rng, OUT_DIR / "sift_voxel_robustness.png"
    )

    print("  SIFT Radii PC robustness…")
    _robustness_sift_pc(
        "cube", cube_pts, _sift_pc_detect, rng, OUT_DIR / "sift_pc_robustness.png"
    )

    print(f"\nAll figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
