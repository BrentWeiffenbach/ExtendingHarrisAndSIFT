"""
Joint k × k_neighbors sensitivity sweep for Harris PC.

For each (k, k_neighbors) combination this script runs Harris PC twice:
  - with the default surface variation filter (sv = 0.108)
  - with the filter disabled (sv = 0.0)

It records mean repeatability across all 7 synthetic shapes + rotation
perturbation, then renders three 2D heatmaps:

  1. Repeatability WITH sv filter
  2. Repeatability WITHOUT sv filter
  3. Absolute difference (sv_on − sv_off):
       ≈ 0  → sv filter is redundant in this region
       > 0  → sv filter is adding value

Output: outputs/demos/harris_pc_joint_sensitivity.png
"""

import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common.io import SyntheticPointCloudLoader
from src.common.metrics import match_keypoints, repeatability_score
from src.pointcloud.harris_pc import HarrisPC
from src.pointcloud.params import HarrisPCParams

# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------
K_VALUES = [0.001, 0.004, 0.010, 0.021, 0.030, 0.040]
KNN_VALUES = [20, 50, 100, 170, 250, 400]

ROTATION_ANGLES = [15, 30, 45]   # degrees, applied about Z
MATCH_RADIUS = 0.05
SV_ON  = 0.108
SV_OFF = 0.0

OUT_DIR = Path("outputs/demos")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(pts: np.ndarray) -> np.ndarray:
    lo, hi = pts.min(0), pts.max(0)
    scale = hi - lo
    scale[scale == 0] = 1.0
    return (pts - lo) / scale


def _rotate(pts: np.ndarray, deg: float) -> np.ndarray:
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    ctr = pts.mean(0, keepdims=True)
    return (pts - ctr) @ R.T + ctr


def _mean_repeatability(pts: np.ndarray, params: HarrisPCParams) -> float:
    """
    Mean repeatability over ROTATION_ANGLES rotations.
    Returns NaN if baseline has 0 keypoints.
    """
    detector = HarrisPC(params)
    baseline = detector.detect(pts)
    if len(baseline) == 0:
        return np.nan

    scores = []
    for angle in ROTATION_ANGLES:
        pts_r = _rotate(pts, angle)
        kps_r = detector.detect(pts_r)
        if len(kps_r) == 0:
            scores.append(0.0)
            continue
        # inverse-rotate detected kps back to baseline frame for matching
        ctr = pts.mean(0)
        kps_r_inv = _rotate(kps_r, -angle)
        kps_r_inv = (kps_r_inv - kps_r_inv.mean(0)) + ctr
        result = match_keypoints(baseline, kps_r_inv, radius=MATCH_RADIUS)
        rep = repeatability_score(len(baseline), len(kps_r), len(result.matched_reference))
        scores.append(rep)
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep():
    loader = SyntheticPointCloudLoader()
    shapes = [(name, _normalize(pts)) for name, pts in loader.load_all()]
    n_shapes = len(shapes)

    grid_shape = (len(K_VALUES), len(KNN_VALUES))
    rep_sv_on  = np.full(grid_shape, np.nan)
    rep_sv_off = np.full(grid_shape, np.nan)

    total = len(K_VALUES) * len(KNN_VALUES)
    done = 0

    for i, k in enumerate(K_VALUES):
        for j, knn in enumerate(KNN_VALUES):
            done += 1
            print(f"  [{done}/{total}]  k={k:.3f}  k_neighbors={knn}", flush=True)

            params_on  = HarrisPCParams(k=k, k_neighbors=knn, min_surface_variation=SV_ON,  max_keypoints=500)
            params_off = HarrisPCParams(k=k, k_neighbors=knn, min_surface_variation=SV_OFF, max_keypoints=500)

            reps_on, reps_off = [], []
            for name, pts in shapes:
                r_on  = _mean_repeatability(pts, params_on)
                r_off = _mean_repeatability(pts, params_off)
                if not np.isnan(r_on):
                    reps_on.append(r_on)
                if not np.isnan(r_off):
                    reps_off.append(r_off)

            rep_sv_on[i, j]  = np.mean(reps_on)  if reps_on  else np.nan
            rep_sv_off[i, j] = np.mean(reps_off) if reps_off else np.nan

    return rep_sv_on, rep_sv_off


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_plot(rep_sv_on, rep_sv_off):
    diff = rep_sv_on - rep_sv_off   # positive = sv helps

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        "Harris PC Joint Sensitivity: k  ×  k_neighbors\n"
        "Metric: mean repeatability across 7 synthetic shapes × 3 rotation angles",
        fontsize=12,
    )

    x_labels = [str(v) for v in KNN_VALUES]
    y_labels = [f"{v:.3f}" for v in K_VALUES]

    panel_data = [
        (rep_sv_on,  "Repeatability  (sv ON,  threshold=0.108)", "viridis", False),
        (rep_sv_off, "Repeatability  (sv OFF, threshold=0.0)",   "viridis", False),
        (diff,       "Difference  (sv_on − sv_off)\n"
                     "≈ 0 (white) → sv redundant | > 0 (green) → sv helps | < 0 (purple) → sv hurts",
                     "RdYlGn",   True),
    ]

    ims = []
    for ax, (data, title, cmap, diverging) in zip(axes, panel_data):
        vmin = -0.3 if diverging else 0.0
        vmax =  0.3 if diverging else 1.0
        im = ax.imshow(data, aspect="auto", origin="upper",
                       cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.set_xticks(range(len(KNN_VALUES)))
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_yticks(range(len(K_VALUES)))
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.set_xlabel("k_neighbors", fontsize=9)
        ax.set_ylabel("k  (Harris penalty)", fontsize=9)
        ax.set_title(title, fontsize=9, pad=8)

        # Annotate cells with values
        for ii in range(data.shape[0]):
            for jj in range(data.shape[1]):
                v = data[ii, jj]
                txt = "—" if np.isnan(v) else f"{v:.2f}"
                color = "white" if (not diverging and (np.isnan(v) or v < 0.5)) else "black"
                if diverging:
                    color = "black"
                ax.text(jj, ii, txt, ha="center", va="center", fontsize=7, color=color)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ims.append(im)

    # Overlay the "sv redundant" contour on the difference plot
    diff_clean = np.where(np.isnan(diff), 999, diff)
    axes[2].contour(diff_clean, levels=[0.05], colors=["black"], linewidths=1.5,
                    linestyles="--")
    axes[2].text(0.98, 0.02, "-- boundary: sv adds >0.05 rep",
                 transform=axes[2].transAxes, fontsize=7,
                 ha="right", va="bottom", color="black")

    plt.tight_layout()
    out = OUT_DIR / "harris_pc_joint_sensitivity.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved → {out}")
    return out


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running joint k × k_neighbors sweep …")
    rep_on, rep_off = run_sweep()

    print("\nRepeatability WITH sv filter:")
    print("k \\ knn  ", "  ".join(f"{v:>5}" for v in KNN_VALUES))
    for i, k in enumerate(K_VALUES):
        row = "  ".join(f"{rep_on[i,j]:5.2f}" if not np.isnan(rep_on[i,j]) else "  — " for j in range(len(KNN_VALUES)))
        print(f"  k={k:.3f}  {row}")

    print("\nRepeatability WITHOUT sv filter:")
    for i, k in enumerate(K_VALUES):
        row = "  ".join(f"{rep_off[i,j]:5.2f}" if not np.isnan(rep_off[i,j]) else "  — " for j in range(len(KNN_VALUES)))
        print(f"  k={k:.3f}  {row}")

    make_plot(rep_on, rep_off)
