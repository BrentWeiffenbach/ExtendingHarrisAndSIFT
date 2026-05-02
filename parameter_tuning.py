"""
parameter_tuning.py — Parameter sensitivity analysis for Harris and SIFT detectors.

Sweeps key parameters for five detectors one at a time (holding all others at defaults),
measures five metrics per run, aggregates across shapes, and saves CSVs + plots.

Usage:
    python parameter_tuning.py [--detector all] [--shapes cube cone ...] [--no-plots]
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.common.io import SyntheticPointCloudLoader, SyntheticVoxelLoader
from src.pointcloud.harris_pc import HarrisPC
from src.pointcloud.params import HarrisPCParams, SIFTRadiiPCParams, SIFTVoxelPCParams
from src.pointcloud.sift_pc import SIFTRadiiPC, SIFTVoxelPC
from src.voxel.harris3d import Harris3DVoxel
from src.voxel.params import Harris3DParams, SIFT3DParams
from src.voxel.sift3d import SIFT3DVoxel


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TuningResult:
    detector_name: str
    shape_name: str
    param_name: str
    param_value: str          # str() so list-valued params (radii) serialize cleanly
    keypoint_count: float
    spatial_coverage: float   # fraction of 4×4×4 bins occupied
    response_mean: float
    response_std: float
    response_max: float
    repeatability: float      # fraction of kps matched after adding noise
    runtime_s: float
    error: str = ""


@dataclass
class ParamSpec:
    name: str
    values: list              # sweep values; may contain sub-lists (radii)
    log_scale: bool = False   # x-axis hint for plots
    default_value: Any = None # drawn as red dashed line on plots


# ---------------------------------------------------------------------------
# Parameter sweep definitions
# ---------------------------------------------------------------------------

HARRIS3D_CONFIGS: dict[str, ParamSpec] = {
    "k": ParamSpec(
        "k",
        np.logspace(-3, -1, 7).tolist(),
        log_scale=True,
        default_value=0.02358,
    ),
    "gradient_sigma": ParamSpec(
        "gradient_sigma",
        [0.1, 0.2, 0.35, 0.48, 0.7, 1.0, 1.5],
        default_value=0.48266,
    ),
    "tensor_sigma": ParamSpec(
        "tensor_sigma",
        [0.2, 0.35, 0.5, 0.68, 1.0, 1.5, 2.0],
        default_value=0.67641,
    ),
    "threshold_rel": ParamSpec(
        "threshold_rel",
        np.logspace(-4, -1, 7).tolist(),
        log_scale=True,
        default_value=0.00387,
    ),
}

SIFT3D_CONFIGS: dict[str, ParamSpec] = {
    "num_octaves": ParamSpec("num_octaves", [1, 2, 3, 4], default_value=3),
    "scales_per_octave": ParamSpec("scales_per_octave", [3, 4, 5, 6, 7], default_value=5),
    "base_sigma": ParamSpec(
        "base_sigma",
        [0.8, 1.0, 1.2, 1.6, 2.0, 2.5],
        default_value=1.6,
    ),
    "extrema_contrast_threshold": ParamSpec(
        "extrema_contrast_threshold",
        [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
        default_value=0.3,
    ),
}

HARRIS_PC_CONFIGS: dict[str, ParamSpec] = {
    "k": ParamSpec(
        "k",
        np.logspace(-3, -1, 7).tolist(),
        log_scale=True,
        default_value=0.02,
    ),
    "k_neighbors": ParamSpec(
        "k_neighbors",
        [20, 50, 100, 170, 250, 400],
        default_value=170,
    ),
    "radius": ParamSpec(
        "radius",
        [0.01, 0.02, 0.05, 0.1, 0.15, 0.2],
        default_value=0.05,
    ),
}

_BASE_RADII = [0.05, 0.1, 0.2, 0.4]
SIFT_RADII_PC_CONFIGS: dict[str, ParamSpec] = {
    "num_octaves": ParamSpec("num_octaves", [1, 2, 3, 4], default_value=3),
    "radii": ParamSpec(
        "radii",
        [
            [r * s for r in _BASE_RADII]
            for s in [0.4, 0.6, 1.0, 1.6, 2.0]
        ],
        default_value=_BASE_RADII,
    ),
    "contrast_threshold": ParamSpec(
        "contrast_threshold",
        [0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8],
        default_value=0.45,
    ),
}

SIFT_VOXEL_PC_CONFIGS: dict[str, ParamSpec] = {
    "voxel_size": ParamSpec(
        "voxel_size",
        [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.15],
        default_value=0.05,
    ),
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _normalize_pointcloud(pts: np.ndarray) -> np.ndarray:
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    rng = hi - lo
    rng[rng == 0] = 1.0
    return (pts - lo) / rng


def load_voxel_data(shapes: list[str]) -> list[tuple[str, np.ndarray]]:
    loader = SyntheticVoxelLoader()
    all_shapes = loader.load_all()
    if shapes:
        all_shapes = [(n, v) for n, v in all_shapes if n in shapes]
    return all_shapes


def load_pc_data(shapes: list[str]) -> list[tuple[str, np.ndarray]]:
    loader = SyntheticPointCloudLoader()
    all_shapes = loader.load_all()
    if shapes:
        all_shapes = [(n, pts) for n, pts in all_shapes if n in shapes]
    return [(n, _normalize_pointcloud(pts.astype(np.float64))) for n, pts in all_shapes]


# ---------------------------------------------------------------------------
# Detector factories
# ---------------------------------------------------------------------------

def make_harris3d(overrides: dict) -> Harris3DVoxel:
    params = dataclasses.replace(Harris3DParams(), **overrides)
    return Harris3DVoxel(params)


def make_sift3d(overrides: dict) -> SIFT3DVoxel:
    params = dataclasses.replace(SIFT3DParams(), **overrides)
    return SIFT3DVoxel(params)


def make_harris_pc(overrides: dict) -> HarrisPC:
    params = dataclasses.replace(HarrisPCParams(), **overrides)
    return HarrisPC(params)


def make_sift_radii_pc(overrides: dict) -> SIFTRadiiPC:
    params = dataclasses.replace(SIFTRadiiPCParams(), **overrides)
    return SIFTRadiiPC(params)


def make_sift_voxel_pc(overrides: dict) -> SIFTVoxelPC:
    params = dataclasses.replace(SIFTVoxelPCParams(), **overrides)
    return SIFTVoxelPC(params)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _keypoint_count(kps: np.ndarray) -> int:
    return int(kps.shape[0])


def _spatial_coverage(kps: np.ndarray, data: np.ndarray, is_voxel: bool) -> float:
    if kps.shape[0] == 0:
        return 0.0
    n_bins = 4
    if is_voxel:
        bbox_min = np.zeros(3)
        bbox_max = np.array(data.shape[::-1], dtype=float)  # (W, H, D) = (x, y, z)
    else:
        bbox_min = data.min(axis=0)
        bbox_max = data.max(axis=0)
    extent = bbox_max - bbox_min
    extent[extent == 0] = 1.0
    bin_idx = np.floor((kps - bbox_min) / extent * n_bins).astype(int).clip(0, n_bins - 1)
    flat = bin_idx[:, 0] + n_bins * bin_idx[:, 1] + n_bins**2 * bin_idx[:, 2]
    counts = np.bincount(flat, minlength=n_bins**3)
    return float(np.count_nonzero(counts)) / float(n_bins**3)


def _response_stats_harris3d(
    detector: Harris3DVoxel, kps: np.ndarray
) -> tuple[float, float, float]:
    resp = getattr(detector, "last_response", None)
    if resp is None or kps.shape[0] == 0:
        return (np.nan, np.nan, np.nan)
    # kps are (x, y, z) → index as [z, y, x]
    r_vals = resp[kps[:, 2], kps[:, 1], kps[:, 0]].astype(float)
    return (float(np.mean(r_vals)), float(np.std(r_vals)), float(np.max(np.abs(r_vals))))


def _response_stats_harris_pc(detector: HarrisPC, kps: np.ndarray, data: np.ndarray) -> tuple[float, float, float]:
    resp = getattr(detector, "last_response", None)
    if resp is None or resp.size == 0 or kps.shape[0] == 0:
        return (np.nan, np.nan, np.nan)
    # Map detected keypoints back to the nearest original points so we can index per-point responses
    tree = cKDTree(data.astype(float))
    _, idx = tree.query(kps.astype(float), k=1)
    r_vals = resp[idx].astype(float)
    return (float(np.mean(r_vals)), float(np.std(r_vals)), float(np.max(np.abs(r_vals))))


def _bbox_diagonal(data: np.ndarray, is_voxel: bool) -> float:
    if is_voxel:
        return float(np.linalg.norm(np.array(data.shape, dtype=float)))
    else:
        return float(np.linalg.norm(data.max(axis=0) - data.min(axis=0)))


def _repeatability(
    make_fn: Callable,
    data: np.ndarray,
    is_voxel: bool,
    kps1: np.ndarray,
    noise_std: float = 0.01,
    tol_frac: float = 0.05,
) -> float:
    """Fraction of kps1 that have a nearest-neighbor in kps2 (noisy data) within tolerance."""
    if kps1.shape[0] == 0:
        return np.nan

    rng = np.random.default_rng(42)
    if is_voxel:
        data_noisy = np.clip(data.astype(float) + rng.normal(0, noise_std, data.shape), 0.0, 1.0)
    else:
        data_noisy = data + rng.normal(0, noise_std, data.shape)

    try:
        det2 = make_fn()
        kps2 = det2.detect(data_noisy)
    except Exception:
        return np.nan

    if kps2.shape[0] == 0:
        return 0.0

    tol = tol_frac * _bbox_diagonal(data, is_voxel)
    tree2 = cKDTree(kps2.astype(float))
    dists, _ = tree2.query(kps1.astype(float), k=1)
    return float(np.sum(dists <= tol)) / float(kps1.shape[0])


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def run_single_trial(
    detector_name: str,
    make_fn: Callable,
    param_name: str,
    param_value: Any,
    shape_name: str,
    data: np.ndarray,
    is_voxel: bool,
) -> TuningResult:
    nan6 = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    try:
        detector = make_fn()

        # SIFT detectors: call run() once to get both kps and response
        if detector_name == "sift3d":
            t0 = time.perf_counter()
            result = detector.run(data)
            rt = time.perf_counter() - t0
            eg = result.extrema_global
            if eg.shape[0] > 0:
                # extrema_global columns: [z, y, x, sigma_char, response, octave, dog_idx]
                kps = eg[:, [2, 1, 0]].astype(float)  # (x, y, z)
                r_vals = eg[:, 4].astype(float)
                r_mean, r_std, r_max = float(np.mean(r_vals)), float(np.std(r_vals)), float(np.max(np.abs(r_vals)))
            else:
                kps = np.empty((0, 3))
                r_mean, r_std, r_max = np.nan, np.nan, np.nan

        elif detector_name == "sift_radii_pc":
            t0 = time.perf_counter()
            result = detector.run(data)
            rt = time.perf_counter() - t0
            kp_full = result.keypoints  # (N, 5): [x, y, z, radius, dog_val]
            if kp_full.shape[0] > 0:
                kps = kp_full[:, :3].astype(float)
                r_vals = kp_full[:, 4].astype(float)
                r_mean, r_std, r_max = float(np.mean(r_vals)), float(np.std(r_vals)), float(np.max(np.abs(r_vals)))
            else:
                kps = np.empty((0, 3))
                r_mean, r_std, r_max = np.nan, np.nan, np.nan

        else:
            t0 = time.perf_counter()
            kps = detector.detect(data)
            rt = time.perf_counter() - t0
            kps = kps.astype(float) if kps.shape[0] > 0 else np.empty((0, 3))

        if detector_name == "harris3d":
            r_mean, r_std, r_max = _response_stats_harris3d(detector, kps.astype(int) if kps.shape[0] > 0 else kps)
        elif detector_name == "harris_pc":
            r_mean, r_std, r_max = _response_stats_harris_pc(detector, kps, data)
        else:  # sift_voxel_pc — response not exposed
            r_mean, r_std, r_max = np.nan, np.nan, np.nan

        count = _keypoint_count(kps)
        coverage = _spatial_coverage(kps, data, is_voxel)
        rep = _repeatability(make_fn, data, is_voxel, kps)

        return TuningResult(
            detector_name=detector_name,
            shape_name=shape_name,
            param_name=param_name,
            param_value=str(param_value),
            keypoint_count=float(count),
            spatial_coverage=coverage,
            response_mean=r_mean,
            response_std=r_std,
            response_max=r_max,
            repeatability=rep,
            runtime_s=rt,
        )

    except Exception as e:
        logging.warning(f"[{detector_name}] {shape_name} {param_name}={param_value}: {e}")
        return TuningResult(
            detector_name=detector_name,
            shape_name=shape_name,
            param_name=param_name,
            param_value=str(param_value),
            keypoint_count=np.nan,
            spatial_coverage=np.nan,
            response_mean=np.nan,
            response_std=np.nan,
            response_max=np.nan,
            repeatability=np.nan,
            runtime_s=np.nan,
            error=str(e),
        )


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def run_detector_sweep(
    detector_name: str,
    shapes_data: list[tuple[str, np.ndarray]],
    is_voxel: bool,
    configs: dict[str, ParamSpec],
    base_factory: Callable,
    verbose: bool = True,
) -> list[TuningResult]:
    results: list[TuningResult] = []
    for param_name, spec in configs.items():
        if verbose:
            logging.info(
                f"  [{detector_name}] sweeping '{param_name}': "
                f"{len(spec.values)} values × {len(shapes_data)} shapes"
            )
        for val in spec.values:
            overrides: dict[str, Any] = {param_name: val}
            # radius sweep on HarrisPC requires switching neighborhood mode
            if detector_name == "harris_pc" and param_name == "radius":
                overrides["neighborhood"] = "radius"
            # capture overrides by value with default arg trick
            make_fn: Callable = lambda o=overrides: base_factory(o)
            for shape_name, data in shapes_data:
                trial = run_single_trial(
                    detector_name, make_fn, param_name, val,
                    shape_name, data, is_voxel,
                )
                results.append(trial)
    return results


# ---------------------------------------------------------------------------
# Aggregation and CSV
# ---------------------------------------------------------------------------

METRIC_COLS = [
    "keypoint_count", "spatial_coverage",
    "response_mean", "response_std", "response_max",
    "repeatability", "runtime_s",
]

METRIC_LABELS = {
    "keypoint_count":   "Keypoint Count",
    "spatial_coverage": "Spatial Coverage (4³ bins)",
    "response_mean":    "Response Mean",
    "response_std":     "Response Std",
    "response_max":     "Response Max",
    "repeatability":    "Repeatability",
    "runtime_s":        "Runtime (s)",
}


def aggregate_to_dataframe(results: list[TuningResult]) -> pd.DataFrame:
    return pd.DataFrame([dataclasses.asdict(r) for r in results])


def save_csvs(df: pd.DataFrame, detector_name: str, out_root: Path) -> None:
    out_dir = out_root / detector_name
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "results.csv", index=False)

    agg = (
        df.groupby(["detector_name", "param_name", "param_value"])[METRIC_COLS]
        .mean()
        .reset_index()
    )
    agg.to_csv(out_dir / "results_aggregated.csv", index=False)
    logging.info(f"  CSVs saved → {out_dir}/")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _param_x_values(spec: ParamSpec) -> tuple[list[float], str]:
    """Convert spec.values to scalar x coordinates and a label."""
    vals = spec.values
    if isinstance(vals[0], list):
        # radii: use geometric mean of first and last element as scalar summary
        x = [float(np.sqrt(v[0] * v[-1])) for v in vals]
        label = f"{spec.name} (√(r_min × r_max))"
    else:
        x = [float(v) for v in vals]
        label = spec.name
    return x, label


def _default_x(spec: ParamSpec, x: list[float]) -> float | None:
    """Return the x-axis position of the default value, or None if not found."""
    dv = spec.default_value
    if dv is None:
        return None
    if isinstance(dv, list):
        target = float(np.sqrt(dv[0] * dv[-1]))
    else:
        target = float(dv)
    # Find nearest x value
    diffs = [abs(xi - target) for xi in x]
    return x[int(np.argmin(diffs))]


def plot_param_sweep(
    df_raw: pd.DataFrame,
    detector_name: str,
    param_name: str,
    spec: ParamSpec,
    out_dir: Path,
) -> None:
    sub = df_raw[
        (df_raw["detector_name"] == detector_name) &
        (df_raw["param_name"] == param_name)
    ].copy()

    x, x_label = _param_x_values(spec)
    metrics_to_plot = [m for m in METRIC_COLS if m != "response_max"]  # 6 subplots

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes_flat = axes.flatten()
    fig.suptitle(f"{detector_name} — {param_name} sensitivity", fontsize=13, fontweight="bold")

    for ax, metric in zip(axes_flat, metrics_to_plot):
        means, stds = [], []
        for val in spec.values:
            mask = sub["param_value"] == str(val)
            col = sub.loc[mask, metric].dropna()
            means.append(col.mean() if len(col) > 0 else np.nan)
            stds.append(col.std() if len(col) > 1 else 0.0)

        means_arr = np.array(means, dtype=float)
        stds_arr = np.array(stds, dtype=float)

        valid = ~np.isnan(means_arr)
        if valid.any():
            x_arr = np.array(x)
            ax.plot(x_arr[valid], means_arr[valid], marker="o", linewidth=1.8, color="steelblue")
            ax.fill_between(
                x_arr[valid],
                (means_arr - stds_arr)[valid],
                (means_arr + stds_arr)[valid],
                alpha=0.2, color="steelblue",
            )

        def_x = _default_x(spec, x)
        if def_x is not None:
            ax.axvline(def_x, color="red", linestyle="--", alpha=0.6, linewidth=1.2, label="default")
            ax.legend(fontsize=8, loc="best")

        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=9)
        ax.set_title(METRIC_LABELS[metric], fontsize=10)
        if spec.log_scale and len(x) > 1 and all(xi > 0 for xi in x):
            ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{param_name}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_detector(
    df_raw: pd.DataFrame,
    detector_name: str,
    configs: dict[str, ParamSpec],
    out_root: Path,
) -> None:
    out_dir = out_root / detector_name
    for param_name, spec in configs.items():
        plot_param_sweep(df_raw, detector_name, param_name, spec, out_dir)
    logging.info(f"  Plots saved → {out_dir}/")


# ---------------------------------------------------------------------------
# Detector registry
# ---------------------------------------------------------------------------

DETECTOR_REGISTRY: dict[str, dict] = {
    "harris3d": {
        "is_voxel": True,
        "configs": HARRIS3D_CONFIGS,
        "factory": make_harris3d,
    },
    "sift3d": {
        "is_voxel": True,
        "configs": SIFT3D_CONFIGS,
        "factory": make_sift3d,
    },
    "harris_pc": {
        "is_voxel": False,
        "configs": HARRIS_PC_CONFIGS,
        "factory": make_harris_pc,
    },
    "sift_radii_pc": {
        "is_voxel": False,
        "configs": SIFT_RADII_PC_CONFIGS,
        "factory": make_sift_radii_pc,
    },
    "sift_voxel_pc": {
        "is_voxel": False,
        "configs": SIFT_VOXEL_PC_CONFIGS,
        "factory": make_sift_voxel_pc,
    },
}

ALL_SHAPES = ["cone", "cube", "cuboid", "cylinder", "pyramid", "sphere", "torus"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(
        description="Parameter sensitivity analysis for Harris and SIFT detectors"
    )
    parser.add_argument(
        "--detector",
        nargs="+",
        choices=list(DETECTOR_REGISTRY.keys()) + ["all"],
        default=["all"],
        metavar="DETECTOR",
        help=(
            "Detector(s) to tune. Choices: "
            + ", ".join(list(DETECTOR_REGISTRY.keys()) + ["all"])
            + " (default: all)"
        ),
    )
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=ALL_SHAPES,
        metavar="SHAPE",
        help=f"Shapes to evaluate on (default: all 7). Choices: {ALL_SHAPES}",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating matplotlib plots",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/parameter_tuning",
        metavar="PATH",
        help="Root output directory (default: outputs/parameter_tuning)",
    )
    args = parser.parse_args()

    detectors_to_run = (
        list(DETECTOR_REGISTRY.keys())
        if "all" in args.detector
        else args.detector
    )
    out_root = Path(args.output_dir)

    # Load data once
    logging.info("Loading voxel data...")
    voxel_data = load_voxel_data(args.shapes)
    logging.info(f"  Loaded {len(voxel_data)} voxel shapes")

    logging.info("Loading point cloud data...")
    pc_data = load_pc_data(args.shapes)
    logging.info(f"  Loaded {len(pc_data)} point cloud shapes")

    for det_name in detectors_to_run:
        logging.info(f"=== Tuning '{det_name}' ===")
        reg = DETECTOR_REGISTRY[det_name]
        shapes_data = voxel_data if reg["is_voxel"] else pc_data

        if not shapes_data:
            logging.warning(f"  No data available for {det_name}, skipping.")
            continue

        results = run_detector_sweep(
            detector_name=det_name,
            shapes_data=shapes_data,
            is_voxel=reg["is_voxel"],
            configs=reg["configs"],
            base_factory=reg["factory"],
        )

        df_raw = aggregate_to_dataframe(results)
        save_csvs(df_raw, det_name, out_root)

        if not args.no_plots:
            plot_detector(df_raw, det_name, reg["configs"], out_root)

    logging.info("Done.")


if __name__ == "__main__":
    main()
