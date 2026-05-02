from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

from src.common.io import RealPointCloudLoader, SyntheticPointCloudLoader
from src.common.metrics import (
    keypoint_count_stability,
    localization_error,
    match_keypoints,
    repeatability_score,
)
from src.common.visualization import plot_pointcloud
from src.pointcloud.harris_pc import HarrisPC
from src.pointcloud.params import (
    HarrisPCParams,
    SIFTRadiiPCParams,
    default_harris_pc_params,
)
from src.pointcloud.sift_pc import SIFTRadiiPC

PointCloudEntry = tuple[str, np.ndarray]


def _normalize_points(pts: np.ndarray) -> np.ndarray:
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    scale = hi - lo
    scale[scale == 0.0] = 1.0
    return (pts - lo) / scale


def _rotate_points(pts: np.ndarray, angle_deg: float) -> np.ndarray:
    theta = np.deg2rad(angle_deg)
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    center = pts.mean(axis=0, keepdims=True)
    return (pts - center) @ rot.T + center


def _inverse_rotate_points(
    pts: np.ndarray, angle_deg: float, center: np.ndarray
) -> np.ndarray:
    theta = np.deg2rad(-angle_deg)
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return (pts - center) @ rot.T + center


def _add_gaussian_noise(
    pts: np.ndarray, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    return pts + rng.normal(loc=0.0, scale=sigma, size=pts.shape)


def _downsample_points(
    pts: np.ndarray,
    keep_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    keep = max(8, int(round(keep_ratio * pts.shape[0])))
    if keep >= pts.shape[0]:
        return pts.copy()
    idx = rng.choice(pts.shape[0], size=keep, replace=False)
    return pts[idx]


def load_synthetic_pc_dataset(
    root: str = "data/Pointcloud/synthetic",
) -> List[PointCloudEntry]:
    """Load all synthetic point cloud shapes. Delegates to SyntheticPointCloudLoader."""
    return SyntheticPointCloudLoader(root).load_all()


def load_real_pc_dataset(
    root: str = "data/Pointcloud/real",
) -> List[PointCloudEntry]:
    """Load all real point cloud samples. Delegates to RealPointCloudLoader."""
    return RealPointCloudLoader(root).load_all()


def run_harris_pc_synthetic(
    params: HarrisPCParams | None = None,
    out_dir: str = "outputs/harris_pc",
) -> List[dict]:
    cfg = params or default_harris_pc_params()
    detector = HarrisPC(cfg)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    for name, pts in load_synthetic_pc_dataset():
        kps = detector.detect(pts)
        save_path = out / f"{name}_pc.png"
        plot_pointcloud(
            [pts],
            titles=[f"{name}: Harris PC ({kps.shape[0]} kps)"],
            keypoints_list=[kps],
            show=False,
            save_path=str(save_path),
        )
        rows.append(
            {
                "shape": name,
                "num_keypoints": int(kps.shape[0]),
                "plot_path": str(save_path),
            }
        )
    return rows


def run_harris_pc_real(
    params: HarrisPCParams | None = None,
    out_dir: str = "outputs/harris_pc",
) -> List[dict]:
    cfg = params or default_harris_pc_params()
    detector = HarrisPC(cfg)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    for name, pts in load_real_pc_dataset():
        kps = detector.detect(pts)
        save_path = out / f"{name}_pc.png"
        plot_pointcloud(
            [pts],
            titles=[f"{name}: Harris PC ({kps.shape[0]} kps)"],
            keypoints_list=[kps],
            show=False,
            save_path=str(save_path),
        )
        rows.append(
            {
                "shape": name,
                "num_keypoints": int(kps.shape[0]),
                "plot_path": str(save_path),
            }
        )
    return rows


def run_harris_pc_quantitative_evaluation(
    params: HarrisPCParams | None = None,
    out_dir: str = "outputs/evaluation/harris_pc",
    random_seed: int = 0,
    match_radius: float = 0.05,
    dataset_type: str = "synthetic",
    max_real_samples: int | None = None,
) -> dict:
    """Structured quantitative benchmark over synthetic and real point clouds.

    Metrics per perturbation:
    - repeatability under rotation, noise, and downsampling
    - keypoint count stability
    - localization error after inverse transform alignment

    Args:
        dataset_type: "synthetic", "real", or "both" (default: "synthetic")
    """
    cfg = params or default_harris_pc_params()
    detector = HarrisPC(cfg)
    rng = np.random.default_rng(random_seed)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    datasets = {}
    if dataset_type in ("synthetic", "both"):
        datasets["synthetic"] = load_synthetic_pc_dataset()
    if dataset_type in ("real", "both"):
        real_samples = load_real_pc_dataset()
        if max_real_samples is not None:
            real_samples = real_samples[:max_real_samples]
        datasets["real"] = real_samples

    perturb_specs = {
        "rotation": [10.0, 20.0, 30.0, 45.0],
        "noise": [0.002, 0.005, 0.010],
        "downsample": [0.9, 0.7, 0.5],
    }

    rows: list[dict] = []

    for split_name, samples in datasets.items():
        for sample_name, raw_pts in samples:
            pts = _normalize_points(raw_pts)
            baseline_kps = detector.detect(pts)
            baseline_count = int(baseline_kps.shape[0])
            center = pts.mean(axis=0, keepdims=True)

            for angle_deg in perturb_specs["rotation"]:
                pert = _rotate_points(pts, angle_deg)
                pert_kps = detector.detect(pert)
                pert_kps_back = _inverse_rotate_points(pert_kps, angle_deg, center)
                matching = match_keypoints(baseline_kps, pert_kps_back, match_radius)
                rows.append(
                    {
                        "dataset": split_name,
                        "sample": sample_name,
                        "perturbation": "rotation",
                        "level": float(angle_deg),
                        "baseline_count": baseline_count,
                        "perturbed_count": int(pert_kps.shape[0]),
                        "num_matches": int(matching.distances.size),
                        "repeatability": repeatability_score(
                            baseline_count,
                            int(pert_kps.shape[0]),
                            int(matching.distances.size),
                        ),
                        "localization_error": localization_error(matching.distances),
                    }
                )

            for sigma in perturb_specs["noise"]:
                pert = _add_gaussian_noise(pts, sigma, rng)
                pert_kps = detector.detect(pert)
                matching = match_keypoints(baseline_kps, pert_kps, match_radius)
                rows.append(
                    {
                        "dataset": split_name,
                        "sample": sample_name,
                        "perturbation": "noise",
                        "level": float(sigma),
                        "baseline_count": baseline_count,
                        "perturbed_count": int(pert_kps.shape[0]),
                        "num_matches": int(matching.distances.size),
                        "repeatability": repeatability_score(
                            baseline_count,
                            int(pert_kps.shape[0]),
                            int(matching.distances.size),
                        ),
                        "localization_error": localization_error(matching.distances),
                    }
                )

            for keep_ratio in perturb_specs["downsample"]:
                pert = _downsample_points(pts, keep_ratio, rng)
                pert_kps = detector.detect(pert)
                matching = match_keypoints(baseline_kps, pert_kps, match_radius)
                rows.append(
                    {
                        "dataset": split_name,
                        "sample": sample_name,
                        "perturbation": "downsample",
                        "level": float(keep_ratio),
                        "baseline_count": baseline_count,
                        "perturbed_count": int(pert_kps.shape[0]),
                        "num_matches": int(matching.distances.size),
                        "repeatability": repeatability_score(
                            baseline_count,
                            int(pert_kps.shape[0]),
                            int(matching.distances.size),
                        ),
                        "localization_error": localization_error(matching.distances),
                    }
                )

    summary: dict[str, dict] = {}
    for split_name in datasets:
        summary[split_name] = {}
        split_rows = [r for r in rows if r["dataset"] == split_name]
        for perturb in perturb_specs:
            perturb_rows = [r for r in split_rows if r["perturbation"] == perturb]
            if not perturb_rows:
                continue
            rep = np.asarray(
                [r["repeatability"] for r in perturb_rows], dtype=np.float64
            )
            loc = np.asarray(
                [
                    r["localization_error"]
                    for r in perturb_rows
                    if r["localization_error"] is not None
                ],
                dtype=np.float64,
            )
            counts = [int(r["perturbed_count"]) for r in perturb_rows]
            summary[split_name][perturb] = {
                "repeatability_mean": float(np.mean(rep)),
                "repeatability_std": float(np.std(rep)),
                "localization_error_mean": float(np.mean(loc))
                if loc.size > 0
                else None,
                "localization_error_std": float(np.std(loc)) if loc.size > 0 else None,
                "count_stability": keypoint_count_stability(counts),
                "num_trials": len(perturb_rows),
            }

    report = {
        "config": {
            "match_radius": match_radius,
            "random_seed": random_seed,
            "params": {
                "k": cfg.k,
                "k_neighbors": cfg.k_neighbors,
                "threshold_rel": cfg.threshold_rel,
                "nms_radius": cfg.nms_radius,
                "min_surface_variation": cfg.min_surface_variation,
            },
        },
        "summary": summary,
        "rows": rows,
    }

    with (out / "quantitative_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def run_sift_geom_pc_quantitative_evaluation(
    params: SIFTRadiiPCParams | None = None,
    out_dir: str = "outputs/evaluation/sift_geom_pc",
    random_seed: int = 0,
    match_radius: float = 0.05,
    dataset_type: str = "synthetic",
    max_real_samples: int | None = None,
) -> dict:
    """Structured quantitative benchmark for SIFT (radii-based) PC over synthetic and real data.

    Args:
        dataset_type: "synthetic", "real", or "both" (default: "synthetic")
        max_real_samples: optionally limit the number of real samples processed
    """
    cfg = params or SIFTRadiiPCParams()
    detector = SIFTRadiiPC(cfg)
    rng = np.random.default_rng(random_seed)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    datasets = {}
    if dataset_type in ("synthetic", "both"):
        datasets["synthetic"] = load_synthetic_pc_dataset()
    if dataset_type in ("real", "both"):
        real_samples = load_real_pc_dataset()
        if max_real_samples is not None:
            real_samples = real_samples[:max_real_samples]
        datasets["real"] = real_samples

    perturb_specs = {
        "rotation": [10.0, 20.0, 30.0, 45.0],
        "noise": [0.002, 0.005, 0.010],
        "downsample": [0.9, 0.7, 0.5],
    }

    rows: list[dict] = []

    for split_name, samples in datasets.items():
        for sample_name, raw_pts in samples:
            pts = _normalize_points(raw_pts)
            baseline_kps = detector.detect(pts)
            baseline_count = int(baseline_kps.shape[0])
            center = pts.mean(axis=0, keepdims=True)

            for angle_deg in perturb_specs["rotation"]:
                pert = _rotate_points(pts, angle_deg)
                pert_kps = detector.detect(pert)
                pert_kps_back = _inverse_rotate_points(pert_kps, angle_deg, center)
                matching = match_keypoints(baseline_kps, pert_kps_back, match_radius)
                rows.append(
                    {
                        "dataset": split_name,
                        "sample": sample_name,
                        "perturbation": "rotation",
                        "level": float(angle_deg),
                        "baseline_count": baseline_count,
                        "perturbed_count": int(pert_kps.shape[0]),
                        "num_matches": int(matching.distances.size),
                        "repeatability": repeatability_score(
                            baseline_count,
                            int(pert_kps.shape[0]),
                            int(matching.distances.size),
                        ),
                        "localization_error": localization_error(matching.distances),
                    }
                )

            for sigma in perturb_specs["noise"]:
                pert = _add_gaussian_noise(pts, sigma, rng)
                pert_kps = detector.detect(pert)
                matching = match_keypoints(baseline_kps, pert_kps, match_radius)
                rows.append(
                    {
                        "dataset": split_name,
                        "sample": sample_name,
                        "perturbation": "noise",
                        "level": float(sigma),
                        "baseline_count": baseline_count,
                        "perturbed_count": int(pert_kps.shape[0]),
                        "num_matches": int(matching.distances.size),
                        "repeatability": repeatability_score(
                            baseline_count,
                            int(pert_kps.shape[0]),
                            int(matching.distances.size),
                        ),
                        "localization_error": localization_error(matching.distances),
                    }
                )

            for keep_ratio in perturb_specs["downsample"]:
                pert = _downsample_points(pts, keep_ratio, rng)
                pert_kps = detector.detect(pert)
                matching = match_keypoints(baseline_kps, pert_kps, match_radius)
                rows.append(
                    {
                        "dataset": split_name,
                        "sample": sample_name,
                        "perturbation": "downsample",
                        "level": float(keep_ratio),
                        "baseline_count": baseline_count,
                        "perturbed_count": int(pert_kps.shape[0]),
                        "num_matches": int(matching.distances.size),
                        "repeatability": repeatability_score(
                            baseline_count,
                            int(pert_kps.shape[0]),
                            int(matching.distances.size),
                        ),
                        "localization_error": localization_error(matching.distances),
                    }
                )

    summary: dict[str, dict] = {}
    for split_name in datasets:
        summary[split_name] = {}
        split_rows = [r for r in rows if r["dataset"] == split_name]
        for perturb in perturb_specs:
            perturb_rows = [r for r in split_rows if r["perturbation"] == perturb]
            if not perturb_rows:
                continue
            rep = np.asarray(
                [r["repeatability"] for r in perturb_rows], dtype=np.float64
            )
            loc = np.asarray(
                [
                    r["localization_error"]
                    for r in perturb_rows
                    if r["localization_error"] is not None
                ],
                dtype=np.float64,
            )
            counts = [int(r["perturbed_count"]) for r in perturb_rows]
            summary[split_name][perturb] = {
                "repeatability_mean": float(np.mean(rep)),
                "repeatability_std": float(np.std(rep)),
                "localization_error_mean": float(np.mean(loc))
                if loc.size > 0
                else None,
                "localization_error_std": float(np.std(loc)) if loc.size > 0 else None,
                "count_stability": keypoint_count_stability(counts),
                "num_trials": len(perturb_rows),
            }

    report = {
        "config": {
            "match_radius": match_radius,
            "random_seed": random_seed,
            "params": {
                "num_octaves": cfg.num_octaves,
                "radii": list(cfg.radii),
                "fps_ratio": cfg.fps_ratio,
                "contrast_threshold": cfg.contrast_threshold,
                "nms_radius_factor": cfg.nms_radius_factor,
            },
        },
        "summary": summary,
        "rows": rows,
    }

    with (out / "quantitative_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    print("=== Synthetic point clouds ===")
    for row in run_harris_pc_synthetic():
        print(f"  {row['shape']:>15} | keypoints={row['num_keypoints']:>4}")

    print("\n=== Real point clouds ===")
    for row in run_harris_pc_real():
        print(f"  {row['shape']:>15} | keypoints={row['num_keypoints']:>4}")

    print("\n=== Quantitative evaluation (Harris PC, synthetic + real) ===")
    harris_report = run_harris_pc_quantitative_evaluation()
    for split_name, split_summary in harris_report["summary"].items():
        print(f"\n[harris_pc | {split_name}]")
        for perturb, stats in split_summary.items():
            print(
                "  "
                f"{perturb:>10} | "
                f"repeatability={stats['repeatability_mean']:.3f}+-{stats['repeatability_std']:.3f} | "
                f"loc_err={stats['localization_error_mean']} | "
                f"count_cv={stats['count_stability']['cv']:.3f}"
            )

    print("\n=== Quantitative evaluation (SIFT-Geom PC, synthetic + real) ===")
    sift_report = run_sift_geom_pc_quantitative_evaluation()
    for split_name, split_summary in sift_report["summary"].items():
        print(f"\n[sift_geom_pc | {split_name}]")
        for perturb, stats in split_summary.items():
            print(
                "  "
                f"{perturb:>10} | "
                f"repeatability={stats['repeatability_mean']:.3f}+-{stats['repeatability_std']:.3f} | "
                f"loc_err={stats['localization_error_mean']} | "
                f"count_cv={stats['count_stability']['cv']:.3f}"
            )

    # Build aggregate tables/plots combining all available reports.
    try:
        from src.evaluation.build_quantitative_tables_plots import (
            build_tables_and_plots,
        )

        reports = {
            "harris_pc": Path("outputs/evaluation/harris_pc/quantitative_report.json"),
            "sift_geom_pc": Path(
                "outputs/evaluation/sift_geom_pc/quantitative_report.json"
            ),
            "harris3d_voxel": Path(
                "outputs/evaluation/harris3d/quantitative_report.json"
            ),
            "sift3d_voxel": Path("outputs/evaluation/sift3d/quantitative_report.json"),
        }
        summary_out = Path("outputs/evaluation/summary")
        build_tables_and_plots(reports, summary_out)
        print(f"Wrote summary tables/plots to: {summary_out}")
    except Exception:
        pass
