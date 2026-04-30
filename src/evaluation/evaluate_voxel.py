from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, cast

import numpy as np
from scipy.ndimage import rotate, zoom

from src.common.io import ModelNetLoader, SyntheticVoxelLoader
from src.common.metrics import (
    keypoint_count_stability,
    localization_error,
    match_keypoints,
    repeatability_score,
)
from src.common.visualization import plot_voxels
from src.voxel.harris3d import Harris3DVoxel
from src.voxel.params import Harris3DParams, SIFT3DParams, default_harris3d_params
from src.voxel.sift3d import SIFT3DVoxel

VolumeEntry = Tuple[str, np.ndarray]


def _rotate_volume_z(volume: np.ndarray, angle_deg: float) -> np.ndarray:
    rotated = rotate(
        volume.astype(np.float64),
        angle=angle_deg,
        axes=(1, 2),
        reshape=False,
        order=0,
        mode="nearest",
    )
    return rotated > 0.5


def _inverse_rotate_points_voxel(
    points_xyz: np.ndarray,
    angle_deg: float,
    shape_xyz: np.ndarray,
) -> np.ndarray:
    center = (shape_xyz - 1.0) / 2.0
    theta = np.deg2rad(-angle_deg)
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return (points_xyz - center) @ rot.T + center


def _flip_noise(
    volume: np.ndarray, flip_prob: float, rng: np.random.Generator
) -> np.ndarray:
    mask = rng.random(volume.shape) < flip_prob
    return np.logical_xor(volume, mask)


def _downsample_then_upsample(volume: np.ndarray, factor: float) -> np.ndarray:
    if factor >= 0.999:
        return volume.copy()
    down = zoom(volume.astype(np.float64), zoom=factor, order=0)
    up_factors = np.asarray(volume.shape, dtype=np.float64) / np.asarray(
        down.shape, dtype=np.float64
    )
    restored = zoom(down, zoom=tuple(up_factors), order=0)
    if restored.shape != volume.shape:
        # Crop/pad to original shape to keep detector interface fixed.
        out = np.zeros(volume.shape, dtype=np.float64)
        out_z, out_y, out_x = cast(Tuple[int, int, int], out.shape)
        if len(restored.shape) >= 3:
            rest_z, rest_y, rest_x = cast(
                Tuple[int, int, int], tuple(restored.shape[:3])
            )
        else:
            rest_z = rest_y = rest_x = 0
        z = min(out_z, rest_z)
        y = min(out_y, rest_y)
        x = min(out_x, rest_x)
        out[:z, :y, :x] = restored[:z, :y, :x]
        restored = out
    return np.asarray(restored > 0.5)


def load_synthetic_voxel_dataset(
    root: str = "data/Voxel/synthetic",
) -> List[VolumeEntry]:
    """Load all synthetic voxel shapes. Delegates to SyntheticVoxelLoader."""
    return SyntheticVoxelLoader(root).load_all()


def run_harris3d_synthetic(
    params: Harris3DParams | None = None,
    out_dir: str = "outputs/harris3d",
) -> List[dict]:
    cfg = params or default_harris3d_params()
    detector = Harris3DVoxel(cfg)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []

    for name, vol in load_synthetic_voxel_dataset():
        kps = detector.detect(vol)
        volume_path = out / f"{name}_volume.png"
        plot_voxels(
            [vol],
            titles=[f"{name}: 3D Volume + Keypoints"],
            keypoints_list=[kps],
            show=False,
            save_path=str(volume_path),
        )

        rows.append(
            {
                "shape": name,
                "num_keypoints": int(kps.shape[0]),
                "volume_plot": str(volume_path),
            }
        )
    return rows


def run_harris3d_real_chair(
    params: Harris3DParams | None = None,
    out_dir: str = "outputs/harris3d",
) -> dict:
    cfg = params or default_harris3d_params()
    detector = Harris3DVoxel(cfg)

    modelnet_loader = ModelNetLoader(
        "data/Voxel/real/ModelNet10-dataset/modelnet10.npy.gz"
    )
    # Find first non-empty chair sample
    idx = -1
    chair = None
    for idx, chair in modelnet_loader.load_sequential():
        if chair.any():
            break

    keypoints = detector.detect(chair)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    volume_path = out / "chair_volume.png"

    plot_voxels(
        [chair],
        titles=[f"ModelNet10 Chair sample {idx}: 3D Volume + Keypoints"],
        keypoints_list=[keypoints],
        show=False,
        save_path=str(volume_path),
    )

    report = {
        "shape": "chair",
        "sample_index": int(idx),
        "num_keypoints": int(keypoints.shape[0]),
        "volume_plot": str(volume_path),
    }
    return report


def run_harris3d_random_modelnet_sample(
    params: Harris3DParams | None = None,
    sample_index: int | None = None,
) -> dict:
    cfg = params or default_harris3d_params()
    detector = Harris3DVoxel(cfg)

    modelnet_loader = ModelNetLoader(
        "data/Voxel/real/ModelNet10-dataset/modelnet10.npy.gz"
    )

    if sample_index is not None:
        idx, sample = sample_index, modelnet_loader.load_by_index(sample_index)
    else:
        idx, sample = modelnet_loader.load_random()

    keypoints = detector.detect(sample)

    return {
        "shape": "modelnet_random",
        "sample_index": int(idx),
        "num_keypoints": int(keypoints.shape[0]),
        "sample": sample,
        "keypoints": keypoints,
    }


def run_harris3d_quantitative_evaluation(
    params: Harris3DParams | None = None,
    out_dir: str = "outputs/evaluation/harris3d",
    random_seed: int = 0,
    match_radius: float = 2.0,
    dataset_type: str = "synthetic",
    max_real_samples: int | None = None,
) -> dict:
    """Structured robustness benchmark over synthetic + real voxel datasets.

    Args:
        dataset_type: "synthetic", "real", or "both" (default: "synthetic")
    """
    cfg = params or default_harris3d_params()
    detector = Harris3DVoxel(cfg)
    rng = np.random.default_rng(random_seed)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    datasets: dict[str, list[VolumeEntry]] = {}

    if dataset_type in ("synthetic", "both"):
        datasets["synthetic"] = load_synthetic_voxel_dataset()

    if dataset_type in ("real", "both"):
        modelnet_loader = ModelNetLoader(
            "data/Voxel/real/ModelNet10-dataset/modelnet10.npy.gz"
        )
        real_samples: list[VolumeEntry] = []
        # allow caller to override how many real samples to collect (default: 8)
        max_to_collect = max_real_samples if max_real_samples is not None else 8
        for idx, sample in modelnet_loader.load_sequential():
            if sample.any():
                real_samples.append((f"modelnet_{idx}", sample))
            if len(real_samples) >= max_to_collect:
                break
        datasets["real"] = real_samples

    perturb_specs = {
        "rotation": [10.0, 20.0, 30.0],
        "noise": [0.005, 0.010, 0.020],
        "downsample": [0.85, 0.70, 0.50],
    }

    rows: list[dict] = []

    for split_name, samples in datasets.items():
        for sample_name, volume in samples:
            baseline_kps = detector.detect(volume).astype(np.float64)
            baseline_count = int(baseline_kps.shape[0])
            shape_xyz = np.asarray(
                [volume.shape[2], volume.shape[1], volume.shape[0]], dtype=np.float64
            )

            for angle_deg in perturb_specs["rotation"]:
                pert = _rotate_volume_z(volume, angle_deg)
                pert_kps = detector.detect(pert).astype(np.float64)
                pert_kps_back = _inverse_rotate_points_voxel(
                    pert_kps, angle_deg, shape_xyz
                )
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

            for flip_prob in perturb_specs["noise"]:
                pert = _flip_noise(volume, flip_prob, rng)
                pert_kps = detector.detect(pert).astype(np.float64)
                matching = match_keypoints(baseline_kps, pert_kps, match_radius)
                rows.append(
                    {
                        "dataset": split_name,
                        "sample": sample_name,
                        "perturbation": "noise",
                        "level": float(flip_prob),
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

            for factor in perturb_specs["downsample"]:
                pert = _downsample_then_upsample(volume, factor)
                pert_kps = detector.detect(pert).astype(np.float64)
                matching = match_keypoints(baseline_kps, pert_kps, match_radius)
                rows.append(
                    {
                        "dataset": split_name,
                        "sample": sample_name,
                        "perturbation": "downsample",
                        "level": float(factor),
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
                "gradient_sigma": cfg.gradient_sigma,
                "tensor_sigma": cfg.tensor_sigma,
                "threshold_rel": cfg.threshold_rel,
                "nms_window": cfg.nms_window,
            },
        },
        "summary": summary,
        "rows": rows,
    }

    with (out / "quantitative_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def run_sift3d_quantitative_evaluation(
    params: SIFT3DParams | None = None,
    out_dir: str = "outputs/evaluation/sift3d",
    random_seed: int = 0,
    match_radius: float = 2.0,
    dataset_type: str = "synthetic",
    max_real_samples: int | None = None,
) -> dict:
    """Structured robustness benchmark for SIFT3D over synthetic + real voxel datasets.

    Args:
        dataset_type: "synthetic", "real", or "both" (default: "synthetic")
    """
    cfg = params or SIFT3DParams()
    detector = SIFT3DVoxel(cfg)
    rng = np.random.default_rng(random_seed)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    datasets: dict[str, list[VolumeEntry]] = {}

    if dataset_type in ("synthetic", "both"):
        datasets["synthetic"] = load_synthetic_voxel_dataset()

    if dataset_type in ("real", "both"):
        modelnet_loader = ModelNetLoader(
            "data/Voxel/real/ModelNet10-dataset/modelnet10.npy.gz"
        )
        real_samples: list[VolumeEntry] = []
        max_to_collect = max_real_samples if max_real_samples is not None else 8
        for idx, sample in modelnet_loader.load_sequential():
            if sample.any():
                real_samples.append((f"modelnet_{idx}", sample))
            if len(real_samples) >= max_to_collect:
                break
        datasets["real"] = real_samples

    perturb_specs = {
        "rotation": [10.0, 20.0, 30.0],
        "noise": [0.005, 0.010, 0.020],
        "downsample": [0.85, 0.70, 0.50],
    }

    rows: list[dict] = []

    for split_name, samples in datasets.items():
        for sample_name, volume in samples:
            baseline_kps = detector.detect(volume).astype(np.float64)
            baseline_count = int(baseline_kps.shape[0])
            shape_xyz = np.asarray(
                [volume.shape[2], volume.shape[1], volume.shape[0]], dtype=np.float64
            )

            for angle_deg in perturb_specs["rotation"]:
                pert = _rotate_volume_z(volume, angle_deg)
                pert_kps = detector.detect(pert).astype(np.float64)
                pert_kps_back = _inverse_rotate_points_voxel(
                    pert_kps, angle_deg, shape_xyz
                )
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

            for flip_prob in perturb_specs["noise"]:
                pert = _flip_noise(volume, flip_prob, rng)
                pert_kps = detector.detect(pert).astype(np.float64)
                matching = match_keypoints(baseline_kps, pert_kps, match_radius)
                rows.append(
                    {
                        "dataset": split_name,
                        "sample": sample_name,
                        "perturbation": "noise",
                        "level": float(flip_prob),
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

            for factor in perturb_specs["downsample"]:
                pert = _downsample_then_upsample(volume, factor)
                pert_kps = detector.detect(pert).astype(np.float64)
                matching = match_keypoints(baseline_kps, pert_kps, match_radius)
                rows.append(
                    {
                        "dataset": split_name,
                        "sample": sample_name,
                        "perturbation": "downsample",
                        "level": float(factor),
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
                "scales_per_octave": cfg.scales_per_octave,
                "base_sigma": cfg.base_sigma,
                "extrema_contrast_threshold": cfg.extrema_contrast_threshold,
            },
        },
        "summary": summary,
        "rows": rows,
    }

    with (out / "quantitative_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    rows = run_harris3d_synthetic()
    for row in rows:
        print(f"{row['shape']:>8} | keypoints={row['num_keypoints']:>4}")

    chair = run_harris3d_real_chair()
    print(f"{chair['shape']:>8} | keypoints={chair['num_keypoints']:>4}")

    print("\n=== Quantitative evaluation (Harris3D, synthetic + real) ===")
    harris_report = run_harris3d_quantitative_evaluation()
    for split_name, split_summary in harris_report["summary"].items():
        print(f"\n[harris3d | {split_name}]")
        for perturb, stats in split_summary.items():
            print(
                "  "
                f"{perturb:>10} | "
                f"repeatability={stats['repeatability_mean']:.3f}+-{stats['repeatability_std']:.3f} | "
                f"loc_err={stats['localization_error_mean']} | "
                f"count_cv={stats['count_stability']['cv']:.3f}"
            )

    print("\n=== Quantitative evaluation (SIFT3D, synthetic + real) ===")
    sift_report = run_sift3d_quantitative_evaluation()
    for split_name, split_summary in sift_report["summary"].items():
        print(f"\n[sift3d | {split_name}]")
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
