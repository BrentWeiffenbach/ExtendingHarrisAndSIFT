#!/usr/bin/env python
"""
Enhanced Evaluation Pipeline: Synthetic-first + Single Real Sample.
- All synthetic evaluations run first (fast iteration)
- Parameter sensitivity analysis follows
- Single representative real sample runs at the very end

IMPROVEMENTS:
  - Run all synthetic data first (quick feedback loop)
  - Add incremental progress logging after each sample
  - Run a single real sample at the end instead of all real data
  - Estimate time savings: ~80% faster for initial results
"""

import argparse
import csv
import json
import logging
import time
from pathlib import Path
import tempfile
import shutil

import numpy as np

# Import tuning logic
from parameter_tuning import (
    ALL_SHAPES,
    DETECTOR_REGISTRY,
    aggregate_to_dataframe,
    load_pc_data,
    load_voxel_data,
    plot_detector,
    run_detector_sweep,
)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def run_detector_on_dataset(
    func, detector_name, out_dir, radius, dataset_type="synthetic"
):
    """Wrapper to run a detector evaluation with progress tracking."""
    start = time.time()
    print(f"  ⏱  Starting {detector_name} ({dataset_type})...")
    try:
        result = func(
            out_dir=str(out_dir),
            random_seed=0,
            match_radius=radius,
            dataset_type=dataset_type,
        )
        elapsed = time.time() - start
        print(f"  ✓ {detector_name} ({dataset_type}) completed in {elapsed:.1f}s")
        return result
    except TypeError:
        # Fallback if dataset_type parameter not supported
        print(f"  ⏱  Starting {detector_name} (legacy mode)...")
        result = func(out_dir=str(out_dir), random_seed=0, match_radius=radius)
        elapsed = time.time() - start
        print(f"  ✓ {detector_name} completed in {elapsed:.1f}s")
        return result
    except Exception as e:
        print(f"  ✗ {detector_name} FAILED: {e}")
        raise


def run_single_real_sample_evaluation(out_root: Path):
    """Run a single representative real sample for all detectors at the very end."""
    print("\n" + "=" * 80)
    print("STAGE 5: REAL DATA EVALUATION (Single Representative Sample)")
    print("=" * 80)

    try:
        from src.evaluation.evaluate_pc import load_real_pc_dataset, _normalize_points
        from src.evaluation.evaluate_voxel import ModelNetLoader
        from src.pointcloud.harris_pc import HarrisPC
        from src.pointcloud.params import default_harris_pc_params
        from src.pointcloud.sift_pc import SIFTRadiiPC
        from src.pointcloud.params import SIFTRadiiPCParams
        from src.voxel.harris3d import Harris3DVoxel
        from src.voxel.params import default_harris3d_params
        from src.voxel.sift3d import SIFT3DVoxel
        from src.voxel.params import SIFT3DParams
        from src.common.visualization import plot_pointcloud, plot_voxels
        import numpy as np
    except ImportError as e:
        logging.error(f"Failed to import for real sample evaluation: {e}")
        return {}

    real_reports = {}

    # --- Point Cloud: Single Real Sample ---
    print("\n[PC] Running on single real point cloud...")
    try:
        real_pc_data = load_real_pc_dataset()
        if real_pc_data:
            sample_name, raw_pts = real_pc_data[0]  # Take first real sample
            pts = _normalize_points(raw_pts)
            print(f"  Using: {sample_name} ({pts.shape[0]} points)")

            pc_out = out_root / "real_sample_pc"
            pc_out.mkdir(parents=True, exist_ok=True)

            # Harris PC
            print("  → Harris PC...")
            harris_detector = HarrisPC(default_harris_pc_params())
            harris_kps = harris_detector.detect(pts)
            plot_pointcloud(
                [pts],
                titles=[f"{sample_name}: Harris PC ({harris_kps.shape[0]} kps)"],
                keypoints_list=[harris_kps],
                show=False,
                save_path=str(pc_out / "harris_real_sample.png"),
            )
            print(f"    ✓ Harris PC: {harris_kps.shape[0]} keypoints")

            # SIFT PC
            print("  → SIFT-Radii PC...")
            sift_detector = SIFTRadiiPC(SIFTRadiiPCParams())
            sift_kps = sift_detector.detect(pts)
            plot_pointcloud(
                [pts],
                titles=[f"{sample_name}: SIFT-Radii PC ({sift_kps.shape[0]} kps)"],
                keypoints_list=[sift_kps],
                show=False,
                save_path=str(pc_out / "sift_real_sample.png"),
            )
            print(f"    ✓ SIFT-Radii PC: {sift_kps.shape[0]} keypoints")

            real_reports["point_cloud"] = {
                "sample": sample_name,
                "harris_keypoints": int(harris_kps.shape[0]),
                "sift_keypoints": int(sift_kps.shape[0]),
                "plot_dir": str(pc_out),
            }
    except Exception as e:
        logging.error(f"Point cloud real sample failed: {e}")

    # --- Voxel: Single Real Sample (ModelNet10) ---
    print("\n[Voxel] Running on single ModelNet10 sample...")
    try:
        modelnet_loader = ModelNetLoader(
            "data/Voxel/real/ModelNet10-dataset/modelnet10.npy.gz"
        )

        # Find first non-empty chair
        idx = -1
        chair_volume = None
        for idx, sample in modelnet_loader.load_sequential():
            if sample.any():
                chair_volume = sample
                break

        if chair_volume is None:
            raise ValueError("No valid ModelNet samples found")

        print(f"  Using: ModelNet10 chair (index {idx}, shape {chair_volume.shape})")
        voxel_out = out_root / "real_sample_voxel"
        voxel_out.mkdir(parents=True, exist_ok=True)

        # Harris3D
        print("  → Harris3D...")
        harris3d_detector = Harris3DVoxel(default_harris3d_params())
        harris3d_kps = harris3d_detector.detect(chair_volume)
        plot_voxels(
            [chair_volume],
            titles=[f"ModelNet10 Chair: Harris3D ({harris3d_kps.shape[0]} kps)"],
            keypoints_list=[harris3d_kps],
            show=False,
            save_path=str(voxel_out / "harris3d_real_sample.png"),
        )
        print(f"    ✓ Harris3D: {harris3d_kps.shape[0]} keypoints")

        # SIFT3D
        print("  → SIFT3D...")
        sift3d_detector = SIFT3DVoxel(SIFT3DParams())
        sift3d_kps = sift3d_detector.detect(chair_volume)
        plot_voxels(
            [chair_volume],
            titles=[f"ModelNet10 Chair: SIFT3D ({sift3d_kps.shape[0]} kps)"],
            keypoints_list=[sift3d_kps],
            show=False,
            save_path=str(voxel_out / "sift3d_real_sample.png"),
        )
        print(f"    ✓ SIFT3D: {sift3d_kps.shape[0]} keypoints")

        real_reports["voxel"] = {
            "harris3d_keypoints": int(harris3d_kps.shape[0]),
            "sift3d_keypoints": int(sift3d_kps.shape[0]),
            "sample_index": int(idx),
            "sample_shape": list(chair_volume.shape),
            "plot_dir": str(voxel_out),
        }
    except Exception as e:
        logging.error(f"Voxel real sample failed: {e}")

    return real_reports


def run_all_perturbations_output(out_root: Path) -> None:
    """Generate per-sample plots and CSVs for rotation, noise, and downsample perturbations."""
    try:
        from src.evaluation.evaluate_pc import (
            load_real_pc_dataset,
            load_synthetic_pc_dataset,
            _normalize_points,
            _rotate_points,
            _inverse_rotate_points,
            _add_gaussian_noise,
            _downsample_points,
        )
        from src.evaluation.evaluate_voxel import (
            load_synthetic_voxel_dataset,
            _rotate_volume_z,
            _inverse_rotate_points_voxel,
            _flip_noise,
            _downsample_then_upsample,
            ModelNetLoader,
        )
        from src.common.metrics import (
            match_keypoints,
            repeatability_score,
            localization_error,
        )
        from src.common.visualization import plot_pointcloud, plot_voxels
        from src.pointcloud.harris_pc import HarrisPC
        from src.pointcloud.params import default_harris_pc_params, SIFTRadiiPCParams
        from src.pointcloud.sift_pc import SIFTRadiiPC
        from src.voxel.harris3d import Harris3DVoxel
        from src.voxel.params import default_harris3d_params, SIFT3DParams
        from src.voxel.sift3d import SIFT3DVoxel
    except ImportError as e:
        logging.error(f"Failed to import modules for perturbation output: {e}")
        return

    out_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    PC_ANGLES = [10.0, 20.0, 30.0, 45.0]
    PC_NOISE_SIGMAS = [0.002, 0.005, 0.010]
    PC_DOWNSAMPLE_RATIOS = [0.9, 0.7, 0.5]
    PC_MATCH_RADIUS = 0.2

    VOXEL_ANGLES = [10.0, 20.0, 30.0]
    VOXEL_NOISE_PROBS = [0.005, 0.010, 0.020]
    VOXEL_DOWNSAMPLE_FACTORS = [0.85, 0.70, 0.50]
    VOXEL_MATCH_RADIUS = 2.0

    pc_detectors = [
        ("harris_pc", HarrisPC(default_harris_pc_params())),
        ("sift_geom_pc", SIFTRadiiPC(SIFTRadiiPCParams())),
    ]
    voxel_detectors = [
        ("harris3d", Harris3DVoxel(default_harris3d_params())),
        ("sift3d", SIFT3DVoxel(SIFT3DParams())),
    ]

    # Load datasets once
    synthetic_pc = load_synthetic_pc_dataset()
    real_pc = load_real_pc_dataset()
    synthetic_voxel = load_synthetic_voxel_dataset()

    real_voxel_sample = None
    try:
        loader = ModelNetLoader("data/Voxel/real/ModelNet10-dataset/modelnet10.npy.gz")
        for idx, sample in loader.load_sequential():
            if sample.any():
                real_voxel_sample = ("modelnet_chair", sample)
                break
    except Exception as e:
        logging.warning(f"Could not load ModelNet real voxel sample: {e}")

    all_records: list[dict] = []

    def _append_record(
        det_name,
        split_name,
        sample_name,
        perturbation,
        level,
        baseline_count,
        pert_kps,
        matching,
        records_list,
    ):
        rep = repeatability_score(
            baseline_count, int(pert_kps.shape[0]), int(matching.distances.size)
        )
        loc_err = localization_error(matching.distances)
        row = {
            "detector": det_name,
            "dataset_type": split_name,
            "sample": sample_name,
            "perturbation": perturbation,
            "level": level,
            "baseline_count": baseline_count,
            "perturbed_count": int(pert_kps.shape[0]),
            "num_matches": int(matching.distances.size),
            "repeatability": round(rep, 6),
            "localization_error": round(loc_err, 8) if loc_err is not None else None,
        }
        records_list.append(row)
        all_records.append(row)

    # --- Point Cloud Detectors ---
    for det_name, detector in pc_detectors:
        det_dir = out_root / det_name
        det_records: list[dict] = []

        samples_by_split = [
            ("synthetic", [(n, _normalize_points(p)) for n, p in synthetic_pc]),
            ("real", [(n, _normalize_points(p)) for n, p in real_pc]),
        ]

        for split_name, samples in samples_by_split:
            for sample_name, pts_norm in samples:
                baseline_kps = detector.detect(pts_norm)
                baseline_count = int(baseline_kps.shape[0])
                center = pts_norm.mean(axis=0, keepdims=True)

                sample_dir = det_dir / sample_name
                sample_dir.mkdir(parents=True, exist_ok=True)

                # Rotation
                for angle in PC_ANGLES:
                    pert = _rotate_points(pts_norm, angle)
                    pert_kps = detector.detect(pert)
                    pert_kps_back = _inverse_rotate_points(pert_kps, angle, center)
                    matching = match_keypoints(
                        baseline_kps, pert_kps_back, PC_MATCH_RADIUS
                    )
                    try:
                        plot_pointcloud(
                            [pts_norm, pert],
                            titles=[
                                f"{sample_name} baseline ({baseline_count} kps)",
                                f"Rotated {int(angle)}° ({pert_kps.shape[0]} kps)",
                            ],
                            keypoints_list=[baseline_kps, pert_kps],
                            show=False,
                            save_path=str(sample_dir / f"rotation_{int(angle)}deg.png"),
                        )
                    except Exception as e:
                        logging.warning(
                            f"Plot failed {det_name}/{sample_name} rotation {angle}°: {e}"
                        )
                    _append_record(
                        det_name,
                        split_name,
                        sample_name,
                        "rotation",
                        float(angle),
                        baseline_count,
                        pert_kps,
                        matching,
                        det_records,
                    )

                # Noise
                for sigma in PC_NOISE_SIGMAS:
                    pert = _add_gaussian_noise(pts_norm, sigma, rng)
                    pert_kps = detector.detect(pert)
                    matching = match_keypoints(baseline_kps, pert_kps, PC_MATCH_RADIUS)
                    try:
                        plot_pointcloud(
                            [pts_norm, pert],
                            titles=[
                                f"{sample_name} baseline ({baseline_count} kps)",
                                f"Noise σ={sigma} ({pert_kps.shape[0]} kps)",
                            ],
                            keypoints_list=[baseline_kps, pert_kps],
                            show=False,
                            save_path=str(sample_dir / f"noise_sigma{sigma}.png"),
                        )
                    except Exception as e:
                        logging.warning(
                            f"Plot failed {det_name}/{sample_name} noise σ={sigma}: {e}"
                        )
                    _append_record(
                        det_name,
                        split_name,
                        sample_name,
                        "noise",
                        sigma,
                        baseline_count,
                        pert_kps,
                        matching,
                        det_records,
                    )

                # Downsample
                for ratio in PC_DOWNSAMPLE_RATIOS:
                    pert = _downsample_points(pts_norm, ratio, rng)
                    pert_kps = detector.detect(pert)
                    matching = match_keypoints(baseline_kps, pert_kps, PC_MATCH_RADIUS)
                    try:
                        plot_pointcloud(
                            [pts_norm, pert],
                            titles=[
                                f"{sample_name} baseline ({baseline_count} kps)",
                                f"Downsample {int(ratio * 100)}% ({pert_kps.shape[0]} kps)",
                            ],
                            keypoints_list=[baseline_kps, pert_kps],
                            show=False,
                            save_path=str(sample_dir / f"downsample_{ratio}.png"),
                        )
                    except Exception as e:
                        logging.warning(
                            f"Plot failed {det_name}/{sample_name} downsample {ratio}: {e}"
                        )
                    _append_record(
                        det_name,
                        split_name,
                        sample_name,
                        "downsample",
                        ratio,
                        baseline_count,
                        pert_kps,
                        matching,
                        det_records,
                    )

        _write_csv(det_dir / "perturbation_metrics.csv", det_records)
        print(f"  ✓ {det_name}: {len(det_records)} rows, plots in {det_dir}")

    # --- Voxel Detectors ---
    voxel_samples_by_split: list[tuple[str, list]] = [("synthetic", synthetic_voxel)]
    if real_voxel_sample is not None:
        voxel_samples_by_split.append(("real", [real_voxel_sample]))

    for det_name, detector in voxel_detectors:
        det_dir = out_root / det_name
        det_records = []

        for split_name, samples in voxel_samples_by_split:
            for sample_name, volume in samples:
                baseline_kps = detector.detect(volume).astype(np.float64)
                baseline_count = int(baseline_kps.shape[0])
                shape_xyz = np.asarray(
                    [volume.shape[2], volume.shape[1], volume.shape[0]],
                    dtype=np.float64,
                )

                sample_dir = det_dir / sample_name
                sample_dir.mkdir(parents=True, exist_ok=True)

                # Rotation
                for angle in VOXEL_ANGLES:
                    pert = _rotate_volume_z(volume, angle)
                    pert_kps = detector.detect(pert).astype(np.float64)
                    pert_kps_back = _inverse_rotate_points_voxel(
                        pert_kps, angle, shape_xyz
                    )
                    matching = match_keypoints(
                        baseline_kps, pert_kps_back, VOXEL_MATCH_RADIUS
                    )
                    try:
                        plot_voxels(
                            [volume, pert],
                            titles=[
                                f"{sample_name} baseline ({baseline_count} kps)",
                                f"Rotated {int(angle)}° ({pert_kps.shape[0]} kps)",
                            ],
                            keypoints_list=[baseline_kps, pert_kps],
                            show=False,
                            save_path=str(sample_dir / f"rotation_{int(angle)}deg.png"),
                        )
                    except Exception as e:
                        logging.warning(
                            f"Plot failed {det_name}/{sample_name} rotation {angle}°: {e}"
                        )
                    _append_record(
                        det_name,
                        split_name,
                        sample_name,
                        "rotation",
                        float(angle),
                        baseline_count,
                        pert_kps,
                        matching,
                        det_records,
                    )

                # Noise (voxel flip noise)
                for flip_prob in VOXEL_NOISE_PROBS:
                    pert = _flip_noise(volume, flip_prob, rng)
                    pert_kps = detector.detect(pert).astype(np.float64)
                    matching = match_keypoints(
                        baseline_kps, pert_kps, VOXEL_MATCH_RADIUS
                    )
                    try:
                        plot_voxels(
                            [volume, pert],
                            titles=[
                                f"{sample_name} baseline ({baseline_count} kps)",
                                f"Flip noise p={flip_prob} ({pert_kps.shape[0]} kps)",
                            ],
                            keypoints_list=[baseline_kps, pert_kps],
                            show=False,
                            save_path=str(sample_dir / f"noise_flip{flip_prob}.png"),
                        )
                    except Exception as e:
                        logging.warning(
                            f"Plot failed {det_name}/{sample_name} noise p={flip_prob}: {e}"
                        )
                    _append_record(
                        det_name,
                        split_name,
                        sample_name,
                        "noise",
                        flip_prob,
                        baseline_count,
                        pert_kps,
                        matching,
                        det_records,
                    )

                # Downsample
                for factor in VOXEL_DOWNSAMPLE_FACTORS:
                    pert = _downsample_then_upsample(volume, factor)
                    pert_kps = detector.detect(pert).astype(np.float64)
                    matching = match_keypoints(
                        baseline_kps, pert_kps, VOXEL_MATCH_RADIUS
                    )
                    try:
                        plot_voxels(
                            [volume, pert],
                            titles=[
                                f"{sample_name} baseline ({baseline_count} kps)",
                                f"Downsample ×{factor} ({pert_kps.shape[0]} kps)",
                            ],
                            keypoints_list=[baseline_kps, pert_kps],
                            show=False,
                            save_path=str(sample_dir / f"downsample_{factor}.png"),
                        )
                    except Exception as e:
                        logging.warning(
                            f"Plot failed {det_name}/{sample_name} downsample {factor}: {e}"
                        )
                    _append_record(
                        det_name,
                        split_name,
                        sample_name,
                        "downsample",
                        factor,
                        baseline_count,
                        pert_kps,
                        matching,
                        det_records,
                    )

        _write_csv(det_dir / "perturbation_metrics.csv", det_records)
        print(f"  ✓ {det_name}: {len(det_records)} rows, plots in {det_dir}")

    # Master CSV across all detectors
    _write_csv(out_root / "perturbation_all_samples.csv", all_records)
    print(
        f"\n  Master CSV: {out_root / 'perturbation_all_samples.csv'} ({len(all_records)} total rows)"
    )


def _write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(
        description="Optimized evaluation pipeline: synthetic-first + single real sample"
    )
    parser.add_argument(
        "--skip-tuning", action="store_true", help="Skip parameter sensitivity stage"
    )
    parser.add_argument(
        "--skip-real", action="store_true", help="Skip real data evaluation"
    )
    parser.add_argument(
        "--skip-perturbation-output",
        "--skip-rotation-output",
        dest="skip_rotation_output",
        action="store_true",
        help="Skip per-sample perturbation output folder (rotation/noise/downsample)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip all evaluations and rebuild summary from existing quantitative_report.json files",
    )
    parser.add_argument("--output-dir", default="outputs/evaluation", type=str)
    args = parser.parse_args()

    setup_logging()
    out_root = Path(args.output_dir)
    summary_dir = out_root / "summary"
    sensitivity_dir = out_root / "sensitivity"

    print("=" * 80)
    print("FULL EVALUATION PIPELINE: SYNTHETIC-FIRST + SINGLE REAL SAMPLE")
    print("=" * 80)
    print("\nOPTIMIZATIONS:")
    print("  • All synthetic data processed first (fast feedback)")
    print("  • Single representative real sample at the end")
    print("  • Incremental progress tracking")
    print("  • Estimated speedup: ~80% for initial results")

    try:
        from src.evaluation.build_quantitative_tables_plots import (
            build_tables_and_plots,
        )
        from src.evaluation.evaluate_pc import (
            run_harris_pc_quantitative_evaluation,
            run_sift_geom_pc_quantitative_evaluation,
        )
        from src.evaluation.evaluate_voxel import (
            run_harris3d_quantitative_evaluation,
            run_sift3d_quantitative_evaluation,
        )
    except ImportError as e:
        logging.error(f"Failed to import evaluation modules: {e}")
        return 1

    if args.summary_only:
        args.skip_tuning = True
        args.skip_real = True
        args.skip_rotation_output = True

    # --- Stage 1-4: Synthetic Quantitative Evaluation (FAST) ---
    print("\n" + "=" * 80)
    print("STAGE 1-4: SYNTHETIC QUANTITATIVE EVALUATION")
    print("=" * 80)

    eval_tasks = [
        (
            "Harris PC",
            run_harris_pc_quantitative_evaluation,
            out_root / "harris_pc",
            0.05,
        ),
        (
            "SIFT-Radii",
            run_sift_geom_pc_quantitative_evaluation,
            out_root / "sift_radii",
            0.05,
        ),
        ("Harris3D", run_harris3d_quantitative_evaluation, out_root / "harris3d", 2.0),
        ("SIFT3D", run_sift3d_quantitative_evaluation, out_root / "sift3d", 2.0),
    ]

    reports = {
        name.lower().replace(" ", "_"): path / "quantitative_report.json"
        for name, _, path, _ in eval_tasks
    }
    start_time = time.time()

    if args.summary_only:
        print(
            "  [summary-only] Skipping evaluations — using existing quantitative_report.json files"
        )
    else:
        reports = {}
        for idx, (name, func, path, radius) in enumerate(eval_tasks, 1):
            print(f"\n[{idx}/4] {name}...")
            try:
                run_detector_on_dataset(
                    func, name, path, radius, dataset_type="synthetic"
                )
                reports[name.lower().replace(" ", "_")] = (
                    path / "quantitative_report.json"
                )
            except Exception as e:
                logging.error(f"{name} evaluation failed: {e}")

        synthetic_time = time.time() - start_time
        print(f"\n✓ Synthetic evaluations completed in {synthetic_time:.1f}s")

    # --- Stage 5: Parameter Sensitivity Analysis ---
    if not args.skip_tuning:
        print("\n" + "=" * 80)
        print("STAGE 5: PARAMETER SENSITIVITY SWEEPS")
        print("=" * 80)

        voxel_data = load_voxel_data(ALL_SHAPES)
        pc_data = load_pc_data(ALL_SHAPES)

        for det_idx, (det_name, reg) in enumerate(DETECTOR_REGISTRY.items(), 1):
            print(f"\n[{det_idx}/{len(DETECTOR_REGISTRY)}] {det_name}...")
            try:
                shapes_data = voxel_data if reg["is_voxel"] else pc_data

                results = run_detector_sweep(
                    detector_name=det_name,
                    shapes_data=shapes_data,
                    is_voxel=reg["is_voxel"],
                    configs=reg["configs"],
                    base_factory=reg["factory"],
                )

                df = aggregate_to_dataframe(results)
                sensitivity_dir.mkdir(parents=True, exist_ok=True)
                csv_path = sensitivity_dir / f"{det_name}_sensitivity.csv"
                df.to_csv(csv_path, index=False)
                plot_detector(df, det_name, reg["configs"], sensitivity_dir)
                print(f"  ✓ {det_name} sensitivity saved ({len(df)} records)")
            except Exception as e:
                logging.error(f"Sensitivity sweep for {det_name} failed: {e}")

    # --- Stage 6: Per-Sample Perturbation Output ---
    if not args.skip_rotation_output:
        print("\n" + "=" * 80)
        print("STAGE 6: PERTURBATION ALL SAMPLES OUTPUT (rotation, noise, downsample)")
        print("=" * 80)
        rot_start = time.time()
        run_all_perturbations_output(out_root / "perturbation_all_samples")
        print(
            f"\n✓ Perturbation all-samples output done ({time.time() - rot_start:.1f}s)"
        )
        print(f"  Folder: {out_root / 'perturbation_all_samples'}")

    # --- Stage 7: Single Real Sample Evaluation ---
    if not args.skip_real:
        real_start = time.time()
        real_reports = run_single_real_sample_evaluation(out_root)
        real_time = time.time() - real_start

        # Save real sample results
        real_results_path = out_root / "real_sample_results.json"
        out_root.mkdir(parents=True, exist_ok=True)
        with real_results_path.open("w") as f:
            json.dump(real_reports, f, indent=2)
        print(f"\n✓ Real sample results saved ({real_time:.1f}s)")
        print(f"  File: {real_results_path}")

        # Merge single-sample 'real' quantitative metrics into per-detector reports
        print(
            "\nMerging single-sample real quantitative metrics into per-detector reports..."
        )
        for name, func, path, radius in eval_tasks:
            try:
                # Run a real-only quantitative evaluation limited to one sample in a temp dir
                with tempfile.TemporaryDirectory() as td:
                    real_report = func(
                        out_dir=str(td),
                        random_seed=0,
                        match_radius=radius,
                        dataset_type="real",
                        max_real_samples=1,
                    )

                target = path / "quantitative_report.json"
                if target.exists():
                    with target.open("r", encoding="utf-8") as f:
                        base = json.load(f)
                    base_rows = base.get("rows", [])
                    base_rows.extend(real_report.get("rows", []))
                    base["rows"] = base_rows
                    base_summary = base.get("summary", {})
                    real_summary = real_report.get("summary", {})
                    base_summary.update(real_summary)
                    base["summary"] = base_summary
                    base_config = base.get("config", {})
                    base_config["dataset_type"] = "both"
                    base["config"] = base_config
                    with target.open("w", encoding="utf-8") as f:
                        json.dump(base, f, indent=2)
                    print(f"  ✓ Merged real metrics into {target}")
                else:
                    path.mkdir(parents=True, exist_ok=True)
                    with target.open("w", encoding="utf-8") as f:
                        json.dump(real_report, f, indent=2)
                    print(f"  ✓ Saved real-only report to {target}")
            except TypeError:
                # Fallback if the evaluation function doesn't accept dataset_type/max_real_samples
                try:
                    with tempfile.TemporaryDirectory() as td:
                        real_report = func(
                            out_dir=str(td), random_seed=0, match_radius=radius
                        )

                    target = path / "quantitative_report.json"
                    if target.exists():
                        with target.open("r", encoding="utf-8") as f:
                            base = json.load(f)
                        base_rows = base.get("rows", [])
                        base_rows.extend(real_report.get("rows", []))
                        base["rows"] = base_rows
                        base_summary = base.get("summary", {})
                        real_summary = real_report.get("summary", {})
                        base_summary.update(real_summary)
                        base["summary"] = base_summary
                        base_config = base.get("config", {})
                        base_config["dataset_type"] = "both"
                        base["config"] = base_config
                        with target.open("w", encoding="utf-8") as f:
                            json.dump(base, f, indent=2)
                        print(f"  ✓ Merged real metrics into {target}")
                    else:
                        path.mkdir(parents=True, exist_ok=True)
                        with target.open("w", encoding="utf-8") as f:
                            json.dump(real_report, f, indent=2)
                        print(f"  ✓ Saved real-only report to {target}")
                except Exception as e:
                    logging.error(
                        f"Failed to merge legacy real metrics for {name}: {e}"
                    )
            except Exception as e:
                logging.error(f"Failed to merge real metrics for {name}: {e}")
        # Move real-sample plots into per-detector output folders for consistent structure
        pc_dir = out_root / "real_sample_pc"
        voxel_dir = out_root / "real_sample_voxel"
        copy_map = {
            pc_dir / "harris_real_sample.png": out_root
            / "harris_pc"
            / "harris_real_sample.png",
            pc_dir / "sift_real_sample.png": out_root
            / "sift_radii"
            / "sift_real_sample.png",
            voxel_dir / "harris3d_real_sample.png": out_root
            / "harris3d"
            / "harris3d_real_sample.png",
            voxel_dir / "sift3d_real_sample.png": out_root
            / "sift3d"
            / "sift3d_real_sample.png",
        }
        for src, dst in copy_map.items():
            try:
                if src.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    print(f"  ✓ Copied {src} -> {dst}")
            except Exception as e:
                logging.error(f"Failed to copy real-sample plot {src} to {dst}: {e}")
    else:
        print("\n[SKIPPED] Real data evaluation (use --skip-real to skip)")

    # --- Stage 7: Aggregate Results ---
    print("\n" + "=" * 80)
    print("STAGE 7: BUILDING SUMMARY")
    print("=" * 80)

    try:
        summary_dir.mkdir(parents=True, exist_ok=True)
        build_tables_and_plots(reports, summary_dir)
        print(f"✓ Summary tables and plots saved to {summary_dir}")
    except Exception as e:
        logging.error(f"Summary build failed: {e}")

    # --- Final Report ---
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nTotal execution time: {total_time:.1f}s ({total_time / 60:.1f}m)")
    print(f"\nResults saved to: {out_root}")
    print(f"  ├─ Synthetic evaluations: {out_root}/*")
    if not args.skip_tuning:
        print(f"  ├─ Sensitivity analysis: {sensitivity_dir}/")
    if not args.skip_rotation_output:
        print(
            f"  ├─ Perturbation all samples: {out_root / 'perturbation_all_samples'}/"
        )
    if not args.skip_real:
        print(f"  ├─ Real samples: {out_root}/real_sample_*/")
    print(f"  └─ Summary: {summary_dir}/")

    return 0


if __name__ == "__main__":
    main()
