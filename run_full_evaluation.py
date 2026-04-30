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
import json
import logging
import time
from pathlib import Path
import tempfile
import shutil

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


def run_detector_on_dataset(func, detector_name, out_dir, radius, dataset_type="synthetic"):
    """Wrapper to run a detector evaluation with progress tracking."""
    start = time.time()
    print(f"  ⏱  Starting {detector_name} ({dataset_type})...")
    try:
        result = func(out_dir=str(out_dir), random_seed=0, match_radius=radius, dataset_type=dataset_type)
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
            print("  → SIFT-Geom PC...")
            sift_detector = SIFTRadiiPC(SIFTRadiiPCParams())
            sift_kps = sift_detector.detect(pts)
            plot_pointcloud(
                [pts],
                titles=[f"{sample_name}: SIFT-Geom PC ({sift_kps.shape[0]} kps)"],
                keypoints_list=[sift_kps],
                show=False,
                save_path=str(pc_out / "sift_real_sample.png"),
            )
            print(f"    ✓ SIFT-Geom PC: {sift_kps.shape[0]} keypoints")
            
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


def main():
    parser = argparse.ArgumentParser(description="Optimized evaluation pipeline: synthetic-first + single real sample")
    parser.add_argument(
        "--skip-tuning", action="store_true", help="Skip parameter sensitivity stage"
    )
    parser.add_argument("--skip-real", action="store_true", help="Skip real data evaluation")
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
            "SIFT-Geom PC",
            run_sift_geom_pc_quantitative_evaluation,
            out_root / "sift_geom_pc",
            0.05,
        ),
        ("Harris3D", run_harris3d_quantitative_evaluation, out_root / "harris3d", 2.0),
        ("SIFT3D", run_sift3d_quantitative_evaluation, out_root / "sift3d", 2.0),
    ]

    reports = {}
    start_time = time.time()
    
    for idx, (name, func, path, radius) in enumerate(eval_tasks, 1):
        print(f"\n[{idx}/4] {name}...")
        try:
            run_detector_on_dataset(func, name, path, radius, dataset_type="synthetic")
            reports[name.lower().replace(" ", "_")] = path / "quantitative_report.json"
        except Exception as e:
            logging.error(f"{name} evaluation failed: {e}")
            # Continue with other detectors even if one fails
    
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

    # --- Stage 6: Single Real Sample Evaluation ---
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
        print("\nMerging single-sample real quantitative metrics into per-detector reports...")
        for name, func, path, radius in eval_tasks:
            try:
                # Run a real-only quantitative evaluation limited to one sample in a temp dir
                with tempfile.TemporaryDirectory() as td:
                    real_report = func(out_dir=str(td), random_seed=0, match_radius=radius, dataset_type="real", max_real_samples=1)

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
                        real_report = func(out_dir=str(td), random_seed=0, match_radius=radius)

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
                    logging.error(f"Failed to merge legacy real metrics for {name}: {e}")
            except Exception as e:
                logging.error(f"Failed to merge real metrics for {name}: {e}")
        # Move real-sample plots into per-detector output folders for consistent structure
        pc_dir = out_root / "real_sample_pc"
        voxel_dir = out_root / "real_sample_voxel"
        copy_map = {
            pc_dir / "harris_real_sample.png": out_root / "harris_pc" / "harris_real_sample.png",
            pc_dir / "sift_real_sample.png": out_root / "sift_geom_pc" / "sift_real_sample.png",
            voxel_dir / "harris3d_real_sample.png": out_root / "harris3d" / "harris3d_real_sample.png",
            voxel_dir / "sift3d_real_sample.png": out_root / "sift3d" / "sift3d_real_sample.png",
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
    print(f"\nTotal execution time: {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"\nResults saved to: {out_root}")
    print(f"  ├─ Synthetic evaluations: {out_root}/*")
    if not args.skip_tuning:
        print(f"  ├─ Sensitivity analysis: {sensitivity_dir}/")
    if not args.skip_real:
        print(f"  ├─ Real samples: {out_root}/real_sample_*/")
    print(f"  └─ Summary: {summary_dir}/")
    
    return 0


if __name__ == "__main__":
    main()
