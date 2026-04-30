# ExtendingHarrisAndSIFT
Extending Harris &amp; SIFT from 2D to 3D for CS545 at WPI
# 3D Feature Detection Project

This repository contains implementations and analyses of 3D keypoint detectors on both voxel grids and point clouds, including Harris 3D and SIFT 3D.

## Usage

Use the two CLI entry points below for most experiments.

### `main.py` (single run / interactive)

Run one detector on one input volume or image:

```bash
# Harris on a synthetic voxel
python main.py --detector harris --dimension 3d --synthetic-name cube --show

# SIFT on a ModelNet sample
python main.py --detector sift --dimension 3d --modelnet-index 100 --show

# Save only (no UI)
python main.py --detector sift --dimension 3d --modelnet-index 100 --no-show --output-dir /tmp

# 2D SIFT on the default image in data/2d
python main.py --detector sift --dimension 2d --show
```

Run demo pipelines:

```bash
# Default demo source
python main.py --demo 3d-extrema

# Demo on a chosen synthetic shape
python main.py --demo 3d-dog --synthetic-name torus

# Demo on a chosen ModelNet sample
python main.py --demo 3d-gaussian --modelnet-index 10
```

Notes:
- In 3D mode with `--show`, napari is used when available.
- `--synthetic-name` and `--modelnet-index` are ignored for 2D detector mode.

### `run_all.py` (batch runner)

Run full or filtered batches over datasets:

```bash
# Everything (2D + 3D, Harris + SIFT)
python run_all.py

# Only 3D SIFT
python run_all.py --dimension 3d --detector sift

# Only 2D Harris (currently prints skip messages)
python run_all.py --dimension 2d --detector harris
```

CLI filters:
- `--dimension {2d,3d,all}`
- `--detector {harris,sift,all}`

Outputs are written under:
- `outputs/run_all/3d/harris/voxel/*.png`
- `outputs/run_all/3d/harris/pointcloud/*.png`
- `outputs/run_all/3d/sift/voxel/*.png`
- `outputs/run_all/2d/sift/image/*.png`

### Point Cloud Harris evaluation

Run the standalone evaluation script for Harris PC (synthetic + real point clouds):

```bash
python -m src.evaluation.evaluate_pc
```

This generates:
- `outputs/harris_pc/<shape>_pc.png` — per-shape keypoint visualisation

### Quantitative Evaluation

- point cloud Harris: `src.evaluation.evaluate_pc.run_harris_pc_quantitative_evaluation`
- voxel Harris: `src.evaluation.evaluate_voxel.run_harris3d_quantitative_evaluation`

Metrics reported:

- repeatability under rotation, noise, and downsampling
- localization error after alignment
- keypoint count stability (mean/std/CV)

Run all quantitative evaluations:

```bash
# From repo root
source .venv/bin/activate

python -m src.evaluation.evaluate_pc
python -m src.evaluation.evaluate_voxel
```

This writes machine-readable reports:

- `outputs/evaluation/harris_pc/quantitative_report.json`
- `outputs/evaluation/harris3d/quantitative_report.json`

Build aggregate tables and plots:

```bash
python -m src.evaluation.build_quantitative_tables_plots \
	--pc-report outputs/evaluation/harris_pc/quantitative_report.json \
	--voxel-report outputs/evaluation/harris3d/quantitative_report.json \
	--out-dir outputs/evaluation/summary
```

Generated analysis artifacts:

- `outputs/evaluation/summary/summary_metrics.csv`
- `outputs/evaluation/summary/summary_metrics.md`
- `outputs/evaluation/summary/repeatability_mean.png`
- `outputs/evaluation/summary/count_cv.png`
- `outputs/evaluation/summary/localization_error_mean.png`

---
