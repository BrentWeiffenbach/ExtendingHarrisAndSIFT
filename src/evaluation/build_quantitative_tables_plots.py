from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_report(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _flatten_summary(detector: str, report: dict) -> list[dict]:
    rows: list[dict] = []
    for dataset, per_dataset in report.get("summary", {}).items():
        for perturbation, stats in per_dataset.items():
            rows.append(
                {
                    "detector": detector,
                    "dataset": dataset,
                    "perturbation": perturbation,
                    "repeatability_mean": float(stats["repeatability_mean"]),
                    "repeatability_std": float(stats["repeatability_std"]),
                    "localization_error_mean": (
                        None
                        if stats["localization_error_mean"] is None
                        else float(stats["localization_error_mean"])
                    ),
                    "localization_error_std": (
                        None
                        if stats["localization_error_std"] is None
                        else float(stats["localization_error_std"])
                    ),
                    "count_stability": (
                        float(stats["count_stability"]["retention"])
                        if stats["count_stability"].get("retention") is not None
                        else 1.0 / (1.0 + float(stats["count_stability"]["cv"]))
                    ),
                    "num_trials": int(stats["num_trials"]),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "detector",
        "dataset",
        "perturbation",
        "repeatability_mean",
        "repeatability_std",
        "localization_error_mean",
        "localization_error_std",
        "count_stability",
        "num_trials",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown_table(path: Path, rows: list[dict]) -> None:
    header = (
        "| Detector | Dataset | Perturbation | Repeatability (mean±std) | "
        "Localization Error (mean±std) | Count Stability | Trials |\n"
        "|---|---|---|---:|---:|---:|---:|\n"
    )
    lines = [header]
    for row in rows:
        loc_mean = row["localization_error_mean"]
        loc_std = row["localization_error_std"]
        loc_str = "n/a"
        if loc_mean is not None and loc_std is not None:
            loc_str = f"{loc_mean:.4f}±{loc_std:.4f}"

        lines.append(
            "| "
            f"{row['detector']} | "
            f"{row['dataset']} | "
            f"{row['perturbation']} | "
            f"{row['repeatability_mean']:.3f}±{row['repeatability_std']:.3f} | "
            f"{loc_str} | "
            f"{row['count_stability']:.3f} | "
            f"{row['num_trials']} |\n"
        )

    with path.open("w", encoding="utf-8") as f:
        f.writelines(lines)


def _plot_metric_grid(
    rows: list[dict],
    out_path: Path,
    metric: str,
    title: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    detectors = sorted({r["detector"] for r in rows})
    datasets = sorted({r["dataset"] for r in rows})
    perturbations = ["rotation", "noise", "downsample"]

    fig, axes = plt.subplots(
        nrows=len(detectors),
        ncols=len(datasets),
        figsize=(5 * len(datasets), 3.5 * len(detectors)),
        squeeze=False,
    )

    for i, detector in enumerate(detectors):
        for j, dataset in enumerate(datasets):
            ax = axes[i][j]
            sub = [
                r for r in rows if r["detector"] == detector and r["dataset"] == dataset
            ]
            values = []
            for p in perturbations:
                match = next((r for r in sub if r["perturbation"] == p), None)
                val = (
                    np.nan
                    if match is None or match[metric] is None
                    else float(match[metric])
                )
                values.append(val)

            x = np.arange(len(perturbations))
            ax.bar(x, values, color=["#3A7CA5", "#7FB069", "#E07A5F"])
            ax.set_xticks(x)
            ax.set_xticklabels(perturbations)
            ax.set_title(f"{detector} | {dataset}")
            ax.set_ylabel(ylabel)
            if ylim is not None:
                ax.set_ylim(ylim)
            ax.grid(axis="y", alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_tables_and_plots(reports: dict[str, Path], out_dir: Path) -> None:
    """Build summary tables and plots from one or more named evaluation reports.

    Args:
        reports: mapping of detector display-name → path to quantitative_report.json.
                 Paths that do not exist are silently skipped.
        out_dir: directory where CSV, Markdown and PNG outputs are written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for detector_name, path in reports.items():
        if path.exists():
            report = _load_report(path)
            rows.extend(_flatten_summary(detector_name, report))

    if not rows:
        missing = ", ".join(str(p) for p in reports.values())
        print(f"No reports found at {missing}; skipping summary build.")
        return

    rows_sorted = sorted(
        rows, key=lambda r: (r["detector"], r["dataset"], r["perturbation"])
    )

    _write_csv(out_dir / "summary_metrics.csv", rows_sorted)
    _write_markdown_table(out_dir / "summary_metrics.md", rows_sorted)

    _plot_metric_grid(
        rows_sorted,
        out_dir / "repeatability_mean.png",
        metric="repeatability_mean",
        title="Repeatability Mean by Detector, Dataset, Perturbation",
        ylabel="Repeatability",
        ylim=(0.0, 1.0),
    )
    _plot_metric_grid(
        rows_sorted,
        out_dir / "count_stability.png",
        metric="count_stability",
        title="Keypoint Count Stability (higher = more robust)",
        ylabel="Count Stability",
        ylim=(0.0, 1.0),
    )
    _plot_metric_grid(
        rows_sorted,
        out_dir / "localization_error_mean.png",
        metric="localization_error_mean",
        title="Localization Error Mean by Detector, Dataset, Perturbation",
        ylabel="Localization Error",
    )


_DEFAULT_REPORTS: dict[str, Path] = {
    "harris_pc": Path("outputs/evaluation/harris_pc/quantitative_report.json"),
    "sift-radii": Path("outputs/evaluation/sift_radii/quantitative_report.json"),
    "harris3d_voxel": Path("outputs/evaluation/harris3d/quantitative_report.json"),
    "sift3d_voxel": Path("outputs/evaluation/sift3d/quantitative_report.json"),
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build aggregate tables and plots from quantitative evaluation JSON reports. "
            "Pass --report NAME PATH pairs to override the defaults."
        )
    )
    parser.add_argument(
        "--report",
        nargs=2,
        metavar=("NAME", "PATH"),
        action="append",
        default=[],
        help=(
            "Add a named report. Can be repeated. "
            "E.g. --report harris_pc outputs/evaluation/harris_pc/quantitative_report.json"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/evaluation/summary"),
        help="Output directory for aggregate CSV/Markdown and plot files.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    reports: dict[str, Path] = (
        {name: Path(path) for name, path in args.report}
        if args.report
        else _DEFAULT_REPORTS
    )
    build_tables_and_plots(reports, args.out_dir)
    print(f"Wrote summary tables/plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
