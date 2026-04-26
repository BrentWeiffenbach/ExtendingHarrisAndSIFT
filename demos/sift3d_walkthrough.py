"""Step-by-step visualization of the 3D SIFT voxel pipeline.

Walks through every major stage in order:
  1. Original volume — orthogonal cross-section slices
  2. Gaussian pyramid — per-octave/scale mid-slice grid
  3. Difference of Gaussians — per-octave/DoG mid-slice grid
  4. Extrema responses — 3D scatter colored by |DoG response|

Usage
-----
    python -m demos.sift3d_walkthrough
    python -m demos.sift3d_walkthrough --shape pyramid
    python -m demos.sift3d_walkthrough --shape cube --modelnet 3
"""

from __future__ import annotations

import argparse

import numpy as np
from matplotlib import cm, colors
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from src.common.io import ModelNetLoader, SyntheticVoxelLoader
from src.voxel.params import SIFT3DParams
from src.voxel.sift3d import SIFT3DGaussianResult, SIFT3DVoxel

_CMAP_GRAY = "gray"
_CMAP_DOG = "RdBu_r"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_volume(
    shape: str | None, modelnet_index: int | None
) -> tuple[np.ndarray, str]:
    if modelnet_index is not None:
        loader = ModelNetLoader("data/Voxel/real/ModelNet10-dataset/modelnet10.npy.gz")
        vol = loader.load_by_index(modelnet_index)
        return vol, f"ModelNet[{modelnet_index}]"
    name = shape or "sphere"
    vol = SyntheticVoxelLoader().load_by_name(name)
    return vol, name


# ---------------------------------------------------------------------------
# Step 1 — Original volume
# ---------------------------------------------------------------------------


def _plot_original_volume(volume: np.ndarray, title: str) -> None:
    D, H, W = volume.shape
    mid_z, mid_y, mid_x = D // 2, H // 2, W // 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(
        f"Step 1 — Original Volume: '{title}'  shape={volume.shape}", fontsize=11
    )

    axes[0].imshow(volume[mid_z], cmap=_CMAP_GRAY, origin="lower")
    axes[0].set_title(f"Z slice (z={mid_z})")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")

    axes[1].imshow(volume[:, mid_y, :], cmap=_CMAP_GRAY, origin="lower")
    axes[1].set_title(f"Y slice (y={mid_y})")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Z")

    axes[2].imshow(volume[:, :, mid_x], cmap=_CMAP_GRAY, origin="lower")
    axes[2].set_title(f"X slice (x={mid_x})")
    axes[2].set_xlabel("Y")
    axes[2].set_ylabel("Z")

    for ax in axes:
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Step 2 — Gaussian pyramid
# ---------------------------------------------------------------------------


def _plot_gaussian_pyramid(result: SIFT3DGaussianResult) -> None:
    gp = result.gaussian_pyramid
    sp = result.sigma_pyramid
    n_oct = len(gp)
    n_sc = max(len(o) for o in gp) if gp else 0
    if n_oct == 0 or n_sc == 0:
        print("[gaussian pyramid] nothing to show")
        return

    fig, axes = plt.subplots(
        n_oct, n_sc, figsize=(2.6 * n_sc, 2.6 * n_oct), squeeze=False
    )
    fig.suptitle(
        "Step 2 — Gaussian Pyramid  (middle Z slice per octave × scale)", fontsize=11
    )

    for o in range(n_oct):
        for s in range(n_sc):
            ax = axes[o][s]
            if s < len(gp[o]):
                vol = gp[o][s]
                mid_z = vol.shape[0] // 2
                sigma = sp[o][s]
                ax.imshow(vol[mid_z], cmap=_CMAP_GRAY, origin="lower", vmin=0, vmax=1)
                ax.set_title(f"Oct {o}, s {s}\nσ={sigma:.2f}", fontsize=7)
                ax.set_xticks([])
                ax.set_yticks([])
                if s == 0:
                    d, h, w = vol.shape
                    ax.set_ylabel(f"Oct {o}\n{d}×{h}×{w}", fontsize=7)
            else:
                ax.axis("off")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Step 3 — Difference of Gaussians
# ---------------------------------------------------------------------------


def _plot_dog_pyramid(result: SIFT3DGaussianResult) -> None:
    dp = result.dog_pyramid
    pairs = result.dog_sigma_pairs
    n_oct = len(dp)
    n_dog = max(len(o) for o in dp) if dp else 0
    if n_oct == 0 or n_dog == 0:
        print("[DoG pyramid] nothing to show")
        return

    fig, axes = plt.subplots(
        n_oct, n_dog, figsize=(2.6 * n_dog, 2.6 * n_oct), squeeze=False
    )
    fig.suptitle(
        "Step 3 — Difference of Gaussians  (middle Z slice, diverging colormap)",
        fontsize=11,
    )

    for o in range(n_oct):
        for d in range(n_dog):
            ax = axes[o][d]
            if d < len(dp[o]):
                dog = dp[o][d]
                mid_z = dog.shape[0] // 2
                vmax = float(np.abs(dog).max()) or 1e-6
                sigma_low, sigma_high, _ = pairs[o][d]
                im = ax.imshow(
                    dog[mid_z],
                    cmap=_CMAP_DOG,
                    origin="lower",
                    vmin=-vmax,
                    vmax=vmax,
                )
                ax.set_title(
                    f"Oct {o}, DoG {d}\nσ {sigma_low:.2f}→{sigma_high:.2f}",
                    fontsize=7,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                plt.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
                if d == 0:
                    ax.set_ylabel(f"Oct {o}", fontsize=7)
            else:
                ax.axis("off")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Step 4 — Extrema responses
# ---------------------------------------------------------------------------


def _plot_extrema_responses(result: SIFT3DGaussianResult, title: str) -> None:
    vol = result.original_volume
    eg = result.extrema_global  # (N, 7): z,y,x,sigma,response,octave,s_index

    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(
        f"Step 4 — Detected Extrema  '{title}'  ({eg.shape[0]} keypoints)",
        fontsize=11,
    )

    # ---- 4a: 3D scatter of extrema colored by |response| -------------------
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")

    # Background: occupied voxels
    occupied = np.argwhere(vol > 0.5)
    if occupied.shape[0] > 0:
        subsample = np.random.default_rng(0).choice(
            occupied.shape[0], min(2000, occupied.shape[0]), replace=False
        )
        occ = occupied[subsample]
        ax3d.scatter(
            occ[:, 2],
            occ[:, 1],
            occ[:, 0],
            c="steelblue",
            s=2,
            alpha=0.15,
            label="voxels",
        )

    # Extrema
    if eg.shape[0] > 0:
        z_kp, y_kp, x_kp = eg[:, 0], eg[:, 1], eg[:, 2]
        responses = np.abs(eg[:, 4])
        sigmas = eg[:, 3]
        sizes = np.clip(sigmas * 40, 20, 200)
        vmax = float(np.percentile(responses, 95)) or 1e-6
        sc = ax3d.scatter(
            x_kp,
            y_kp,
            z_kp,
            c=responses,
            s=sizes,
            cmap="hot",
            vmin=0,
            vmax=vmax,
            edgecolors="red",
            linewidths=0.4,
            alpha=0.9,
            zorder=5,
            label="extrema",
        )
        fig.colorbar(sc, ax=ax3d, fraction=0.03, pad=0.12, label="|response|")

    ax3d.set_xlabel("X", fontsize=8)
    ax3d.set_ylabel("Y", fontsize=8)
    ax3d.set_zlabel("Z", fontsize=8)
    ax3d.set_title("3D scatter\n(size ∝ σ, color = |response|)", fontsize=9)
    ax3d.tick_params(labelsize=6)
    ax3d.view_init(elev=25, azim=45)

    # ---- 4b: per-octave breakdown ------------------------------------------
    ax_bar = fig.add_subplot(1, 2, 2)

    counts_per_octave: dict[int, int] = {}
    mean_response_per_octave: dict[int, float] = {}
    if eg.shape[0] > 0:
        for oct_idx in np.unique(eg[:, 5]).astype(int):
            mask = eg[:, 5].astype(int) == oct_idx
            counts_per_octave[oct_idx] = int(mask.sum())
            mean_response_per_octave[oct_idx] = float(np.abs(eg[mask, 4]).mean())

    if counts_per_octave:
        octaves = sorted(counts_per_octave)
        counts = [counts_per_octave[o] for o in octaves]
        means = [mean_response_per_octave[o] for o in octaves]

        x_pos = np.arange(len(octaves))
        color = plt.cm.tab10(np.linspace(0, 0.5, len(octaves)))  # type: ignore[attr-defined]
        bars = ax_bar.bar(x_pos, counts, color=color)
        ax_bar.set_xticks(x_pos)
        ax_bar.set_xticklabels([f"Oct {o}" for o in octaves], fontsize=9)
        ax_bar.set_ylabel("# keypoints", fontsize=9)
        ax_bar.set_title("Keypoints & mean |response| per octave", fontsize=9)

        ax_r = ax_bar.twinx()
        ax_r.plot(
            x_pos, means, "D--", color="firebrick", markersize=6, label="mean |resp|"
        )
        ax_r.set_ylabel("mean |response|", fontsize=9, color="firebrick")
        ax_r.tick_params(axis="y", labelcolor="firebrick")

        for bar, cnt in zip(bars, counts):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                str(cnt),
                ha="center",
                va="bottom",
                fontsize=8,
            )
    else:
        ax_bar.text(
            0.5,
            0.5,
            "No extrema detected",
            ha="center",
            va="center",
            transform=ax_bar.transAxes,
            fontsize=12,
        )
        ax_bar.set_title("Keypoints per octave", fontsize=9)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Step 4b — Response distribution analysis
# ---------------------------------------------------------------------------


def _plot_response_analysis(result: SIFT3DGaussianResult, title: str) -> None:
    eg = result.extrema_global
    if eg.shape[0] == 0:
        print("[response analysis] no extrema to plot")
        return

    responses = np.abs(eg[:, 4])
    sigmas = eg[:, 3]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"Step 4 (detail) — Response Analysis  '{title}'", fontsize=11)

    # Histogram of |response|
    axes[0].hist(responses, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
    axes[0].axvline(
        float(np.median(responses)), color="red", linestyle="--", label="median"
    )
    axes[0].set_xlabel("|DoG response|", fontsize=9)
    axes[0].set_ylabel("count", fontsize=9)
    axes[0].set_title("|Response| distribution", fontsize=9)
    axes[0].legend(fontsize=8)

    # Response vs sigma
    axes[1].scatter(sigmas, responses, s=12, alpha=0.6, c="darkorange")
    axes[1].set_xlabel("σ (characteristic scale)", fontsize=9)
    axes[1].set_ylabel("|DoG response|", fontsize=9)
    axes[1].set_title("Response vs Scale", fontsize=9)

    # Sigma histogram
    axes[2].hist(sigmas, bins=20, color="seagreen", edgecolor="white", alpha=0.85)
    axes[2].set_xlabel("σ (characteristic scale)", fontsize=9)
    axes[2].set_ylabel("count", fontsize=9)
    axes[2].set_title("Scale distribution", fontsize=9)

    for ax in axes:
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Step 3b — Interactive DoG viewer (all sides)
# ---------------------------------------------------------------------------


def _plot_dog_pyramid_interactive(result: SIFT3DGaussianResult) -> None:
    dp = result.dog_pyramid
    pairs = result.dog_sigma_pairs
    if not dp or not dp[0]:
        print("[DoG interactive] nothing to show")
        return

    n_oct = len(dp)
    n_dog = max(len(o) for o in dp)

    vol0 = dp[0][0]
    D, H, W = vol0.shape

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    plt.subplots_adjust(bottom=0.32)
    fig.suptitle("Step 3 — DoG  (interactive: all sides)", fontsize=11)

    def _get_dog(o: int, d: int) -> np.ndarray:
        o = int(np.clip(o, 0, len(dp) - 1))
        d = int(np.clip(d, 0, len(dp[o]) - 1))
        return dp[o][d]

    def _draw(o: int, d: int, iz: int, iy: int, ix: int) -> None:
        dog = _get_dog(o, d)
        dD, dH, dW = dog.shape
        iz = int(np.clip(iz, 0, dD - 1))
        iy = int(np.clip(iy, 0, dH - 1))
        ix = int(np.clip(ix, 0, dW - 1))
        vmax = float(np.abs(dog).max()) or 1e-6
        kw = dict(cmap=_CMAP_DOG, origin="lower", vmin=-vmax, vmax=vmax, aspect="auto")
        sl, sh, _ = pairs[o][d] if d < len(pairs[o]) else (0, 0, 0)
        for ax, img, lbl in [
            (axes[0], dog[iz, :, :], f"Z={iz}"),
            (axes[1], dog[:, iy, :], f"Y={iy}"),
            (axes[2], dog[:, :, ix], f"X={ix}"),
        ]:
            ax.clear()
            ax.imshow(img, **kw)
            ax.set_title(
                f"Oct {o}  DoG {d}  σ {sl:.2f}→{sh:.2f}\n{lbl}  vol {dog.shape}",
                fontsize=8,
            )
            ax.tick_params(labelsize=6)
        fig.canvas.draw_idle()

    # Sliders
    sl_oct = Slider(fig.add_axes((0.12, 0.22, 0.78, 0.025)), "Octave",
                    0, max(0, n_oct - 1), valinit=0, valstep=1)
    sl_dog = Slider(fig.add_axes((0.12, 0.18, 0.78, 0.025)), "DoG",
                    0, max(0, n_dog - 1), valinit=0, valstep=1)
    sl_z   = Slider(fig.add_axes((0.12, 0.13, 0.78, 0.025)), "Z slice",
                    0, max(0, D - 1), valinit=D // 2, valstep=1)
    sl_y   = Slider(fig.add_axes((0.12, 0.09, 0.78, 0.025)), "Y slice",
                    0, max(0, H - 1), valinit=H // 2, valstep=1)
    sl_x   = Slider(fig.add_axes((0.12, 0.05, 0.78, 0.025)), "X slice",
                    0, max(0, W - 1), valinit=W // 2, valstep=1)

    def _on_change(_val) -> None:
        _draw(int(sl_oct.val), int(sl_dog.val),
              int(sl_z.val), int(sl_y.val), int(sl_x.val))

    for sl in (sl_oct, sl_dog, sl_z, sl_y, sl_x):
        sl.on_changed(_on_change)

    _draw(0, 0, D // 2, H // 2, W // 2)
    plt.show()


# ---------------------------------------------------------------------------
# Step 4b — Interactive extrema viewer (all sides, circles ∝ σ)
# ---------------------------------------------------------------------------


def _plot_extrema_interactive(result: SIFT3DGaussianResult, title: str) -> None:
    vol = result.original_volume
    eg = result.extrema_global
    D, H, W = vol.shape

    has_kp = eg.shape[0] > 0
    if has_kp:
        zyx = eg[:, :3]
        global_sigma = eg[:, 3] * (2.0 ** eg[:, 5])
        radii = np.sqrt(2.0) * global_sigma
        responses = eg[:, 4]
        max_abs = float(np.max(np.abs(responses))) or 1e-6
        norm = colors.Normalize(vmin=-max_abs, vmax=max_abs)
        cmap_resp = cm.get_cmap("RdBu_r")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    plt.subplots_adjust(bottom=0.22)
    fig.suptitle(
        f"Step 4 — Extrema  '{title}'  ({eg.shape[0]} keypoints)\n"
        "circles = keypoints near slice  |  radius ∝ σ  |  color = DoG response",
        fontsize=10,
    )

    def _draw(iz: int, iy: int, ix: int) -> None:
        configs = [
            (axes[0], vol[iz, :, :],  zyx[:, 2] if has_kp else None,
             zyx[:, 1] if has_kp else None, zyx[:, 0] if has_kp else None,
             f"Z={iz}", "X", "Y"),
            (axes[1], vol[:, iy, :],  zyx[:, 2] if has_kp else None,
             zyx[:, 0] if has_kp else None, zyx[:, 1] if has_kp else None,
             f"Y={iy}", "X", "Z"),
            (axes[2], vol[:, :, ix],  zyx[:, 1] if has_kp else None,
             zyx[:, 0] if has_kp else None, zyx[:, 2] if has_kp else None,
             f"X={ix}", "Y", "Z"),
        ]
        slice_coords = [iz, iy, ix]
        axis_cols    = [0, 1, 2]

        for (ax, img, hc, vc, ac, lbl, hl, vl), sc, acol in zip(
            configs, slice_coords, axis_cols
        ):
            ax.clear()
            ax.imshow(img, cmap=_CMAP_GRAY, origin="lower", aspect="auto")
            ax.set_title(lbl, fontsize=9)
            ax.set_xlabel(hl, fontsize=8)
            ax.set_ylabel(vl, fontsize=8)
            ax.tick_params(labelsize=6)

            if has_kp:
                near = np.abs(zyx[:, acol] - sc) <= global_sigma
                for h, v, r, resp in zip(hc[near], vc[near], radii[near], responses[near]):
                    ax.add_patch(Circle(
                        (float(h), float(v)), radius=float(r),
                        fill=False, edgecolor=cmap_resp(norm(float(resp))),
                        linewidth=1.2, alpha=0.85,
                    ))

        fig.canvas.draw_idle()

    sl_z = Slider(fig.add_axes((0.12, 0.13, 0.78, 0.025)), "Z slice",
                  0, max(0, D - 1), valinit=D // 2, valstep=1)
    sl_y = Slider(fig.add_axes((0.12, 0.09, 0.78, 0.025)), "Y slice",
                  0, max(0, H - 1), valinit=H // 2, valstep=1)
    sl_x = Slider(fig.add_axes((0.12, 0.05, 0.78, 0.025)), "X slice",
                  0, max(0, W - 1), valinit=W // 2, valstep=1)

    def _on_change(_val) -> None:
        _draw(int(sl_z.val), int(sl_y.val), int(sl_x.val))

    for sl in (sl_z, sl_y, sl_x):
        sl.on_changed(_on_change)

    _draw(D // 2, H // 2, W // 2)

    if has_kp:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap_resp)
        sm.set_array([])
        fig.colorbar(sm, ax=axes[-1], shrink=0.75, label="DoG response")

    plt.show()


# ---------------------------------------------------------------------------
# Main walkthrough
# ---------------------------------------------------------------------------


def run_walkthrough(
    shape: str | None = None, modelnet_index: int | None = None
) -> None:
    volume, label = _load_volume(shape, modelnet_index)
    print(f"Loaded '{label}'  volume shape: {volume.shape}")

    params = SIFT3DParams()
    detector = SIFT3DVoxel(params)
    print("Running 3D SIFT pipeline...")
    result = detector.run(volume)

    n_octaves = len(result.gaussian_pyramid)
    n_scales = [len(o) for o in result.gaussian_pyramid]
    n_dogs = [len(o) for o in result.dog_pyramid]
    n_extrema = result.extrema_global.shape[0]
    print(f"  Octaves: {n_octaves}  |  Scales/oct: {n_scales}  |  DoGs/oct: {n_dogs}")
    print(f"  Detected extrema (global): {n_extrema}")

    print("\n-- Step 1: Original volume --")
    _plot_original_volume(volume, label)

    print("-- Step 2: Gaussian pyramid --")
    _plot_gaussian_pyramid(result)

    print("-- Step 3: Difference of Gaussians --")
    _plot_dog_pyramid(result)
    _plot_dog_pyramid_interactive(result)

    print("-- Step 4: Extrema responses --")
    _plot_extrema_responses(result, label)
    _plot_extrema_interactive(result, label)
    _plot_response_analysis(result, label)

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="3D SIFT voxel step-by-step walkthrough"
    )
    parser.add_argument(
        "--shape",
        type=str,
        default=None,
        help="Synthetic shape name (cone/cube/cuboid/cylinder/pyramid/sphere/torus)",
    )
    parser.add_argument(
        "--modelnet",
        type=int,
        default=None,
        dest="modelnet_index",
        help="ModelNet10 sample index (overrides --shape)",
    )
    args = parser.parse_args()
    run_walkthrough(shape=args.shape, modelnet_index=args.modelnet_index)


if __name__ == "__main__":
    main()
