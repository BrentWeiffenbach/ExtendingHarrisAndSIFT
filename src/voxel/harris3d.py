from typing import Any, cast

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

from src.common.base_detector import Detector3D


def _normalize_spacing(spacing):
    arr = np.asarray(spacing, dtype=np.float64)
    if arr.shape != (3,) or np.any(arr <= 0):
        raise ValueError("spacing must be a 3-tuple of positive values")
    return arr


def _sigma_per_axis(base_sigma, spacing):
    if base_sigma <= 0:
        return np.zeros(3, dtype=np.float64)
    return base_sigma / spacing


class Harris3DVoxel(Detector3D):
    def __init__(self, params):
        super().__init__(params)
        self.last_response = None

    def detect(self, data):
        vol = np.asarray(data, dtype=np.float64)
        if vol.ndim != 3:
            raise ValueError("volume must be a 3D array")
        pad_mode = str(self.params.padding_mode).lower()
        if pad_mode not in {"nearest", "reflect", "constant", "mirror", "wrap"}:
            raise ValueError(
                "padding_mode must be one of: nearest, reflect, constant, mirror, wrap"
            )

        spacing = _normalize_spacing(self.params.spacing)
        grad_sigma = _sigma_per_axis(self.params.gradient_sigma, spacing)
        ix = (
            gaussian_filter(
                vol, sigma=tuple(grad_sigma), order=cast(Any, (0, 0, 1)), mode=pad_mode
            )
            / spacing[0]
        )
        iy = (
            gaussian_filter(
                vol, sigma=tuple(grad_sigma), order=cast(Any, (0, 1, 0)), mode=pad_mode
            )
            / spacing[1]
        )
        iz = (
            gaussian_filter(
                vol, sigma=tuple(grad_sigma), order=cast(Any, (1, 0, 0)), mode=pad_mode
            )
            / spacing[2]
        )

        tensor_sigma = _sigma_per_axis(self.params.tensor_sigma, spacing)
        jxx = gaussian_filter(ix * ix, sigma=tuple(tensor_sigma), mode=pad_mode)
        jyy = gaussian_filter(iy * iy, sigma=tuple(tensor_sigma), mode=pad_mode)
        jzz = gaussian_filter(iz * iz, sigma=tuple(tensor_sigma), mode=pad_mode)
        jxy = gaussian_filter(ix * iy, sigma=tuple(tensor_sigma), mode=pad_mode)
        jxz = gaussian_filter(ix * iz, sigma=tuple(tensor_sigma), mode=pad_mode)
        jyz = gaussian_filter(iy * iz, sigma=tuple(tensor_sigma), mode=pad_mode)

        trace = jxx + jyy + jzz
        det = (
            jxx * (jyy * jzz - jyz * jyz)
            - jxy * (jxy * jzz - jyz * jxz)
            + jxz * (jxy * jyz - jyy * jxz)
        )
        response = det - self.params.k * (trace**3)
        self.last_response = response

        response_mode = str(self.params.response_mode).lower()
        if response_mode == "shi_tomasi":
            # Minimum eigenvalue of the 3×3 structure tensor.
            # Eigenvalues are rotation-invariant by construction, so this
            # criterion detects corners regardless of orientation — the
            # staircase produced by voxel-grid rotation no longer inflates
            # a penalty term and suppresses all responses.  No 'k' parameter
            # is needed; the response is always ≥ 0 (J is PSD).
            sh = jxx.shape
            J = np.empty(sh + (3, 3), dtype=np.float64)
            J[..., 0, 0] = jxx
            J[..., 1, 1] = jyy
            J[..., 2, 2] = jzz
            J[..., 0, 1] = J[..., 1, 0] = jxy
            J[..., 0, 2] = J[..., 2, 0] = jxz
            J[..., 1, 2] = J[..., 2, 1] = jyz
            eigs = np.linalg.eigvalsh(J)  # ascending order, shape (*sh, 3)
            score = eigs[..., 0]  # minimum eigenvalue
        elif response_mode == "positive":
            score = response
        elif response_mode == "negative":
            score = -response
        elif response_mode == "absolute":
            score = np.abs(response)
        else:
            raise ValueError(
                "response_mode must be one of: shi_tomasi, positive, negative, absolute"
            )

        nms_window = int(self.params.nms_window)
        if nms_window < 1 or nms_window % 2 == 0:
            raise ValueError("nms_window must be a positive odd integer")

        local_max = maximum_filter(score, size=nms_window, mode=pad_mode)
        maxima = score == local_max
        threshold = (
            float(self.params.threshold_rel) * float(score.max())
            if score.size > 0
            else np.inf
        )
        maxima &= score > threshold

        border = int(self.params.border)
        if border > 0:
            maxima[:border, :, :] = False
            maxima[-border:, :, :] = False
            maxima[:, :border, :] = False
            maxima[:, -border:, :] = False
            maxima[:, :, :border] = False
            maxima[:, :, -border:] = False

        coords_zyx = np.argwhere(maxima)
        if coords_zyx.size == 0:
            return np.empty((0, 3), dtype=np.int32)

        scores = score[coords_zyx[:, 0], coords_zyx[:, 1], coords_zyx[:, 2]]

        # --- Absolute response floor ---
        # Removes weak spurious candidates (e.g. isolated noise voxels) that
        # survive the relative threshold only because the overall response
        # maximum is itself small.  Applied before NMS so the greedy pass
        # only considers geometrically meaningful candidates.
        abs_floor = float(self.params.threshold_abs)
        if abs_floor > 0.0:
            valid = scores >= abs_floor
            coords_zyx = coords_zyx[valid]
            scores = scores[valid]
            if coords_zyx.size == 0:
                return np.empty((0, 3), dtype=np.int32)

        order = np.argsort(scores)[::-1]
        coords_zyx = coords_zyx[order]
        scores = scores[order]

        radius = nms_window // 2
        kept = []
        for p in coords_zyx:
            if all(np.max(np.abs(p - q)) > radius for q in kept):
                kept.append(p)
        if not kept:
            return np.empty((0, 3), dtype=np.int32)
        coords_zyx = np.asarray(kept, dtype=np.int32)

        # --- Hard count cap ---
        # Applied after NMS so only the highest-scoring keypoints are kept.
        max_kp = int(self.params.max_keypoints)
        if max_kp > 0 and coords_zyx.shape[0] > max_kp:
            coords_zyx = coords_zyx[:max_kp]

        bins = self.params.balanced_bins
        if any(b > 1 for b in bins):
            coords_zyx = self._balanced_select(coords_zyx, score, bins)

        coords_xyz = np.stack(
            [coords_zyx[:, 2], coords_zyx[:, 1], coords_zyx[:, 0]], axis=1
        )
        return coords_xyz.astype(np.int32)

    def _balanced_select(self, coords_zyx, score, bins):
        bx, by, bz = bins
        if bx <= 1 and by <= 1 and bz <= 1:
            return coords_zyx

        z = coords_zyx[:, 0]
        y = coords_zyx[:, 1]
        x = coords_zyx[:, 2]

        # Coordinates are already sorted by descending score.
        x_edges = np.linspace(x.min(), x.max() + 1e-6, bx + 1)
        y_edges = np.linspace(y.min(), y.max() + 1e-6, by + 1)
        z_edges = np.linspace(z.min(), z.max() + 1e-6, bz + 1)

        picked = []
        used = set()

        for ix in range(bx):
            x_low = x_edges[ix]
            x_high = x_edges[ix + 1]
            x_mask = (x >= x_low) & (x < x_high)
            for iy in range(by):
                y_low = y_edges[iy]
                y_high = y_edges[iy + 1]
                y_mask = (y >= y_low) & (y < y_high)
                for iz in range(bz):
                    z_low = z_edges[iz]
                    z_high = z_edges[iz + 1]
                    z_mask = (z >= z_low) & (z < z_high)
                    idx = np.where(x_mask & y_mask & z_mask)[0]
                    if idx.size > 0:
                        first = int(idx[0])
                        picked.append(coords_zyx[first])
                        used.add(first)

        if picked:
            remainder = [i for i in range(coords_zyx.shape[0]) if i not in used]
            merged = np.vstack(
                [np.asarray(picked, dtype=np.int32), coords_zyx[remainder]]
            )
            return merged

        return coords_zyx
