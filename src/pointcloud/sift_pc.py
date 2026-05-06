from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree as KDTree
import open3d as o3d

from src.common.base_detector import Detector3D
from src.pointcloud.params import (
    SIFTRadiiPCParams,
    SIFTRadiiPCResult,
    SIFTVoxelPCParams,
)
from src.voxel.sift3d import SIFT3DVoxel


class SIFTRadiiPC(Detector3D):
    def __init__(self, params: SIFTRadiiPCParams | None = None):
        super().__init__(params or SIFTRadiiPCParams())

    def detect(self, points: np.ndarray) -> np.ndarray:
        result = self.run(points)
        if result.keypoints.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float32)
        return result.keypoints[:, :3].astype(np.float32)

    def run(self, points: np.ndarray) -> SIFTRadiiPCResult:
        pts = self._validate_input(points)
        if len(pts) < self.params.min_points_per_octave:
            return SIFTRadiiPCResult(
                points_per_octave=[],
                smoothed_pyramid=[],
                radii_pyramid=[],
                dog_pyramid=[],
                keypoints=np.empty((0, 5), dtype=np.float32),
            )

        smoothed_pyramid, radii_pyramid, points_per_octave = self.build_scale_space(pts)
        dog_pyramid, dog_radii = self.compute_dog(
            smoothed_pyramid, radii_pyramid, points_per_octave
        )
        extrema_by_octave = self.detect_extrema(
            dog_pyramid, dog_radii, points_per_octave
        )

        all_kp = [e for e in extrema_by_octave if e.shape[0] > 0]
        keypoints = (
            np.vstack(all_kp).astype(np.float32)
            if all_kp
            else np.empty((0, 5), dtype=np.float32)
        )

        return SIFTRadiiPCResult(
            points_per_octave=points_per_octave,
            smoothed_pyramid=smoothed_pyramid,
            radii_pyramid=radii_pyramid,
            dog_pyramid=dog_pyramid,
            keypoints=keypoints,
        )

    def build_scale_space(
        self, points: np.ndarray
    ) -> tuple[list[list[np.ndarray]], list[list[float]], list[np.ndarray]]:
        smoothed_pyramid: list[list[np.ndarray]] = []
        radii_pyramid: list[list[float]] = []
        points_per_octave: list[np.ndarray] = []

        current_pts = points.astype(np.float32)

        for octave in range(self.params.num_octaves):
            if len(current_pts) < self.params.min_points_per_octave:
                break

            tree = KDTree(current_pts.astype(np.float64))
            octave_smoothed: list[np.ndarray] = []
            octave_radii: list[float] = []
            octave_scale = 2.0**octave

            for r in self.params.radii:
                r_scaled = float(r) * octave_scale
                smoothed = self._compute_smoothed_positions(current_pts, tree, r_scaled)
                octave_smoothed.append(smoothed)
                octave_radii.append(r_scaled)

            smoothed_pyramid.append(octave_smoothed)
            radii_pyramid.append(octave_radii)
            points_per_octave.append(current_pts)

            n_next = max(
                self.params.min_points_per_octave,
                int(len(current_pts) * self.params.fps_ratio),
            )
            current_pts = self.farthest_point_sample(current_pts, n_next)

        return smoothed_pyramid, radii_pyramid, points_per_octave

    def compute_dog(
        self,
        smoothed_pyramid: list[list[np.ndarray]],
        radii_pyramid: list[list[float]],
        points_per_octave: list[np.ndarray],
    ) -> tuple[list[list[np.ndarray]], list[list[float]]]:
        """DoG as Euclidean drift between consecutive smoothed-position levels.

        dog[s] = ||smoothed[s+1] - smoothed[s]|| — how much each point's smoothed
        position shifts when the radius increases to the next scale.  Large at
        corners/edges (position keeps changing with scale) and small on flat surfaces.
        """
        dog_pyramid: list[list[np.ndarray]] = []
        dog_radii: list[list[float]] = []

        for octave_idx, octave_smoothed in enumerate(smoothed_pyramid):
            octave_radii = radii_pyramid[octave_idx]
            octave_dogs: list[np.ndarray] = []
            octave_r: list[float] = []

            for i in range(len(octave_smoothed) - 1):
                r_char = float(np.sqrt(octave_radii[i] * octave_radii[i + 1]))
                diff = octave_smoothed[i + 1] - octave_smoothed[i]  # (N, 3)
                # Normalise by r_char → dimensionless "drift per unit radius",
                # so contrast_threshold is scale-invariant.
                dog = (np.linalg.norm(diff, axis=1) / r_char).astype(np.float32)
                octave_dogs.append(dog)
                octave_r.append(r_char)

            dog_pyramid.append(octave_dogs)
            dog_radii.append(octave_r)

        return dog_pyramid, dog_radii

    def detect_extrema(
        self,
        dog_pyramid: list[list[np.ndarray]],
        dog_radii: list[list[float]],
        points_per_octave: list[np.ndarray],
    ) -> list[np.ndarray]:
        """NMS detection: each candidate must be a spatial maximum and exceed the
        same point at both adjacent scale levels (analog of the 26-neighbor check
        from the SIFT paper, adapted for unordered point sets).
        """
        threshold = float(self.params.contrast_threshold)
        extrema_by_octave: list[np.ndarray] = []

        for octave_idx, octave_dogs in enumerate(dog_pyramid):
            pts = points_per_octave[octave_idx]
            radii = dog_radii[octave_idx]

            if len(octave_dogs) < 3:
                extrema_by_octave.append(np.empty((0, 5), dtype=np.float32))
                continue

            tree = KDTree(pts.astype(np.float64))
            scale_kept_xyz: list[np.ndarray] = []
            scale_kept_r: list[float] = []
            scale_kept_vals: list[np.ndarray] = []

            for scale_idx in range(1, len(octave_dogs) - 1):
                curr_dog = octave_dogs[scale_idx]
                prev_dog = octave_dogs[scale_idx - 1]
                next_dog = octave_dogs[scale_idx + 1]
                r_char = radii[scale_idx]
                nms_r = r_char * self.params.nms_radius_factor

                # Vectorized threshold + scale-axis check in one numpy pass.
                candidate_indices = np.where(
                    (curr_dog >= threshold)
                    & (curr_dog > prev_dog)
                    & (curr_dog > next_dog)
                )[0]

                if candidate_indices.size == 0:
                    continue

                # Batch spatial query for all candidates at once with threading.
                nbr_lists = tree.query_ball_point(
                    pts[candidate_indices].astype(np.float64),
                    r=nms_r,
                    workers=-1,
                )

                # Build a flat neighbour-value array so the per-candidate max can
                # be computed with np.maximum.reduceat instead of a Python loop.
                flat_nbrs: list[int] = []
                lengths = np.empty(len(candidate_indices), dtype=np.intp)
                for k, (ci, nbrs) in enumerate(
                    zip(candidate_indices.tolist(), nbr_lists)
                ):
                    filtered = [j for j in nbrs if j != ci]
                    flat_nbrs.extend(filtered)
                    lengths[k] = len(filtered)

                has_nbrs = lengths > 0
                if has_nbrs.any() and flat_nbrs:
                    flat_vals = curr_dog[np.array(flat_nbrs, dtype=np.intp)]
                    nonempty_lengths = lengths[has_nbrs]
                    # section_starts[i] is the index in flat_vals where group i begins
                    section_starts = np.empty(len(nonempty_lengths), dtype=np.intp)
                    section_starts[0] = 0
                    np.cumsum(nonempty_lengths[:-1], out=section_starts[1:])
                    max_nbr = np.full(len(candidate_indices), -np.inf, dtype=np.float32)
                    max_nbr[has_nbrs] = np.maximum.reduceat(flat_vals, section_starts)
                else:
                    max_nbr = np.full(len(candidate_indices), -np.inf, dtype=np.float32)

                keep_mask = curr_dog[candidate_indices] > max_nbr
                kept = candidate_indices[keep_mask]
                if kept.size > 0:
                    scale_kept_xyz.append(pts[kept])
                    scale_kept_r.append(r_char)
                    scale_kept_vals.append(curr_dog[kept])

            if not scale_kept_xyz:
                extrema_by_octave.append(np.empty((0, 5), dtype=np.float32))
                continue

            # Assemble candidates array from per-scale results.
            parts = [
                np.hstack(
                    [
                        xyz,
                        np.full((len(xyz), 1), r, dtype=np.float32),
                        vals[:, None].astype(np.float32),
                    ]
                )
                for xyz, r, vals in zip(scale_kept_xyz, scale_kept_r, scale_kept_vals)
            ]
            cands = np.vstack(parts).astype(np.float32)

            # Deduplicate across scales: greedy NMS sorted by response descending.
            order = np.argsort(-cands[:, 4])
            cands = cands[order]
            suppressed = np.zeros(len(cands), dtype=bool)
            dedup_r = float(max(radii)) * self.params.nms_radius_factor
            kd = KDTree(cands[:, :3].astype(np.float64))
            kept_final: list[int] = []
            for k in range(len(cands)):
                if suppressed[k]:
                    continue
                kept_final.append(k)
                for nb in kd.query_ball_point(
                    cands[k, :3].astype(np.float64), r=dedup_r
                ):
                    if nb != k:
                        suppressed[nb] = True

            extrema_by_octave.append(cands[kept_final])

        return extrema_by_octave

    def farthest_point_sample(self, points: np.ndarray, n: int) -> np.ndarray:
        if n >= len(points):
            return points.copy()
        if n <= 0:
            return np.empty((0, 3), dtype=np.float32)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        down = pcd.farthest_point_down_sample(n)
        return np.asarray(down.points, dtype=np.float32)

    def _compute_smoothed_positions(
        self, points: np.ndarray, tree: KDTree, radius: float
    ) -> np.ndarray:
        """Return Gaussian-weighted average neighbour positions for each point.

        Each point is moved to the weighted centroid of its neighbours within
        `radius`.  Points with fewer than min_neighbors neighbours keep their
        original position.  Returns (N, 3) float32.

        Processes query_ball_point in chunks of smoothing_chunk_size so that
        neighbour index lists for all N points are never in RAM at once.
        """
        n = len(points)
        pts_f64 = points.astype(np.float64)
        smoothed = pts_f64.copy()
        two_r2 = 2.0 * radius * radius
        chunk = self.params.smoothing_chunk_size

        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            nbr_lists = tree.query_ball_point(pts_f64[start:end], r=radius, workers=-1)
            for j, nbrs in enumerate(nbr_lists):
                if len(nbrs) < self.params.min_neighbors:
                    continue
                i = start + j
                nbr_pts = pts_f64[nbrs]
                diffs = nbr_pts - pts_f64[i]
                sq_dists = np.einsum("ij,ij->i", diffs, diffs)
                weights = np.exp(-sq_dists / two_r2)
                w_sum = weights.sum()
                if w_sum < 1e-10:
                    continue
                smoothed[i] = (weights[:, None] * nbr_pts).sum(axis=0) / w_sum

        return smoothed.astype(np.float32)

    def _refine_scale(
        self,
        f_prev: float,
        f_curr: float,
        f_next: float,
        threshold: float,
    ) -> tuple[float, float] | tuple[None, None]:
        denom = f_next - 2.0 * f_curr + f_prev
        if abs(denom) < 1e-10:
            return 0.0, f_curr

        offset = -0.5 * (f_next - f_prev) / denom
        if abs(offset) > self.params.max_scale_offset:
            return None, None

        refined = f_curr + 0.5 * offset * (f_next - f_prev)
        if abs(refined) < threshold:
            return None, None

        return float(offset), float(refined)

    def _validate_input(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError("points must be (N, 3)")
        pts = pts[:, :3]
        return pts[np.isfinite(pts).all(axis=1)]


class SIFTVoxelPC(Detector3D):
    def __init__(self, params: SIFTVoxelPCParams | None = None):
        super().__init__(params or SIFTVoxelPCParams())

    def detect(self, points: np.ndarray) -> np.ndarray:
        pts = self._validate_input(points)
        if len(pts) == 0:
            return np.empty((0, 3), dtype=np.float32)

        volume, min_corner = self._voxelize(pts)
        kp_xyz_voxel = SIFT3DVoxel(self.params).detect(volume)

        if kp_xyz_voxel.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float32)

        return self._to_physical(kp_xyz_voxel, min_corner)

    def run(self, points: np.ndarray) -> dict:
        pts = self._validate_input(points)
        volume, min_corner = self._voxelize(pts)
        detector = SIFT3DVoxel(self.params)
        result = detector.run(volume)
        kp_xyz_voxel = (
            result.extrema_global[:, [2, 1, 0]].astype(np.float32)
            if result.extrema_global.shape[0] > 0
            else np.empty((0, 3), dtype=np.float32)
        )
        keypoints_physical = (
            self._to_physical(kp_xyz_voxel, min_corner)
            if kp_xyz_voxel.shape[0] > 0
            else np.empty((0, 3), dtype=np.float32)
        )
        return {
            "volume": volume,
            "min_corner": min_corner,
            "sift3d_result": result,
            "keypoints": keypoints_physical,
        }

    def _voxelize(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        min_corner = points.min(axis=0)
        max_corner = points.max(axis=0)

        shape_xyz = (
            np.ceil((max_corner - min_corner) / self.params.voxel_size)
            .astype(int)
            .clip(4, None)
        )

        # Volume stored as (z, y, x) = (D, H, W)
        volume = np.zeros((shape_xyz[2], shape_xyz[1], shape_xyz[0]), dtype=np.float32)

        indices = np.floor((points - min_corner) / self.params.voxel_size).astype(int)
        # Clip to valid range
        indices[:, 0] = indices[:, 0].clip(0, shape_xyz[0] - 1)
        indices[:, 1] = indices[:, 1].clip(0, shape_xyz[1] - 1)
        indices[:, 2] = indices[:, 2].clip(0, shape_xyz[2] - 1)

        volume[indices[:, 2], indices[:, 1], indices[:, 0]] = 1.0
        return volume, min_corner.astype(np.float32)

    def _to_physical(
        self, kp_xyz_voxel: np.ndarray, min_corner: np.ndarray
    ) -> np.ndarray:
        return (min_corner + kp_xyz_voxel * self.params.voxel_size).astype(np.float32)

    def _validate_input(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError("points must be (N, 3)")
        pts = pts[:, :3]
        return pts[np.isfinite(pts).all(axis=1)]
