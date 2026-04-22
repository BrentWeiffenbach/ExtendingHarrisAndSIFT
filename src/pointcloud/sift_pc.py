from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree
import open3d as o3d

from src.common.base_detector import Detector3D
from src.pointcloud.params import SIFTGeomPCParams, SIFTRadiiPCParams, SIFTRadiiPCResult, SIFTVoxelPCParams
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
                density_pyramid=[],
                radii_pyramid=[],
                dog_pyramid=[],
                keypoints=np.empty((0, 5), dtype=np.float32),
            )

        density_pyramid, radii_pyramid, points_per_octave = self.build_scale_space(pts)
        dog_pyramid, dog_radius_pairs = self.compute_dog(density_pyramid, radii_pyramid)
        extrema_by_octave = self.detect_extrema(dog_pyramid, dog_radius_pairs, points_per_octave)

        all_kp = [e for e in extrema_by_octave if e.shape[0] > 0]
        keypoints = np.vstack(all_kp).astype(np.float32) if all_kp else np.empty((0, 5), dtype=np.float32)

        return SIFTRadiiPCResult(
            points_per_octave=points_per_octave,
            density_pyramid=density_pyramid,
            radii_pyramid=radii_pyramid,
            dog_pyramid=dog_pyramid,
            keypoints=keypoints,
        )

    def build_scale_space(
        self, points: np.ndarray
    ) -> tuple[list[list[np.ndarray]], list[list[float]], list[np.ndarray]]:
        density_pyramid: list[list[np.ndarray]] = []
        radii_pyramid: list[list[float]] = []
        points_per_octave: list[np.ndarray] = []

        current_pts = points.astype(np.float32)

        for octave in range(self.params.num_octaves):
            if len(current_pts) < self.params.min_points_per_octave:
                break

            tree = KDTree(current_pts.astype(np.float64))
            octave_densities: list[np.ndarray] = []
            octave_radii: list[float] = []

            for scale in range(self.params.scales_per_octave):
                r = (
                    self.params.base_radius
                    * (self.params.radius_growth_factor ** octave)
                    * (2.0 ** (scale / self.params.scales_per_octave))
                )
                density = self._compute_density(current_pts, tree, r)
                octave_densities.append(density)
                octave_radii.append(float(r))

            density_pyramid.append(octave_densities)
            radii_pyramid.append(octave_radii)
            points_per_octave.append(current_pts)

            n_next = max(
                self.params.min_points_per_octave,
                int(len(current_pts) * self.params.fps_ratio),
            )
            current_pts = self.farthest_point_sample(current_pts, n_next)

        return density_pyramid, radii_pyramid, points_per_octave

    def compute_dog(
        self,
        density_pyramid: list[list[np.ndarray]],
        radii_pyramid: list[list[float]],
    ) -> tuple[list[list[np.ndarray]], list[list[tuple[float, float]]]]:
        dog_pyramid: list[list[np.ndarray]] = []
        dog_radius_pairs: list[list[tuple[float, float]]] = []

        for octave_idx, octave_densities in enumerate(density_pyramid):
            if len(octave_densities) < 2:
                dog_pyramid.append([])
                dog_radius_pairs.append([])
                continue

            octave_dogs: list[np.ndarray] = []
            octave_pairs: list[tuple[float, float]] = []
            octave_radii = radii_pyramid[octave_idx]

            for i in range(len(octave_densities) - 1):
                dog = (octave_densities[i + 1] - octave_densities[i]).astype(np.float32)
                octave_dogs.append(dog)
                octave_pairs.append((octave_radii[i], octave_radii[i + 1]))

            dog_pyramid.append(octave_dogs)
            dog_radius_pairs.append(octave_pairs)

        return dog_pyramid, dog_radius_pairs

    def detect_extrema(
        self,
        dog_pyramid: list[list[np.ndarray]],
        dog_radius_pairs: list[list[tuple[float, float]]],
        points_per_octave: list[np.ndarray],
    ) -> list[np.ndarray]:
        threshold = float(self.params.contrast_threshold)
        extrema_by_octave: list[np.ndarray] = []

        for octave_idx, octave_dogs in enumerate(dog_pyramid):
            pts = points_per_octave[octave_idx]
            pairs = dog_radius_pairs[octave_idx]

            if len(octave_dogs) < 3:
                extrema_by_octave.append(np.empty((0, 5), dtype=np.float32))
                continue

            tree = KDTree(pts.astype(np.float64))
            candidates: list[tuple[float, float, float, float, float]] = []

            for dog_idx in range(1, len(octave_dogs) - 1):
                prev_dog = octave_dogs[dog_idx - 1]
                curr_dog = octave_dogs[dog_idx]
                next_dog = octave_dogs[dog_idx + 1]

                r_low, r_high = pairs[dog_idx]
                r_char = float(np.sqrt(r_low * r_high))
                nms_r = r_low * self.params.nms_radius_factor

                # Vectorized pre-filter: threshold and scale-axis extremum check
                mask = (np.abs(curr_dog) >= threshold) & (
                    (curr_dog > np.maximum(prev_dog, next_dog))
                    | (curr_dog < np.minimum(prev_dog, next_dog))
                )
                candidate_indices = np.where(mask)[0]

                for i in candidate_indices.tolist():
                    val = float(curr_dog[i])
                    is_max = val > max(float(prev_dog[i]), float(next_dog[i]))

                    # Spatial NMS
                    nbr_indices = tree.query_ball_point(pts[i].astype(np.float64), r=nms_r)
                    nbr_indices_excl = [j for j in nbr_indices if j != i]

                    if nbr_indices_excl:
                        nbr_vals = curr_dog[nbr_indices_excl]
                        if is_max and val <= float(np.max(nbr_vals)):
                            continue
                        if not is_max and val >= float(np.min(nbr_vals)):
                            continue

                    # Scale refinement
                    offset, refined_response = self._refine_scale(
                        float(prev_dog[i]), val, float(next_dog[i]), threshold
                    )
                    if offset is None:
                        continue

                    r_refined = r_char * (2.0 ** offset)
                    candidates.append((
                        float(pts[i, 0]),
                        float(pts[i, 1]),
                        float(pts[i, 2]),
                        float(r_refined),
                        float(refined_response),
                    ))

            if candidates:
                extrema_by_octave.append(np.array(candidates, dtype=np.float32))
            else:
                extrema_by_octave.append(np.empty((0, 5), dtype=np.float32))

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

    def _compute_density(
        self, points: np.ndarray, tree: KDTree, radius: float
    ) -> np.ndarray:
        density = np.zeros(len(points), dtype=np.float32)
        two_r2 = 2.0 * radius * radius
        neighbor_lists = tree.query_ball_point(points.astype(np.float64), r=radius, workers=-1)

        for i, nbrs in enumerate(neighbor_lists):
            if len(nbrs) < self.params.min_neighbors:
                density[i] = 0.0
                continue
            diffs = points[nbrs].astype(np.float64) - points[i].astype(np.float64)
            sq_dists = np.einsum("ij,ij->i", diffs, diffs)
            density[i] = float(np.sum(np.exp(-sq_dists / two_r2)))

        return density

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
        kp_xyz_voxel = SIFT3DVoxel(self.params.sift3d).detect(volume)

        if kp_xyz_voxel.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float32)

        return self._to_physical(kp_xyz_voxel, min_corner)

    def run(self, points: np.ndarray) -> dict:
        pts = self._validate_input(points)
        volume, min_corner = self._voxelize(pts)
        detector = SIFT3DVoxel(self.params.sift3d)
        result = detector.run(volume)
        kp_xyz_voxel = (
            result.extrema_global[:, [2, 1, 0]].astype(np.float32)
            if result.extrema_global.shape[0] > 0
            else np.empty((0, 3), dtype=np.float32)
        )
        keypoints_physical = self._to_physical(kp_xyz_voxel, min_corner) if kp_xyz_voxel.shape[0] > 0 else np.empty((0, 3), dtype=np.float32)
        return {
            "volume": volume,
            "min_corner": min_corner,
            "sift3d_result": result,
            "keypoints": keypoints_physical,
        }

    def _voxelize(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        min_corner = points.min(axis=0)
        max_corner = points.max(axis=0)

        shape_xyz = np.ceil(
            (max_corner - min_corner) / self.params.voxel_size
        ).astype(int).clip(4, None)

        # Volume stored as (z, y, x) = (D, H, W)
        volume = np.zeros((shape_xyz[2], shape_xyz[1], shape_xyz[0]), dtype=np.float32)

        indices = np.floor(
            (points - min_corner) / self.params.voxel_size
        ).astype(int)
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


class SIFTGeomPC(SIFTRadiiPC):
    """SIFT on point clouds using local covariance geometry as the scale-space signal.

    Replaces Gaussian KDE density with the smallest eigenvalue of the Gaussian-weighted
    local covariance matrix, normalised by r².  On a perfectly flat surface this value
    is ~0 (no thickness); at corners / edges all three principal variances are non-zero,
    so the signal is large.  DoG of this geometric field finds multi-scale extrema of
    3-D shape complexity rather than point-density variation — meaningful even for
    uniformly-sampled synthetic meshes.
    """

    def __init__(self, params: SIFTGeomPCParams | None = None):
        super().__init__(params or SIFTGeomPCParams())

    # --- override the scalar-field computation only ----------------------------

    def _compute_density(self, points: np.ndarray, tree: KDTree, radius: float) -> np.ndarray:
        return self._compute_geometry(points, tree, radius)

    def _compute_geometry(self, points: np.ndarray, tree: KDTree, radius: float) -> np.ndarray:
        """Return the scale-normalised smallest covariance eigenvalue for each point.

        For each point i the Gaussian-weighted covariance of its neighbours is
        computed and divided by r² so that the eigenvalues are dimensionless and
        comparable across octaves.  The minimum eigenvalue λ_min is ~0 on flat
        surfaces and increases as local geometry becomes more corner-like.
        """
        n = len(points)
        feature = np.zeros(n, dtype=np.float32)
        r2 = float(radius * radius)
        two_r2 = 2.0 * r2
        neighbor_lists = tree.query_ball_point(points.astype(np.float64), r=radius, workers=-1)

        for i, nbrs in enumerate(neighbor_lists):
            if len(nbrs) < self.params.min_neighbors:
                continue
            nbr_pts = points[nbrs].astype(np.float64)
            diffs = nbr_pts - points[i].astype(np.float64)   # (k, 3)
            sq_dists = np.einsum("ij,ij->i", diffs, diffs)
            weights = np.exp(-sq_dists / two_r2)
            w_sum = weights.sum()
            if w_sum < 1e-10:
                continue
            # Gaussian-weighted covariance centred at p_i, scale-normalised by r²
            w_norm = weights / w_sum
            cov = (diffs.T * w_norm) @ diffs / r2   # (3, 3)
            # eigvalsh returns eigenvalues in ascending order (λ_min first)
            eigvals = np.linalg.eigvalsh(cov)
            feature[i] = float(max(eigvals[0], 0.0))

        return feature
