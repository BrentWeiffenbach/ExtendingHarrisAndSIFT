from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from src.common.base_detector import Detector3D


def _to_xyz(points) -> np.ndarray:
    if not isinstance(points, np.ndarray) and hasattr(points, "points"):
        pts = np.asarray(points.points)
    else:
        pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("points must have shape (N, 3)")
    return pts[:, :3].astype(np.float64)


class HarrisPC(Detector3D):
    def __init__(self, params):
        super().__init__(params)
        self.last_response: np.ndarray | None = None

    def detect(self, data) -> np.ndarray:
        pts = _to_xyz(data)
        if pts.shape[0] < 4:
            self.last_response = np.zeros(pts.shape[0])
            return np.empty((0, 3), dtype=np.float64)

        tree = cKDTree(pts)

        # --- 1. Build neighborhoods and covariance tensors ---
        mode = str(self.params.neighborhood).lower()
        if mode == "knn":
            covs = self._covariance_knn(pts, tree)
        elif mode == "radius":
            covs = self._covariance_radius(pts, tree)
        else:
            raise ValueError("neighborhood must be 'knn' or 'radius'")

        # --- 2. Cornerness response: det(C) - k * trace(C)^3 ---
        trace = covs[:, 0, 0] + covs[:, 1, 1] + covs[:, 2, 2]
        det = np.linalg.det(covs)
        response = det - self.params.k * (trace**3)
        self.last_response = response

        # --- 2b. Surface variation filter (edge suppression) ---
        # sv = λ_min / (λ1 + λ2 + λ3); high at corners, low on edges/faces
        eigvals = np.linalg.eigvalsh(covs)  # sorted ascending
        surface_variation = eigvals[:, 0] / (trace + 1e-30)
        sv_threshold = float(self.params.min_surface_variation)

        # --- 3. Response mode + threshold ---
        response_mode = str(self.params.response_mode).lower()
        if response_mode == "positive":
            score = response
        elif response_mode == "negative":
            score = -response
        elif response_mode == "absolute":
            score = np.abs(response)
        else:
            raise ValueError(
                "response_mode must be one of: positive, negative, absolute"
            )

        max_score = score.max() if score.size > 0 else 0.0
        threshold = float(self.params.threshold_rel) * max_score
        mask = (score > threshold) & (surface_variation >= sv_threshold)
        candidate_idx = np.where(mask)[0]

        if candidate_idx.size == 0:
            return np.empty((0, 3), dtype=np.float64)

        # --- 4. NMS: greedy Euclidean-radius suppression ---
        nms_r = float(self.params.nms_radius)
        if nms_r <= 0:
            nms_r = self._auto_nms_radius(pts, tree)

        kept_idx = self._nms(pts, score, candidate_idx, tree, nms_r)

        # cap
        max_kp = int(self.params.max_keypoints)
        if len(kept_idx) > max_kp:
            kept_idx = kept_idx[:max_kp]

        kps = pts[kept_idx]

        # --- 5. Optional balanced spatial selection ---
        bins = self.params.balanced_bins
        if any(b > 1 for b in bins):
            kps = self._balanced_select(kps, score[kept_idx], bins)

        return kps

    # ------------------------------------------------------------------
    # Neighborhood covariance
    # ------------------------------------------------------------------

    def _covariance_knn(self, pts: np.ndarray, tree: cKDTree) -> np.ndarray:
        k = int(self.params.k_neighbors)
        _, indices = tree.query(pts, k=k + 1)
        # drop self (column 0)
        nbr_idx = indices[:, 1:]  # (N, k)
        nbrs = pts[nbr_idx]  # (N, k, 3)
        centroid = nbrs.mean(axis=1, keepdims=True)  # (N, 1, 3)
        diff = nbrs - centroid  # (N, k, 3)
        covs = np.einsum("nkx,nky->nxy", diff, diff) / k  # (N, 3, 3)
        return covs

    def _covariance_radius(self, pts: np.ndarray, tree: cKDTree) -> np.ndarray:
        r = float(self.params.radius)
        neighbors_list = tree.query_ball_point(pts, r)
        n = pts.shape[0]
        covs = np.zeros((n, 3, 3), dtype=np.float64)
        for i, nbrs_i in enumerate(neighbors_list):
            # remove self
            nbrs_i = [j for j in nbrs_i if j != i]
            if len(nbrs_i) < 3:
                continue
            nbr_pts = pts[nbrs_i]
            centroid = nbr_pts.mean(axis=0)
            diff = nbr_pts - centroid
            covs[i] = (diff.T @ diff) / len(nbrs_i)
        return covs

    # ------------------------------------------------------------------
    # NMS
    # ------------------------------------------------------------------

    @staticmethod
    def _auto_nms_radius(pts: np.ndarray, tree: cKDTree) -> float:
        bbox_diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        return bbox_diag * 0.3

    @staticmethod
    def _nms(
        pts: np.ndarray,
        score: np.ndarray,
        candidate_idx: np.ndarray,
        tree: cKDTree,
        nms_radius: float,
    ) -> np.ndarray:
        # sort candidates by descending score
        order = np.argsort(score[candidate_idx])[::-1]
        sorted_idx = candidate_idx[order]

        suppressed = set()
        kept: list[int] = []
        for idx in sorted_idx:
            if idx in suppressed:
                continue
            kept.append(int(idx))
            # suppress all neighbors within nms_radius
            nbrs = tree.query_ball_point(pts[idx], nms_radius)
            suppressed.update(nbrs)

        return np.array(kept, dtype=np.intp)

    # ------------------------------------------------------------------
    # Balanced spatial selection (mirrors voxel pattern)
    # ------------------------------------------------------------------

    @staticmethod
    def _balanced_select(
        kps: np.ndarray, scores: np.ndarray, bins: tuple
    ) -> np.ndarray:
        bx, by, bz = bins
        if bx <= 1 and by <= 1 and bz <= 1:
            return kps

        x, y, z = kps[:, 0], kps[:, 1], kps[:, 2]
        x_edges = np.linspace(x.min(), x.max() + 1e-9, bx + 1)
        y_edges = np.linspace(y.min(), y.max() + 1e-9, by + 1)
        z_edges = np.linspace(z.min(), z.max() + 1e-9, bz + 1)

        # kps are already sorted by decreasing score from NMS
        picked: list[int] = []
        used: set[int] = set()

        for ix in range(bx):
            x_mask = (x >= x_edges[ix]) & (x < x_edges[ix + 1])
            for iy in range(by):
                y_mask = (y >= y_edges[iy]) & (y < y_edges[iy + 1])
                for iz in range(bz):
                    z_mask = (z >= z_edges[iz]) & (z < z_edges[iz + 1])
                    idx = np.where(x_mask & y_mask & z_mask)[0]
                    if idx.size > 0:
                        first = int(idx[0])
                        if first not in used:
                            picked.append(first)
                            used.add(first)

        if picked:
            remainder = [i for i in range(kps.shape[0]) if i not in used]
            merged = np.vstack([kps[picked], kps[remainder]])
            return merged

        return kps
