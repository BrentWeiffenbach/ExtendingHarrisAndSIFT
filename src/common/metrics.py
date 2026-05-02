from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree  # type: ignore[import]


@dataclass
class MatchingResult:
    matched_reference: np.ndarray
    matched_candidate: np.ndarray
    distances: np.ndarray


def match_keypoints(
    reference: np.ndarray,
    candidate: np.ndarray,
    radius: float,
) -> MatchingResult:
    """One-to-one radius-constrained matching from reference to candidate.

    Matches are built greedily by ascending distance so each candidate keypoint
    can be used at most once.
    """
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)

    if ref.size == 0 or cand.size == 0:
        return MatchingResult(
            matched_reference=np.empty((0,), dtype=np.int32),
            matched_candidate=np.empty((0,), dtype=np.int32),
            distances=np.empty((0,), dtype=np.float64),
        )

    tree = cKDTree(cand)
    distances, indices = tree.query(ref, distance_upper_bound=float(radius))

    valid = np.isfinite(distances) & (indices < cand.shape[0])
    if not np.any(valid):
        return MatchingResult(
            matched_reference=np.empty((0,), dtype=np.int32),
            matched_candidate=np.empty((0,), dtype=np.int32),
            distances=np.empty((0,), dtype=np.float64),
        )

    ref_idx = np.where(valid)[0]
    cand_idx = indices[valid]
    dists = distances[valid]

    order = np.argsort(dists)
    used_cand: set[int] = set()
    kept_ref: list[int] = []
    kept_cand: list[int] = []
    kept_dist: list[float] = []

    for oi in order:
        c_idx = int(cand_idx[oi])
        if c_idx in used_cand:
            continue
        used_cand.add(c_idx)
        kept_ref.append(int(ref_idx[oi]))
        kept_cand.append(c_idx)
        kept_dist.append(float(dists[oi]))

    return MatchingResult(
        matched_reference=np.asarray(kept_ref, dtype=np.int32),
        matched_candidate=np.asarray(kept_cand, dtype=np.int32),
        distances=np.asarray(kept_dist, dtype=np.float64),
    )


def repeatability_score(
    num_reference: int,
    num_candidate: int,
    num_matches: int,
) -> float:
    denom = max(int(num_reference), int(num_candidate), 1)
    return float(num_matches) / float(denom)


def localization_error(distances: np.ndarray) -> float | None:
    d = np.asarray(distances, dtype=np.float64)
    if d.size == 0:
        return None
    return float(np.mean(d))


def keypoint_count_stability(counts: list[int]) -> dict:
    if not counts:
        return {
            "mean": 0.0,
            "std": 0.0,
            "cv": 0.0,
            "min": 0,
            "max": 0,
        }

    arr = np.asarray(counts, dtype=np.float64)
    mean_v = float(np.mean(arr))
    std_v = float(np.std(arr))
    cv_v = float(std_v / mean_v) if mean_v > 0 else 0.0
    return {
        "mean": mean_v,
        "std": std_v,
        "cv": cv_v,
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
    }
