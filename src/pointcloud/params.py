from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.voxel.params import SIFT3DParams


@dataclass
class SIFTRadiiPCParams:
    num_octaves: int = 3
    radii: list[float] = field(default_factory=lambda: [0.05, 0.1, 0.2, 0.4])
    fps_ratio: float = 0.5
    contrast_threshold: float = 0.45
    nms_radius_factor: float = 1.0
    max_scale_offset: float = 1.0
    min_points_per_octave: int = 10
    min_neighbors: int = 3


@dataclass
class SIFTRadiiPCResult:
    points_per_octave: list[np.ndarray]
    smoothed_pyramid: list[list[np.ndarray]]
    radii_pyramid: list[list[float]]
    dog_pyramid: list[list[np.ndarray]]
    keypoints: np.ndarray


@dataclass
class SIFTVoxelPCParams(SIFT3DParams):
    voxel_size: float = 0.02


@dataclass
class HarrisPCParams:
    k: float = 0.02
    neighborhood: str = "knn"
    k_neighbors: int = 250
    radius: float = 0.2
    threshold_rel: float = 0.02
    response_mode: str = "positive"
    nms_radius: float = 0.0
    max_keypoints: int = 500
    min_surface_variation: float = 0.108
    balanced_bins: tuple[int, int, int] = (1, 1, 1)


def default_harris_pc_params() -> HarrisPCParams:
    return HarrisPCParams()
