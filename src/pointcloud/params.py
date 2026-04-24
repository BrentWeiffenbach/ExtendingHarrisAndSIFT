from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.voxel.params import SIFT3DParams


@dataclass
class SIFTRadiiPCParams:
    num_octaves: int = 3
    radii: list[float] = field(default_factory=lambda: [0.05, 0.1, 0.2, 0.4])
    fps_ratio: float = 0.5
    contrast_threshold: float = 0.42
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
class SIFTVoxelPCParams:
    voxel_size: float = 0.05
    sift3d: SIFT3DParams = field(default_factory=SIFT3DParams)
