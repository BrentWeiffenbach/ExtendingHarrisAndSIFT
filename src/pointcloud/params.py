from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.voxel.params import SIFT3DParams


@dataclass
class SIFTRadiiPCParams:
    num_octaves: int = 4
    scales_per_octave: int = 5
    base_radius: float = 5
    radius_growth_factor: float = 2.0
    fps_ratio: float = 0.5
    contrast_threshold: float = 0.001
    nms_radius_factor: float = 1.0
    max_scale_offset: float = 1.0
    min_points_per_octave: int = 10
    min_neighbors: int = 3


@dataclass
class SIFTRadiiPCResult:
    points_per_octave: list[np.ndarray]
    density_pyramid: list[list[np.ndarray]]
    radii_pyramid: list[list[float]]
    dog_pyramid: list[list[np.ndarray]]
    keypoints: np.ndarray


@dataclass
class SIFTVoxelPCParams:
    voxel_size: float = 0.05
    sift3d: SIFT3DParams = field(default_factory=SIFT3DParams)


@dataclass
class SIFTGeomPCParams:
    """Parameters for geometry-based scale-space SIFT on point clouds.

    Uses the smallest eigenvalue of the local Gaussian-weighted covariance matrix
    (scale-normalised by r²) as the scalar field instead of KDE density.  This
    signal is ~0 on flat surfaces and large at corners/edges, giving non-trivial
    DoG responses even on uniformly-sampled synthetic shapes.
    """

    num_octaves: int = 4
    scales_per_octave: int = 5
    base_radius: float = 0.05
    radius_growth_factor: float = 2.0
    fps_ratio: float = 0.5
    # Eigenvalue DoG values are small; use a tight threshold relative to the signal
    contrast_threshold: float = 1e-4
    nms_radius_factor: float = 1.0
    max_scale_offset: float = 1.0
    min_points_per_octave: int = 10
    # Need ≥4 points for a meaningful 3-D covariance estimate
    min_neighbors: int = 4
