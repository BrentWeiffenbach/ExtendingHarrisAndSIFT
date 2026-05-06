from dataclasses import dataclass
from typing import Tuple


@dataclass
class Harris3DParams:
    k: float = 0.02358
    gradient_sigma: float = 0.48266
    tensor_sigma: float = 0.67641
    threshold_rel: float = 0.00387
    # Absolute response floor applied *after* the relative threshold.
    # Any candidate whose score is below this value is discarded regardless
    # of how it compares to the volume maximum.  Set to 0.0 to disable.
    # Calibrated so isolated noise voxels (~2e-6) are suppressed while
    # genuine cube/pyramid/torus corners (~8e-6 and above) are kept.
    threshold_abs: float = 5e-6
    # Hard cap on the number of keypoints returned, ordered by descending
    # response score.  Prevents noise explosions from flooding downstream
    # consumers.  Set to 0 to disable.
    max_keypoints: int = 50
    response_mode: str = "positive"
    nms_window: int = 5
    border: int = 0
    padding_mode: str = "constant"
    balanced_bins: Tuple[int, int, int] = (1, 1, 1)
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)


def default_harris3d_params() -> Harris3DParams:
    return Harris3DParams()


@dataclass
class SIFT2DParams:
    num_octaves: int = 4
    scales_per_octave: int = 5
    base_sigma: float = 1.6
    contrast_threshold: float = 0.03
    border: int = 1
    refinement_offset_threshold: float = 1.6
    refinement_singular_eps: float = 1e-10
    orientation_bins: int = 36
    orientation_window_factor: float = 3.0
    orientation_weight_sigma_factor: float = 1.5
    orientation_visual_keypoint_index: int = 0
    max_plot_points: int = 4000


@dataclass
class SIFT3DParams:
    num_octaves: int = 3
    scales_per_octave: int = 8
    base_sigma: float = 1
    min_size: int = 8
    downsample_factor: int = 2
    downsample_sigma: float = 1
    border_mode: str = "reflect"
    slice_axis: int = 0
    extrema_contrast_threshold: float = 0.2
    extrema_border: int = 2
    refinement_offset_threshold: float = 1.0
    refinement_singular_eps: float = 1e-10
    blob_radius_factor: float = 1.0
    max_blob_keypoints: int = 2000
    suppression_min_dist_factor: float = 1.5
