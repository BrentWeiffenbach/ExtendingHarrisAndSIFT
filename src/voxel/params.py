from dataclasses import dataclass
from typing import Tuple


@dataclass
class Harris3DParams:
    k: float = 0.02358
    gradient_sigma: float = 0.48266
    tensor_sigma: float = 0.67641
    threshold_rel: float = 0.00387
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
    num_octaves: int = 4
    scales_per_octave: int = 5
    base_sigma: float = 1.6
    min_size: int = 8
    downsample_factor: int = 2
    downsample_sigma: float = 1.0
    border_mode: str = "reflect"
    slice_axis: int = 0
    extrema_contrast_threshold: float = 0.01
    extrema_border: int = 1
    refinement_offset_threshold: float = 1.0
    refinement_singular_eps: float = 1e-10
    blob_radius_factor: float = 1.0
    max_blob_keypoints: int = 2000
