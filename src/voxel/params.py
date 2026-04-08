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
