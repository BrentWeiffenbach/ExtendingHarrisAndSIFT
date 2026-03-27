from src.common.base_detector import Detector3D
import numpy as np

class SIFTPC(Detector3D):
    def __init__(self, params):
        super().__init__(params)

    def detect(self, points):
        # 1. Build pseudo-scale-space (radius increasing, voxelization, or diffusion)
        # 2. Compute DoG-like operator
        # 3. Detect multi-scale extrema
        return keypoints  # Nx3 array
