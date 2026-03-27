from src.common.base_detector import Detector3D
import numpy as np

class SIFT3DVoxel(Detector3D):
    def __init__(self, params):
        super().__init__(params)

    def detect(self, volume):
        # 1. Build Gaussian pyramid
        # 2. Compute DoG volumes
        # 3. Detect extrema
        # 4. (Optional) refine location
        return keypoints  # Nx3 array
