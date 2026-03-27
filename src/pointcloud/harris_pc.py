from src.common.base_detector import Detector3D
import numpy as np

class HarrisPC(Detector3D):
    def __init__(self, params):
        super().__init__(params)

    def detect(self, points):
        # 1. Build neighborhood graph (k-NN or radius)
        # 2. Compute local covariance tensors
        # 3. Compute cornerness from eigenvalues
        # 4. NMS in point domain
        return keypoints  # Nx3 array
