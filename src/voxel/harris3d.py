from src.common.base_detector import Detector3D
import numpy as np

class Harris3DVoxel(Detector3D):
    def __init__(self, params):
        super().__init__(params)

    def detect(self, volume):
        # 1. Compute gradients
        # 2. Build structure tensor
        # 3. Compute cornerness
        # 4. Apply 3D NMS
        # Return list of keypoints
        return keypoints  # Nx3 array
