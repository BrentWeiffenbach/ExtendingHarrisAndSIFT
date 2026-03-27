class Detector3D:
    def __init__(self, params):
        self.params = params

    def detect(self, data):
        """
        Input:
            data - either a voxel volume (3D array) or point cloud (Nx3 array)
        Output:
            keypoints - list of coordinates (x, y, z) and optional scale/orientation
        """
        raise NotImplementedError
