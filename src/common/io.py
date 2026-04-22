"""Generic data loaders for voxels, point clouds, and images."""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import open3d as o3d

SYNTHETIC_ROOT = "data/Voxel/synthetic"
MODELNET_ROOT = "data/Voxel/real/ModelNet10-dataset"
POINTCLOUD_ROOT = "data/Pointcloud"


class SyntheticVoxelLoader:
    """Load synthetic voxel shapes (cone, cube, cuboid, cylinder, pyramid, sphere, torus)."""

    SHAPES = ["cone", "cube", "cuboid", "cylinder", "pyramid", "sphere", "torus"]

    def __init__(self, root: str = SYNTHETIC_ROOT):
        """Initialize loader with root directory path.

        Parameters
        ----------
        root : str
            Root directory containing .npy files for each shape
        """
        self.root = Path(root)

    def load_all(self) -> list[tuple[str, np.ndarray]]:
        """Load all 7 shapes.

        Returns
        -------
        list[tuple[str, np.ndarray]]
            List of (name, volume) pairs where volume is bool (32, 32, 32)

        Raises
        ------
        FileNotFoundError
            If any shape file is missing
        """
        result = []
        for name in self.SHAPES:
            volume = self.load_by_name(name)
            result.append((name, volume))
        return result

    def load_by_name(self, name: str) -> np.ndarray:
        """Load single shape by stem name.

        Parameters
        ----------
        name : str
            Shape name (e.g., 'cube')

        Returns
        -------
        np.ndarray
            Boolean volume (32, 32, 32)

        Raises
        ------
        FileNotFoundError
            If the shape file does not exist
        ValueError
            If the volume has unexpected shape
        """
        if name not in self.SHAPES:
            raise ValueError(f"Unknown shape: {name}. Must be one of {self.SHAPES}")

        path = self.root / f"{name}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Shape file not found: {path}")

        volume = np.load(path).astype(bool)
        if volume.shape != (32, 32, 32):
            raise ValueError(
                f"Unexpected volume shape for {name}: got {volume.shape}, "
                f"expected (32, 32, 32)"
            )
        return volume


class SyntheticPointCloudLoader:
    """Load synthetic point cloud shapes from PLY files."""

    SHAPES = ["cone", "cube", "cuboid", "cylinder", "pyramid", "sphere", "torus"]

    def __init__(self, root: str = POINTCLOUD_ROOT + "/synthetic"):
        self.root = Path(root)

    def load_all(self) -> list[tuple[str, np.ndarray]]:
        result = []
        for name in self.SHAPES:
            pts = self.load_by_name(name)
            result.append((name, pts))
        return result

    def load_by_name(self, name: str) -> np.ndarray:
        if name not in self.SHAPES:
            raise ValueError(f"Unknown shape: {name}. Must be one of {self.SHAPES}")
        path = self.root / f"{name}.ply"
        if not path.exists():
            raise FileNotFoundError(f"Point cloud file not found: {path}")
        pcd = o3d.io.read_point_cloud(str(path))
        return np.asarray(pcd.points, dtype=np.float64)


class RealPointCloudLoader:
    """Load real point cloud samples from all PLY files in a directory.

    Mirrors the interface of SyntheticPointCloudLoader so the two can be
    used interchangeably in batch runners.  Any ``.ply`` file present in
    *root* is treated as a sample; the stem of the filename is the name.

    The expected directory is ``data/Pointcloud/real/``, pre-populated by
    ``data/Pointcloud/generate_real.py``.
    """

    def __init__(self, root: str = POINTCLOUD_ROOT + "/real"):
        self.root = Path(root)

    def load_all(self) -> list[tuple[str, np.ndarray]]:
        """Load all PLY files found in the root directory.

        Returns
        -------
        list[tuple[str, np.ndarray]]
            List of (name, points) pairs, sorted by name.  *points* is
            a float64 (N, 3) array.

        Raises
        ------
        FileNotFoundError
            If the root directory does not exist.
        """
        if not self.root.exists():
            raise FileNotFoundError(
                f"Real point cloud directory not found: {self.root}\n"
                "Run data/Pointcloud/generate_real.py to populate it."
            )
        result: list[tuple[str, np.ndarray]] = []
        for ply_path in sorted(self.root.glob("*.ply")):
            pcd = o3d.io.read_point_cloud(str(ply_path))
            pts = np.asarray(pcd.points, dtype=np.float64)
            if pts.shape[0] > 0:
                result.append((ply_path.stem, pts))
        return result


class ModelNetLoader:
    """Lazy loader for ModelNet10 voxel dataset (compressed .npy.gz)."""

    def __init__(self, path: str):
        """Initialize loader. Data is loaded lazily on first access.

        Parameters
        ----------
        path : str
            Path to .npy.gz file
        """
        self.path = Path(path)
        self._data: np.ndarray | None = None

    def _load(self) -> None:
        """Load data from gzip file if not already loaded."""
        if self._data is None:
            if not self.path.exists():
                raise FileNotFoundError(f"ModelNet file not found: {self.path}")
            try:
                with gzip.open(self.path, "rb") as f:
                    self._data = np.load(f).astype(bool)
            except Exception as e:
                raise RuntimeError(f"Failed to load ModelNet file {self.path}: {e}")

    def __len__(self) -> int:
        """Get number of samples in dataset."""
        self._load()
        assert self._data is not None
        return self._data.shape[0]

    def load_by_index(self, idx: int) -> np.ndarray:
        """Load single sample by index.

        Parameters
        ----------
        idx : int
            Sample index

        Returns
        -------
        np.ndarray
            Boolean volume (32, 32, 32)

        Raises
        ------
        IndexError
            If index is out of range
        """
        self._load()
        assert self._data is not None
        if not (0 <= idx < len(self._data)):
            raise IndexError(
                f"Index {idx} out of range for dataset with {len(self._data)} samples"
            )
        return self._data[idx, 0].astype(bool)

    def load_random(self) -> tuple[int, np.ndarray]:
        """Load a random sample.

        Returns
        -------
        tuple[int, np.ndarray]
            (index, volume) where volume is bool (32, 32, 32)
        """
        self._load()
        assert self._data is not None
        idx = np.random.randint(0, len(self._data))
        return (idx, self.load_by_index(idx))

    def load_sequential(self) -> Iterator[tuple[int, np.ndarray]]:
        """Iterate over all samples sequentially.

        Yields
        ------
        tuple[int, np.ndarray]
            (index, volume) pairs for every sample
        """
        self._load()
        assert self._data is not None
        for idx in range(len(self._data)):
            yield (idx, self.load_by_index(idx))

    def load_first_n(self, n: int) -> list[tuple[int, np.ndarray]]:
        """Load first n samples.

        Parameters
        ----------
        n : int
            Number of samples to load

        Returns
        -------
        list[tuple[int, np.ndarray]]
            List of (index, volume) pairs

        Raises
        ------
        ValueError
            If n > number of samples
        """
        self._load()
        assert self._data is not None
        if n > len(self._data):
            raise ValueError(
                f"Cannot load {n} samples: dataset has only {len(self._data)}"
            )
        return [(idx, self.load_by_index(idx)) for idx in range(n)]

    def load_range(self, start: int, end: int) -> list[tuple[int, np.ndarray]]:
        """Load samples in index range [start, end).

        Parameters
        ----------
        start : int
            Starting index (inclusive)
        end : int
            Ending index (exclusive)

        Returns
        -------
        list[tuple[int, np.ndarray]]
            List of (index, volume) pairs

        Raises
        ------
        ValueError
            If start/end are out of range or invalid
        """
        self._load()
        assert self._data is not None
        if not (0 <= start < end <= len(self._data)):
            raise ValueError(
                f"Invalid range [{start}, {end}): must satisfy 0 <= start < end <= {len(self._data)}"
            )
        return [(idx, self.load_by_index(idx)) for idx in range(start, end)]


def load_pointcloud(path: str) -> "o3d.geometry.PointCloud":
    """Load PLY point cloud via open3d.

    Parameters
    ----------
    path : str
        Path to PLY file

    Returns
    -------
    o3d.geometry.PointCloud
        Open3D point cloud object

    Raises
    ------
    FileNotFoundError
        If file does not exist
    RuntimeError
        If file cannot be read
    """
    import open3d as o3d

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Point cloud file not found: {path}")

    try:
        pcd = o3d.io.read_point_cloud(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read point cloud {path}: {e}")

    if pcd.has_points() and len(pcd.points) == 0:
        raise RuntimeError(f"Point cloud {path} is empty")

    return pcd


def load_image(path: str) -> np.ndarray:
    """Load image via cv2, returns float32 [0, 1] grayscale (H, W).

    Parameters
    ----------
    path : str
        Path to image file

    Returns
    -------
    np.ndarray
        Grayscale image as float32 in range [0, 1], shape (H, W)

    Raises
    ------
    FileNotFoundError
        If file does not exist
    RuntimeError
        If file cannot be read
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image {path}")

    # Convert to float32 [0, 1]
    img_float = img.astype(np.float32) / 255.0
    return img_float
