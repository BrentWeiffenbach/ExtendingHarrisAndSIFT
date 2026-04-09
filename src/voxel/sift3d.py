from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from src.common.base_detector import Detector3D
from src.voxel.params import SIFT3DParams


@dataclass
class SIFT3DGaussianResult:
    original_volume: np.ndarray
    gaussian_pyramid: list[list[np.ndarray]]
    sigma_pyramid: list[list[float]]
    dog_pyramid: list[list[np.ndarray]]
    dog_sigma_pairs: list[list[tuple[float, float, float]]]
    extrema_local: list[np.ndarray]
    extrema_global: np.ndarray


class SIFT3DVoxel(Detector3D):
    def __init__(self, params: SIFT3DParams | None = None):
        super().__init__(params or SIFT3DParams())

    def run(self, volume_input: str | np.ndarray) -> SIFT3DGaussianResult:
        volume = self._to_float_volume(volume_input)
        gaussian_pyramid, sigma_pyramid = self.build_gaussian_pyramid(volume)
        dog_pyramid, dog_sigma_pairs = self.compute_dog_pyramid(
            gaussian_pyramid, sigma_pyramid
        )
        extrema_local = self.detect_extrema_3d(dog_pyramid, dog_sigma_pairs)
        extrema_global = self.to_global_extrema(extrema_local)
        return SIFT3DGaussianResult(
            original_volume=volume,
            gaussian_pyramid=gaussian_pyramid,
            sigma_pyramid=sigma_pyramid,
            dog_pyramid=dog_pyramid,
            dog_sigma_pairs=dog_sigma_pairs,
            extrema_local=extrema_local,
            extrema_global=extrema_global,
        )

    def detect(self, data: np.ndarray) -> np.ndarray:
        _ = self.run(data)
        return np.empty((0, 3), dtype=np.float32)

    def _to_float_volume(self, volume_input: str | np.ndarray) -> np.ndarray:
        if isinstance(volume_input, str):
            volume = np.load(volume_input)
        else:
            volume = np.asarray(volume_input)

        if volume.ndim != 3:
            raise ValueError("Voxel input must be a 3D array with shape (D, H, W)")

        volume = volume.astype(np.float32)
        max_val = float(np.max(volume)) if volume.size > 0 else 0.0
        if max_val > 1.0:
            volume /= max_val
        return volume

    def build_gaussian_pyramid(
        self, volume: np.ndarray
    ) -> Tuple[list[list[np.ndarray]], list[list[float]]]:
        gaussian_pyramid: list[list[np.ndarray]] = []
        sigma_pyramid: list[list[float]] = []

        current = volume
        for _octave in range(self.params.num_octaves):
            if min(current.shape) < self.params.min_size:
                break

            octave_volumes: list[np.ndarray] = []
            octave_sigmas: list[float] = []

            for scale in range(self.params.scales_per_octave):
                sigma = self.params.base_sigma * (
                    2.0 ** (scale / self.params.scales_per_octave)
                )
                blurred = gaussian_filter(
                    current,
                    sigma=sigma,
                    mode=self.params.border_mode,
                ).astype(np.float32)
                octave_volumes.append(blurred)
                octave_sigmas.append(float(sigma))

            gaussian_pyramid.append(octave_volumes)
            sigma_pyramid.append(octave_sigmas)

            downsample_source = octave_volumes[-1]
            downsampled = self.downsample_octave(downsample_source)
            if min(downsampled.shape) < self.params.min_size:
                break
            current = downsampled

        return gaussian_pyramid, sigma_pyramid

    def downsample_octave(self, volume: np.ndarray) -> np.ndarray:
        smooth = gaussian_filter(
            volume,
            sigma=self.params.downsample_sigma,
            mode=self.params.border_mode,
        )
        factor = max(1, int(self.params.downsample_factor))
        return smooth[::factor, ::factor, ::factor].astype(np.float32)

    def compute_dog_pyramid(
        self,
        gaussian_pyramid: list[list[np.ndarray]],
        sigma_pyramid: list[list[float]],
    ) -> Tuple[list[list[np.ndarray]], list[list[tuple[float, float, float]]]]:
        dog_pyramid: list[list[np.ndarray]] = []
        dog_sigma_pairs: list[list[tuple[float, float, float]]] = []

        for octave_idx, octave_volumes in enumerate(gaussian_pyramid):
            octave_dogs: list[np.ndarray] = []
            octave_pairs: list[tuple[float, float, float]] = []
            octave_sigmas = sigma_pyramid[octave_idx]

            if len(octave_volumes) < 2:
                dog_pyramid.append(octave_dogs)
                dog_sigma_pairs.append(octave_pairs)
                continue

            for i in range(len(octave_volumes) - 1):
                sigma_low = float(octave_sigmas[i])
                sigma_high = float(octave_sigmas[i + 1])
                dog = (octave_volumes[i + 1] - octave_volumes[i]).astype(np.float32)
                octave_dogs.append(dog)
                octave_pairs.append((sigma_low, sigma_high, sigma_high - sigma_low))

            dog_pyramid.append(octave_dogs)
            dog_sigma_pairs.append(octave_pairs)

        return dog_pyramid, dog_sigma_pairs

    def detect_extrema_3d(
        self,
        dog_pyramid: list[list[np.ndarray]],
        dog_sigma_pairs: list[list[tuple[float, float, float]]],
    ) -> list[np.ndarray]:
        threshold = float(self.params.extrema_contrast_threshold)
        border = int(self.params.extrema_border)
        extrema_by_octave: list[np.ndarray] = []

        for octave_idx, octave_dogs in enumerate(dog_pyramid):
            if len(octave_dogs) < 3:
                extrema_by_octave.append(np.empty((0, 7), dtype=np.float32))
                continue

            points: list[tuple[float, float, float, float, float, float, float]] = []
            for dog_idx in range(1, len(octave_dogs) - 1):
                prev_vol = octave_dogs[dog_idx - 1]
                curr_vol = octave_dogs[dog_idx]
                next_vol = octave_dogs[dog_idx + 1]
                z_max, y_max, x_max = curr_vol.shape
                sigma_low, sigma_high, _ = dog_sigma_pairs[octave_idx][dog_idx]
                sigma_char = float(np.sqrt(max(1e-12, sigma_low * sigma_high)))

                for z in range(border, z_max - border):
                    for y in range(border, y_max - border):
                        for x in range(border, x_max - border):
                            value = float(curr_vol[z, y, x])
                            if abs(value) < threshold:
                                continue

                            patch_prev = prev_vol[
                                z - 1 : z + 2, y - 1 : y + 2, x - 1 : x + 2
                            ]
                            patch_curr = curr_vol[
                                z - 1 : z + 2, y - 1 : y + 2, x - 1 : x + 2
                            ]
                            patch_next = next_vol[
                                z - 1 : z + 2, y - 1 : y + 2, x - 1 : x + 2
                            ]

                            neighbors = np.concatenate(
                                [
                                    patch_prev.ravel(),
                                    patch_curr.ravel(),
                                    patch_next.ravel(),
                                ]
                            )
                            center_idx = 27 + 13
                            neighbors = np.delete(neighbors, center_idx)

                            is_max = value > float(np.max(neighbors))
                            is_min = value < float(np.min(neighbors))
                            if is_max or is_min:
                                points.append(
                                    (
                                        float(z),
                                        float(y),
                                        float(x),
                                        float(dog_idx),
                                        value,
                                        float(octave_idx),
                                        sigma_char,
                                    )
                                )

            extrema = (
                np.asarray(points, dtype=np.float32)
                if points
                else np.empty((0, 7), dtype=np.float32)
            )
            extrema_by_octave.append(extrema)

        return extrema_by_octave

    def to_global_extrema(self, extrema_local: list[np.ndarray]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for local in extrema_local:
            if local.size == 0:
                continue

            scale_factor = 2.0 ** local[:, 5:6]
            zyx_global = local[:, 0:3] * scale_factor
            row = np.concatenate(
                [
                    zyx_global,
                    local[:, 6:7],
                    local[:, 4:5],
                    local[:, 5:6],
                    local[:, 3:4],
                ],
                axis=1,
            )
            rows.append(row)

        if not rows:
            return np.empty((0, 7), dtype=np.float32)
        return np.vstack(rows).astype(np.float32)
