from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import cv2
import numpy as np
from matplotlib import image as mpimg

from src.voxel.params import SIFT2DParams


@dataclass
class SIFT2DResult:
    original_image: np.ndarray
    gaussian_pyramid: list[list[np.ndarray]]
    sigma_pyramid: list[list[float]]
    dog_pyramid: list[list[np.ndarray]]
    extrema_local: list[np.ndarray]
    extrema_global: np.ndarray


class SIFT2D:
    def __init__(self, params: SIFT2DParams | None = None):
        self.params = params or SIFT2DParams()

    def run(self, image_input: str | np.ndarray) -> SIFT2DResult:
        image = self._to_grayscale_float(image_input)
        gaussians, sigmas = self.build_gaussian_pyramid(image)
        dogs = self.compute_dog_pyramid(gaussians)
        extrema_local = self.detect_extrema(dogs)
        extrema_global = self.to_global_coordinates(extrema_local)
        return SIFT2DResult(
            original_image=image,
            gaussian_pyramid=gaussians,
            sigma_pyramid=sigmas,
            dog_pyramid=dogs,
            extrema_local=extrema_local,
            extrema_global=extrema_global,
        )

    def _to_grayscale_float(self, image_input: str | np.ndarray) -> np.ndarray:
        if isinstance(image_input, str):
            image = mpimg.imread(image_input)
        else:
            image = np.asarray(image_input)

        if image.ndim == 3:
            image = image[..., :3].mean(axis=2)

        image = image.astype(np.float32)
        max_val = float(np.max(image)) if image.size > 0 else 0.0
        if max_val > 1.0:
            image /= 255.0
        return image

    def build_gaussian_pyramid(
        self, image: np.ndarray
    ) -> Tuple[list[list[np.ndarray]], list[list[float]]]:
        num_octaves = self.params.num_octaves
        scales_per_octave = self.params.scales_per_octave
        base_sigma = self.params.base_sigma

        current = image
        gaussian_pyramid: list[list[np.ndarray]] = []
        sigma_pyramid: list[list[float]] = []

        for _octave in range(num_octaves):
            octave_images: list[np.ndarray] = []
            octave_sigmas: list[float] = []
            for scale in range(scales_per_octave):
                sigma = base_sigma * (2.0 ** (scale / scales_per_octave))
                blurred = cv2.GaussianBlur(current, (0, 0), sigmaX=sigma, sigmaY=sigma)
                octave_images.append(blurred)
                octave_sigmas.append(sigma)

            gaussian_pyramid.append(octave_images)
            sigma_pyramid.append(octave_sigmas)

            if min(current.shape[:2]) <= 16:
                break
            current = cv2.pyrDown(current)

        return gaussian_pyramid, sigma_pyramid

    def compute_dog_pyramid(
        self, gaussian_pyramid: Iterable[Iterable[np.ndarray]]
    ) -> list[list[np.ndarray]]:
        dog_pyramid: list[list[np.ndarray]] = []
        for octave_images in gaussian_pyramid:
            octave_images_list = list(octave_images)
            octave_dogs: list[np.ndarray] = []
            for idx in range(len(octave_images_list) - 1):
                dog = octave_images_list[idx + 1] - octave_images_list[idx]
                octave_dogs.append(dog.astype(np.float32))
            dog_pyramid.append(octave_dogs)
        return dog_pyramid

    def detect_extrema(
        self, dog_pyramid: Iterable[Iterable[np.ndarray]]
    ) -> list[np.ndarray]:
        extrema_by_octave: list[np.ndarray] = []
        threshold = float(self.params.contrast_threshold)
        border = int(self.params.border)

        for octave, dogs in enumerate(dog_pyramid):
            dogs_list = list(dogs)
            if len(dogs_list) < 3:
                extrema_by_octave.append(np.empty((0, 5), dtype=np.float32))
                continue

            points: list[tuple[float, float, float, float]] = []
            for scale_idx in range(1, len(dogs_list) - 1):
                prev_img = dogs_list[scale_idx - 1]
                curr_img = dogs_list[scale_idx]
                next_img = dogs_list[scale_idx + 1]
                h, w = curr_img.shape

                for y in range(border, h - border):
                    for x in range(border, w - border):
                        value = float(curr_img[y, x])
                        if abs(value) < threshold:
                            continue

                        patch_prev = prev_img[y - 1 : y + 2, x - 1 : x + 2]
                        patch_curr = curr_img[y - 1 : y + 2, x - 1 : x + 2]
                        patch_next = next_img[y - 1 : y + 2, x - 1 : x + 2]

                        neighbors = np.concatenate(
                            [patch_prev.ravel(), patch_curr.ravel(), patch_next.ravel()]
                        )
                        center_idx = 9 + 4
                        neighbors = np.delete(neighbors, center_idx)

                        is_max = value > float(np.max(neighbors))
                        is_min = value < float(np.min(neighbors))
                        if is_max or is_min:
                            points.append((float(y), float(x), float(scale_idx), value))

            if points:
                extrema = np.asarray(points, dtype=np.float32)
                octave_col = np.full(
                    (extrema.shape[0], 1), float(octave), dtype=np.float32
                )
                extrema = np.concatenate([extrema, octave_col], axis=1)
            else:
                extrema = np.empty((0, 5), dtype=np.float32)

            extrema_by_octave.append(extrema)

        return extrema_by_octave

    def to_global_coordinates(self, extrema_local: Iterable[np.ndarray]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for extrema in extrema_local:
            if extrema.size == 0:
                continue
            scale_factor = 2.0 ** extrema[:, 4:5]
            yx_global = extrema[:, 0:2] * scale_factor
            row = np.concatenate(
                [
                    yx_global,
                    extrema[:, 2:3],
                    extrema[:, 3:4],
                    extrema[:, 4:5],
                ],
                axis=1,
            )
            rows.append(row)

        if not rows:
            return np.empty((0, 5), dtype=np.float32)
        return np.vstack(rows).astype(np.float32)
