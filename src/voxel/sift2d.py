from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import cv2
import numpy as np
from matplotlib import image as mpimg

from src.voxel.params import SIFT2DParams


@dataclass
class SIFT2DOrientationSignature:
    keypoint: np.ndarray
    octave_index: int
    scale_index: int
    center_yx: np.ndarray
    sigma: float
    patch: np.ndarray
    patch_origin_yx: np.ndarray
    sample_points: np.ndarray
    gradient_vectors: np.ndarray
    magnitudes: np.ndarray
    orientations: np.ndarray
    histogram: np.ndarray
    bin_edges: np.ndarray
    dominant_orientation: float


@dataclass
class SIFT2DResult:
    original_image: np.ndarray
    gaussian_pyramid: list[list[np.ndarray]]
    sigma_pyramid: list[list[float]]
    dog_pyramid: list[list[np.ndarray]]
    extrema_local: list[np.ndarray]
    extrema_global: np.ndarray
    orientation_signatures: list[SIFT2DOrientationSignature]


class SIFT2D:
    def __init__(self, params: SIFT2DParams | None = None):
        self.params = params or SIFT2DParams()

    def run(self, image_input: str | np.ndarray) -> SIFT2DResult:
        image = self._to_grayscale_float(image_input)
        gaussians, sigmas = self.build_gaussian_pyramid(image)
        dogs = self.compute_dog_pyramid(gaussians)
        extrema_local = self.detect_extrema(dogs)
        extrema_global = self.to_global_coordinates(extrema_local)
        orientation_signatures = self.compute_orientation_signatures(
            gaussians, sigmas, extrema_local
        )
        return SIFT2DResult(
            original_image=image,
            gaussian_pyramid=gaussians,
            sigma_pyramid=sigmas,
            dog_pyramid=dogs,
            extrema_local=extrema_local,
            extrema_global=extrema_global,
            orientation_signatures=orientation_signatures,
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
                            refined = self._refine_extremum_2d(
                                dogs_list=dogs_list,
                                scale_idx=scale_idx,
                                y=y,
                                x=x,
                                threshold=threshold,
                            )
                            if refined is not None:
                                points.append(refined)

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

    def compute_orientation_signatures(
        self,
        gaussian_pyramid: list[list[np.ndarray]],
        sigma_pyramid: list[list[float]],
        extrema_local: Iterable[np.ndarray],
    ) -> list[SIFT2DOrientationSignature]:
        signatures: list[SIFT2DOrientationSignature] = []
        bins = int(self.params.orientation_bins)
        if bins <= 0:
            return signatures

        bin_edges = np.linspace(0.0, 2.0 * np.pi, bins + 1, dtype=np.float32)

        for octave_extrema in extrema_local:
            if octave_extrema.size == 0:
                continue

            octave_index = (
                int(round(float(octave_extrema[0, 4])))
                if octave_extrema.shape[0]
                else 0
            )
            octave_index = int(np.clip(octave_index, 0, len(gaussian_pyramid) - 1))

            for row in octave_extrema:
                center_y = float(row[0])
                center_x = float(row[1])
                scale_index = int(
                    np.clip(
                        round(float(row[2])), 0, len(gaussian_pyramid[octave_index]) - 1
                    )
                )
                response = float(row[3])
                octave = int(round(float(row[4])))
                octave = int(np.clip(octave, 0, len(gaussian_pyramid) - 1))

                signature = self._compute_orientation_signature_for_keypoint(
                    gaussian_pyramid[octave],
                    sigma_pyramid[octave],
                    center_yx=np.array([center_y, center_x], dtype=np.float32),
                    scale_index=scale_index,
                    response=response,
                    octave_index=octave,
                    bin_edges=bin_edges,
                )
                if signature is not None:
                    signatures.append(signature)

        return signatures

    def _compute_orientation_signature_for_keypoint(
        self,
        octave_images: list[np.ndarray],
        octave_sigmas: list[float],
        center_yx: np.ndarray,
        scale_index: int,
        response: float,
        octave_index: int,
        bin_edges: np.ndarray,
    ) -> SIFT2DOrientationSignature | None:
        if scale_index < 0 or scale_index >= len(octave_images):
            return None

        image = octave_images[scale_index].astype(np.float32)
        sigma = float(octave_sigmas[scale_index])
        center_y = float(center_yx[0])
        center_x = float(center_yx[1])

        gradient = np.gradient(image)
        grad_y = np.asarray(gradient[0], dtype=np.float32)
        grad_x = np.asarray(gradient[1], dtype=np.float32)
        window_radius = max(
            1, int(round(self.params.orientation_window_factor * sigma))
        )
        weight_sigma = max(
            1e-6, float(self.params.orientation_weight_sigma_factor) * sigma
        )

        y_center = int(round(center_y))
        x_center = int(round(center_x))
        y0 = max(0, y_center - window_radius)
        y1 = min(image.shape[0] - 1, y_center + window_radius)
        x0 = max(0, x_center - window_radius)
        x1 = min(image.shape[1] - 1, x_center + window_radius)
        if y0 > y1 or x0 > x1:
            return None

        yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
        dy = grad_y[y0 : y1 + 1, x0 : x1 + 1]
        dx = grad_x[y0 : y1 + 1, x0 : x1 + 1]
        patch = image[y0 : y1 + 1, x0 : x1 + 1].astype(np.float32)

        sample_points = np.column_stack([yy.ravel(), xx.ravel()]).astype(np.float32)
        gradient_vectors = np.column_stack([dy.ravel(), dx.ravel()]).astype(np.float32)
        magnitudes = np.sqrt(np.sum(gradient_vectors**2, axis=1)).astype(np.float32)
        orientations = np.mod(
            np.arctan2(gradient_vectors[:, 0], gradient_vectors[:, 1]), 2.0 * np.pi
        ).astype(np.float32)

        dist2 = (yy.astype(np.float32) - center_y) ** 2 + (
            xx.astype(np.float32) - center_x
        ) ** 2
        weights = np.exp(-dist2 / (2.0 * weight_sigma**2)).astype(np.float32).ravel()
        weighted_magnitudes = magnitudes * weights

        histogram, _ = np.histogram(
            orientations,
            bins=bin_edges,
            weights=weighted_magnitudes,
        )
        histogram = histogram.astype(np.float32)
        dominant_bin = int(np.argmax(histogram)) if histogram.size else 0
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        dominant_orientation = (
            float(bin_centers[dominant_bin]) if bin_centers.size else 0.0
        )

        return SIFT2DOrientationSignature(
            keypoint=np.array(
                [center_y, center_x, float(scale_index), response, float(octave_index)],
                dtype=np.float32,
            ),
            octave_index=int(octave_index),
            scale_index=int(scale_index),
            center_yx=center_yx.astype(np.float32),
            sigma=sigma,
            patch=patch,
            patch_origin_yx=np.array([float(y0), float(x0)], dtype=np.float32),
            sample_points=sample_points,
            gradient_vectors=gradient_vectors,
            magnitudes=magnitudes,
            orientations=orientations,
            histogram=histogram,
            bin_edges=bin_edges.astype(np.float32),
            dominant_orientation=dominant_orientation,
        )

    def _refine_extremum_2d(
        self,
        dogs_list: list[np.ndarray],
        scale_idx: int,
        y: int,
        x: int,
        threshold: float,
    ) -> tuple[float, float, float, float] | None:
        s = int(scale_idx)
        yy = int(y)
        xx = int(x)
        offset_threshold = float(self.params.refinement_offset_threshold)
        singular_eps = float(self.params.refinement_singular_eps)
        if s <= 0 or s >= len(dogs_list) - 1:
            return None

        prev_img = dogs_list[s - 1]
        curr_img = dogs_list[s]
        next_img = dogs_list[s + 1]
        h, w = curr_img.shape

        if yy <= 0 or yy >= h - 1 or xx <= 0 or xx >= w - 1:
            return None

        value = float(curr_img[yy, xx])

        dy = 0.5 * float(curr_img[yy + 1, xx] - curr_img[yy - 1, xx])
        dx = 0.5 * float(curr_img[yy, xx + 1] - curr_img[yy, xx - 1])
        ds = 0.5 * float(next_img[yy, xx] - prev_img[yy, xx])
        grad = np.array([dy, dx, ds], dtype=np.float64)

        dyy = float(curr_img[yy + 1, xx] - 2.0 * value + curr_img[yy - 1, xx])
        dxx = float(curr_img[yy, xx + 1] - 2.0 * value + curr_img[yy, xx - 1])
        dss = float(next_img[yy, xx] - 2.0 * value + prev_img[yy, xx])

        dxy = 0.25 * float(
            curr_img[yy + 1, xx + 1]
            - curr_img[yy + 1, xx - 1]
            - curr_img[yy - 1, xx + 1]
            + curr_img[yy - 1, xx - 1]
        )
        dys = 0.25 * float(
            next_img[yy + 1, xx]
            - next_img[yy - 1, xx]
            - prev_img[yy + 1, xx]
            + prev_img[yy - 1, xx]
        )
        dxs = 0.25 * float(
            next_img[yy, xx + 1]
            - next_img[yy, xx - 1]
            - prev_img[yy, xx + 1]
            + prev_img[yy, xx - 1]
        )

        hessian = np.array(
            [
                [dyy, dxy, dys],
                [dxy, dxx, dxs],
                [dys, dxs, dss],
            ],
            dtype=np.float64,
        )

        if float(abs(np.linalg.det(hessian))) < singular_eps:
            return None

        try:
            offset = -np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            return None

        if np.any(np.abs(offset) > offset_threshold):
            return None

        refined_response = value + 0.5 * float(np.dot(grad, offset))
        if abs(refined_response) < threshold:
            return None

        return (
            float(yy + offset[0]),
            float(xx + offset[1]),
            float(s + offset[2]),
            float(refined_response),
        )
