from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PreprocessedPage:
    base_rgb: np.ndarray
    gray: np.ndarray
    binary: np.ndarray
    deskew_angle: float
    scale_factor: float

    @property
    def binary_rgb(self) -> np.ndarray:
        return cv2.cvtColor(self.binary, cv2.COLOR_GRAY2RGB)


def preprocess_page(image_rgb: np.ndarray, min_width: int = 2200) -> PreprocessedPage:
    if image_rgb.ndim != 3:
        raise ValueError("Expected an RGB image array.")

    image_rgb = image_rgb.copy()
    scale_factor = 1.0

    if image_rgb.shape[1] < min_width:
        scale_factor = float(min_width) / float(image_rgb.shape[1])
        image_rgb = cv2.resize(
            image_rgb,
            None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_CUBIC,
        )

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    deskew_angle = estimate_skew_angle(gray)
    if abs(deskew_angle) >= 0.15:
        image_rgb = rotate_image(image_rgb, deskew_angle, 255)
        gray = rotate_image(gray, deskew_angle, 255)

    gray = cv2.GaussianBlur(gray, (0, 0), 0.8)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return PreprocessedPage(
        base_rgb=image_rgb,
        gray=gray,
        binary=binary,
        deskew_angle=float(deskew_angle),
        scale_factor=scale_factor,
    )


def estimate_skew_angle(gray: np.ndarray) -> float:
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 200:
        return 0.0

    rect = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))
    angle = float(rect[-1])

    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle -= 90

    if abs(angle) > 10:
        return 0.0
    return angle


def rotate_image(image: np.ndarray, angle: float, border_value: int | tuple[int, int, int]) -> np.ndarray:
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
