from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import re

import easyocr
import numpy as np
from PIL import Image


@dataclass
class OCRLine:
    text: str
    confidence: float
    bbox: list[int]
    source: str


def _normalize_text(text: str) -> str:
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _polygon_to_bbox(points) -> list[int]:
    xs = [int(p[0]) for p in points]
    ys = [int(p[1]) for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def _iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def _text_quality_score(text: str, confidence: float) -> float:
    text = _normalize_text(text)
    if not text:
        return -10.0
    if confidence < 0.05 and len(text) < 20:
        return -10.0
    repeats = len(re.findall(r"\b(and|the|of|for)\b", text.lower()))
    weird = sum(ch in "[]{}~" for ch in text)
    alpha = sum(ch.isalpha() for ch in text)
    digits = sum(ch.isdigit() for ch in text)
    alnum = alpha + digits
    density = alnum / max(1, len(text))
    return confidence * 10 + min(len(text), 40) * 0.1 + density - repeats * 0.8 - weird


@lru_cache(maxsize=8)
def _easyocr_reader(lang_key: tuple[str, ...]) -> easyocr.Reader:
    return easyocr.Reader(list(lang_key), gpu=False)


class EasyOCRBackend:
    name = "easyocr"

    def __init__(self, languages: list[str] | None = None):
        self.languages = tuple(languages or ["en"])

    def recognize(self, image_rgb: np.ndarray) -> list[OCRLine]:
        reader = _easyocr_reader(self.languages)
        results = reader.readtext(image_rgb, detail=1, paragraph=False)
        lines: list[OCRLine] = []
        for points, text, confidence in results:
            text = _normalize_text(text)
            if not text:
                continue
            lines.append(
                OCRLine(
                    text=text,
                    confidence=float(confidence),
                    bbox=_polygon_to_bbox(points),
                    source=self.name,
                )
            )
        return lines


class SuryaBackend:
    name = "surya"

    def __init__(self, languages: list[str] | None = None):
        self.languages = languages or ["en"]
        from surya.model.detection.model import (
            load_model as load_det_model,
            load_processor as load_det_processor,
        )
        from surya.model.recognition.model import load_model as load_rec_model
        from surya.model.recognition.processor import load_processor as load_rec_processor

        self._det_model = load_det_model()
        self._det_processor = load_det_processor()
        self._rec_model = load_rec_model()
        self._rec_processor = load_rec_processor()

    def recognize(self, image_rgb: np.ndarray) -> list[OCRLine]:
        from surya.detection import batch_text_detection
        from surya.recognition import batch_recognition

        pil_image = Image.fromarray(image_rgb)
        detections = batch_text_detection(
            [pil_image], self._det_model, self._det_processor
        )
        page = detections[0]

        line_crops: list[Image.Image] = []
        bboxes: list[list[int]] = []
        width, height = pil_image.size

        for item in getattr(page, "bboxes", []):
            raw = getattr(item, "bbox", None)
            if raw is None:
                continue
            x1 = max(0, int(raw[0]))
            y1 = max(0, int(raw[1]))
            x2 = min(width, int(raw[2]))
            y2 = min(height, int(raw[3]))
            if x2 <= x1 or y2 <= y1:
                continue
            bboxes.append([x1, y1, x2, y2])
            line_crops.append(pil_image.crop((x1, y1, x2, y2)))

        if not line_crops:
            return []

        texts, confidences = batch_recognition(
            line_crops,
            [self.languages] * len(line_crops),
            self._rec_model,
            self._rec_processor,
        )

        lines: list[OCRLine] = []
        for bbox, text, confidence in zip(bboxes, texts, confidences):
            text = _normalize_text(text)
            if not text:
                continue
            lines.append(
                OCRLine(
                    text=text,
                    confidence=float(confidence),
                    bbox=bbox,
                    source=self.name,
                )
            )
        return lines


class EnsembleOCR:
    def __init__(self, backends: list):
        self.backends = backends

    def recognize(self, image_rgb: np.ndarray) -> list[OCRLine]:
        candidates: list[OCRLine] = []
        for backend in self.backends:
            try:
                candidates.extend(backend.recognize(image_rgb))
            except Exception as exc:
                print(f"[OCR] Backend {backend.name} failed: {exc}")

        candidates = [
            line for line in candidates
            if _text_quality_score(line.text, line.confidence) > 0
        ]
        candidates.sort(
            key=lambda line: (
                -_text_quality_score(line.text, line.confidence),
                line.bbox[1],
                line.bbox[0],
            )
        )

        merged: list[OCRLine] = []
        for candidate in candidates:
            duplicate = False
            for existing in merged:
                same_region = _iou(candidate.bbox, existing.bbox) >= 0.5
                same_text = _normalize_text(candidate.text).lower() == _normalize_text(existing.text).lower()
                near = _centre_distance(candidate.bbox, existing.bbox) <= max(12.0, _line_height(existing.bbox))
                if same_region or (same_text and near):
                    duplicate = True
                    if _text_quality_score(candidate.text, candidate.confidence) > _text_quality_score(existing.text, existing.confidence):
                        existing.text = candidate.text
                        existing.confidence = candidate.confidence
                        existing.bbox = candidate.bbox
                        existing.source = candidate.source
                    break
            if not duplicate:
                merged.append(candidate)

        merged.sort(key=lambda line: (line.bbox[1], line.bbox[0]))
        return merged


def _line_height(bbox: list[int]) -> float:
    return max(1.0, float(bbox[3] - bbox[1]))


def _centre_distance(a: list[int], b: list[int]) -> float:
    ax = (a[0] + a[2]) / 2
    ay = (a[1] + a[3]) / 2
    bx = (b[0] + b[2]) / 2
    by = (b[1] + b[3]) / 2
    return math.hypot(ax - bx, ay - by)
