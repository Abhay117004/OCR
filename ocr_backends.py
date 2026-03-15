from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import html
import math
from pathlib import Path
import re
import shutil
from typing import Iterable

import numpy as np
from PIL import Image
import pytesseract


TESSERACT_FALLBACK_PATHS = (
    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
)

GLUED_FRAGMENT_PATTERNS = (
    "inthe",
    "tothe",
    "ofthe",
    "fromthe",
    "forthe",
    "withthe",
    "onthe",
    "underthe",
    "bythe",
    "normalactivity",
    "meansentire",
    "ofa",
)

COMMON_OCR_REPAIRS = (
    (r"\binthe\b", "in the"),
    (r"\btothe\b", "to the"),
    (r"\bofthe\b", "of the"),
    (r"\bfromthe\b", "from the"),
    (r"\bforthe\b", "for the"),
    (r"\bwiththe\b", "with the"),
    (r"\bonthe\b", "on the"),
    (r"\bofa\b", "of a"),
    (r"\bNormalActivity\b", "Normal Activity"),
    (r"\bmeansentireand\b", "means entire and"),
    (r"\bwithregardto\b", "with regard to"),
    (r"\bexpressedhereon\b", "expressed hereon"),
    (r"\banypurposeinconnectionwiththe\b", "any purpose in connection with the"),
)

ASCII_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2022": "*",
        "\u00bb": "*",
        "\u00a0": " ",
    }
)


@dataclass
class OCRWord:
    text: str
    confidence: float
    bbox: list[int]
    source: str
    block_key: str
    paragraph_key: str
    line_key: str


@dataclass
class OCRLine:
    text: str
    confidence: float
    bbox: list[int]
    source: str
    block_key: str
    paragraph_key: str
    line_key: str
    words: list[OCRWord] = field(default_factory=list)


@dataclass
class OCRPage:
    lines: list[OCRLine]
    stats: dict


def resolve_tesseract_cmd(explicit_cmd: str | None = None) -> str:
    if explicit_cmd:
        path = Path(explicit_cmd)
        if not path.exists():
            raise FileNotFoundError(f"Tesseract executable not found: {explicit_cmd}")
        return str(path)

    discovered = shutil.which("tesseract")
    if discovered:
        return discovered

    for candidate in TESSERACT_FALLBACK_PATHS:
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "Tesseract is required. Install it and/or pass --tesseract-cmd."
    )


def normalize_text(text: str) -> str:
    text = html.unescape(text or "")
    text = _repair_mojibake(text)
    text = text.translate(ASCII_TRANSLATION)
    text = (
        text.replace("â€œ", '"')
        .replace("â€", '"')
        .replace("â€˜", "'")
        .replace("â€™", "'")
        .replace("â€“", "-")
        .replace("â€”", "-")
        .replace("Â", " ")
    )
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    for pattern, replacement in COMMON_OCR_REPAIRS:
        text = re.sub(pattern, replacement, text)
    return text


def _repair_mojibake(text: str) -> str:
    if not text or not any(token in text for token in ("â", "Ã", "Â")):
        return text

    suspicious_count = sum(text.count(token) for token in ("â", "Ã", "Â"))
    best = text
    best_count = suspicious_count

    for encoding in ("latin-1", "cp1252"):
        try:
            candidate = text.encode(encoding).decode("utf-8")
        except Exception:
            continue
        candidate_count = sum(candidate.count(token) for token in ("â", "Ã", "Â"))
        if candidate_count < best_count:
            best = candidate
            best_count = candidate_count

    return best


def join_tokens(tokens: Iterable[str]) -> str:
    parts: list[str] = []
    for token in tokens:
        token = normalize_text(token)
        if not token:
            continue
        if not parts:
            parts.append(token)
            continue

        if re.match(r"^[,.;:!?%)]", token):
            parts[-1] = parts[-1] + token
        elif parts[-1].endswith(("(", "/", "-", '"')):
            parts[-1] = parts[-1] + token
        else:
            parts.append(token)
    return " ".join(parts).strip()


def bbox_iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = float((ix2 - ix1) * (iy2 - iy1))
    area_a = max(1.0, float((ax2 - ax1) * (ay2 - ay1)))
    area_b = max(1.0, float((bx2 - bx1) * (by2 - by1)))
    return inter / (area_a + area_b - inter)


def line_height(bbox: list[int]) -> float:
    return max(1.0, float(bbox[3] - bbox[1]))


def centre_distance(a: list[int], b: list[int]) -> float:
    ax = (a[0] + a[2]) / 2.0
    ay = (a[1] + a[3]) / 2.0
    bx = (b[0] + b[2]) / 2.0
    by = (b[1] + b[3]) / 2.0
    return math.hypot(ax - bx, ay - by)


def text_quality_score(text: str, confidence: float) -> float:
    text = normalize_text(text)
    if not text:
        return -100.0

    alpha = sum(ch.isalpha() for ch in text)
    digits = sum(ch.isdigit() for ch in text)
    spaces = sum(ch.isspace() for ch in text)
    weird = sum(ch in "[]{}~|" for ch in text)
    density = (alpha + digits) / max(1, len(text))
    long_unbroken = max((len(token) for token in text.split()), default=0)

    score = confidence
    score += min(len(text), 80) * 0.15
    score += density * 6.0
    score += min(spaces, 8) * 0.4
    score -= weird * 1.5
    if long_unbroken >= 18:
        score -= 5.0
    if has_glued_fragments(text):
        score -= 6.0
    return score


def has_joined_words(text: str) -> bool:
    text = normalize_text(text)
    if not text:
        return False
    if re.search(r"[a-z][A-Z]", text):
        return True
    for token in text.split():
        if len(token) >= 18 and token.isalpha():
            return True
    return False


def has_glued_fragments(text: str) -> bool:
    tokens = re.findall(r"[A-Za-z]{4,}", normalize_text(text).lower())
    for token in tokens:
        if any(fragment in token for fragment in GLUED_FRAGMENT_PATTERNS):
            return True
    return False


def page_score(lines: list[OCRLine]) -> float:
    if not lines:
        return -100.0
    avg_conf = sum(line.confidence for line in lines) / len(lines)
    density = sum(min(len(line.text), 120) for line in lines) / max(1, len(lines))
    return avg_conf + density * 0.2 + min(len(lines), 120) * 0.08


class TesseractBackend:
    name = "tesseract"

    def __init__(self, language: str = "eng", tesseract_cmd: str | None = None):
        self.language = language
        self.tesseract_cmd = resolve_tesseract_cmd(tesseract_cmd)
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

    def recognize(
        self,
        image_rgb: np.ndarray,
        *,
        psm: int = 3,
        tag: str,
    ) -> list[OCRLine]:
        pil_image = Image.fromarray(image_rgb)
        config = (
            f"--oem 3 --psm {psm} "
            "-c preserve_interword_spaces=1 "
            "-c user_defined_dpi=300"
        )
        data = pytesseract.image_to_data(
            pil_image,
            lang=self.language,
            config=config,
            output_type=pytesseract.Output.DICT,
        )
        return self._dict_to_lines(data, tag)

    def _dict_to_lines(self, data: dict, tag: str) -> list[OCRLine]:
        groups: OrderedDict[tuple[int, int, int], dict] = OrderedDict()
        total = len(data.get("text", []))

        for idx in range(total):
            raw_text = data["text"][idx]
            text = normalize_text(str(raw_text))
            if not text:
                continue

            confidence = _safe_float(data["conf"][idx])
            if confidence < 0:
                continue

            left = int(data["left"][idx])
            top = int(data["top"][idx])
            width = int(data["width"][idx])
            height = int(data["height"][idx])
            if width <= 0 or height <= 0:
                continue

            block_num = int(data["block_num"][idx])
            par_num = int(data["par_num"][idx])
            line_num = int(data["line_num"][idx])
            key = (block_num, par_num, line_num)

            block_key = f"{tag}:block:{block_num}"
            paragraph_key = f"{tag}:paragraph:{block_num}.{par_num}"
            line_key = f"{tag}:line:{block_num}.{par_num}.{line_num}"

            word = OCRWord(
                text=text,
                confidence=confidence,
                bbox=[left, top, left + width, top + height],
                source=self.name,
                block_key=block_key,
                paragraph_key=paragraph_key,
                line_key=line_key,
            )

            group = groups.setdefault(
                key,
                {
                    "words": [],
                    "left": left,
                    "top": top,
                    "right": left + width,
                    "bottom": top + height,
                    "block_key": block_key,
                    "paragraph_key": paragraph_key,
                    "line_key": line_key,
                },
            )
            group["words"].append(word)
            group["left"] = min(group["left"], left)
            group["top"] = min(group["top"], top)
            group["right"] = max(group["right"], left + width)
            group["bottom"] = max(group["bottom"], top + height)

        lines: list[OCRLine] = []
        for group in groups.values():
            words = group["words"]
            text = join_tokens(word.text for word in words)
            if not text:
                continue
            confidence = float(
                sum(word.confidence for word in words) / max(1, len(words))
            )
            lines.append(
                OCRLine(
                    text=text,
                    confidence=confidence,
                    bbox=[
                        int(group["left"]),
                        int(group["top"]),
                        int(group["right"]),
                        int(group["bottom"]),
                    ],
                    source=self.name,
                    block_key=group["block_key"],
                    paragraph_key=group["paragraph_key"],
                    line_key=group["line_key"],
                    words=words,
                )
            )

        lines.sort(key=lambda line: (line.bbox[1], line.bbox[0]))
        return lines


class SuryaLineRefiner:
    name = "surya"

    def __init__(self, max_lines_per_page: int = 12):
        self.max_lines_per_page = max_lines_per_page
        self._recognizer = None

    def refine(self, image_rgb: np.ndarray, lines: list[OCRLine]) -> dict[str, tuple[str, float]]:
        candidates = self._select_candidates(lines)
        if not candidates:
            return {}

        recognizer = self._ensure_loaded()
        from surya.common.surya.schema import TaskNames

        pil_image = Image.fromarray(image_rgb)
        results = recognizer(
            [pil_image],
            task_names=[TaskNames.ocr_without_boxes],
            bboxes=[[line.bbox for line in candidates]],
            sort_lines=False,
            math_mode=False,
        )[0]

        refined: dict[str, tuple[str, float]] = {}
        for line, prediction in zip(candidates, results.text_lines):
            text = normalize_text(prediction.text)
            if not text:
                continue
            refined[line.line_key] = (text, float(prediction.confidence))
        return refined

    def _select_candidates(self, lines: list[OCRLine]) -> list[OCRLine]:
        ranked: list[tuple[float, OCRLine]] = []
        for line in lines:
            suspicion = max(0.0, 88.0 - line.confidence)
            if has_joined_words(line.text):
                suspicion += 12.0
            if has_glued_fragments(line.text):
                suspicion += 16.0
            if re.search(r"[~|}{\[\]]", line.text):
                suspicion += 10.0
            if re.search(r"\b[A-Za-z]{12,}\b", line.text):
                suspicion += 4.0
            if suspicion >= 10.0:
                ranked.append((suspicion, line))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [line for _, line in ranked[: self.max_lines_per_page]]

    def _ensure_loaded(self):
        if self._recognizer is None:
            from surya.foundation import FoundationPredictor
            from surya.recognition import RecognitionPredictor

            foundation = FoundationPredictor()
            recognizer = RecognitionPredictor(foundation)
            recognizer.disable_tqdm = True
            self._recognizer = recognizer
        return self._recognizer


class HybridOCR:
    def __init__(
        self,
        *,
        language: str = "eng",
        use_surya: bool = True,
        max_surya_lines: int = 12,
        tesseract_cmd: str | None = None,
    ):
        self.tesseract = TesseractBackend(language=language, tesseract_cmd=tesseract_cmd)
        self.surya = (
            SuryaLineRefiner(max_lines_per_page=max_surya_lines)
            if use_surya
            else None
        )

    def recognize_page(self, base_rgb: np.ndarray, binary_rgb: np.ndarray) -> OCRPage:
        primary_lines = self.tesseract.recognize(binary_rgb, psm=3, tag="psm3_binary")
        stats = {
            "primary_engine": "tesseract",
            "primary_pass": "psm3_binary",
            "primary_line_count": len(primary_lines),
            "primary_score": page_score(primary_lines),
        }

        if not primary_lines or stats["primary_score"] < 85:
            alt_lines = self.tesseract.recognize(base_rgb, psm=3, tag="psm3_base")
            if page_score(alt_lines) > page_score(primary_lines):
                primary_lines = alt_lines
                stats["primary_pass"] = "psm3_base"
            stats["alt_score"] = page_score(alt_lines)
            stats["primary_score"] = page_score(primary_lines)

        supplemental_lines = self.tesseract.recognize(binary_rgb, psm=11, tag="psm11_binary")
        merged_lines = self._merge_supplemental_lines(primary_lines, supplemental_lines)
        stats["supplemental_line_count"] = len(supplemental_lines)
        stats["merged_line_count"] = len(merged_lines)

        if self.surya is not None and merged_lines:
            refinements = self.surya.refine(base_rgb, merged_lines)
            merged_lines = self._apply_refinements(merged_lines, refinements)
            stats["surya_refinements"] = len(refinements)
        else:
            stats["surya_refinements"] = 0

        merged_lines.sort(key=lambda line: (line.bbox[1], line.bbox[0]))
        return OCRPage(lines=merged_lines, stats=stats)

    def _merge_supplemental_lines(
        self,
        primary_lines: list[OCRLine],
        supplemental_lines: list[OCRLine],
    ) -> list[OCRLine]:
        merged = list(primary_lines)

        for line in supplemental_lines:
            if text_quality_score(line.text, line.confidence) < 5.0:
                continue

            duplicate = False
            for existing in merged:
                same_box = bbox_iou(line.bbox, existing.bbox) >= 0.6
                near = centre_distance(line.bbox, existing.bbox) <= max(
                    14.0, line_height(existing.bbox) * 0.7
                )
                same_text = normalize_text(line.text).lower() == normalize_text(existing.text).lower()

                if same_box or (near and same_text):
                    duplicate = True
                    if text_quality_score(line.text, line.confidence) > text_quality_score(
                        existing.text, existing.confidence
                    ):
                        existing.text = line.text
                        existing.confidence = line.confidence
                        existing.bbox = line.bbox
                        existing.words = line.words
                        existing.source = line.source
                    break

            if not duplicate:
                merged.append(line)

        return merged

    def _apply_refinements(
        self,
        lines: list[OCRLine],
        refinements: dict[str, tuple[str, float]],
    ) -> list[OCRLine]:
        updated: list[OCRLine] = []
        for line in lines:
            refined = refinements.get(line.line_key)
            if not refined:
                updated.append(line)
                continue

            refined_text, refined_confidence = refined
            original_score = text_quality_score(line.text, line.confidence)
            refined_score = text_quality_score(refined_text, refined_confidence)

            should_replace = False
            if refined_score >= original_score + 1.0:
                should_replace = True
            elif line.confidence < 70 and refined_score >= original_score - 0.2:
                should_replace = True
            elif has_joined_words(line.text) and refined_score >= original_score - 1.0:
                should_replace = True
            elif has_glued_fragments(line.text) and refined_score >= original_score - 1.0:
                should_replace = True

            if should_replace:
                updated.append(
                    OCRLine(
                        text=refined_text,
                        confidence=refined_confidence,
                        bbox=line.bbox,
                        source="tesseract+surya",
                        block_key=line.block_key,
                        paragraph_key=line.paragraph_key,
                        line_key=line.line_key,
                        words=line.words,
                    )
                )
            else:
                updated.append(line)
        return updated


def _safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return -1.0
