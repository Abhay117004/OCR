from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import math
import re
from statistics import median

import cv2
import numpy as np

from document_loader import NativeDocument, RasterDocument, load_document
from image_preprocessing import preprocess_page
from ocr_backends import OCRLine, HybridOCR, normalize_text, text_quality_score


LIST_PATTERN = re.compile(
    r"^\s*(?:[-*\u2022]|[0-9]+[.)]|[A-Za-z][.)]|[ivxlcdmIVXLCDM]+[.)])\s+"
)
INLINE_SECTION_PATTERN = re.compile(
    r"(?:[A-Z][A-Za-z0-9&()/,'-]+(?: [A-Z][A-Za-z0-9&()/,'-]+){0,5}):"
)


@dataclass
class ParsedDocument:
    text: str
    markdown: str
    pages: list[dict]
    metadata: dict
    page_images: list[np.ndarray] = field(default_factory=list)


class DocumentParser:
    def __init__(
        self,
        *,
        use_surya: bool = True,
        pdf_scale: float = 4.0,
        max_pages: int | None = None,
        max_surya_lines: int = 20,
        tesseract_cmd: str | None = None,
    ):
        self.pdf_scale = pdf_scale
        self.max_pages = max_pages
        self.ocr = HybridOCR(
            use_surya=use_surya,
            max_surya_lines=max_surya_lines,
            tesseract_cmd=tesseract_cmd,
        )

    def parse(self, path: str | Path) -> ParsedDocument:
        loaded = load_document(path, pdf_scale=self.pdf_scale, max_pages=self.max_pages)
        if isinstance(loaded, NativeDocument):
            return self._parse_native(loaded)
        return self._parse_raster(loaded)

    def _parse_native(self, document: NativeDocument) -> ParsedDocument:
        text = document.text.strip()
        return ParsedDocument(
            text=text,
            markdown=text,
            pages=document.pages,
            metadata=document.metadata,
            page_images=[],
        )

    def _parse_raster(self, document: RasterDocument) -> ParsedDocument:
        metadata = dict(document.metadata)
        document_warnings: list[str] = []
        page_results: list[dict] = []
        page_images: list[np.ndarray] = []

        for page_index, image_rgb in enumerate(document.pages, start=1):
            preprocessed = preprocess_page(image_rgb)
            ocr_page = self.ocr.recognize_page(
                base_rgb=preprocessed.base_rgb,
                binary_rgb=preprocessed.binary_rgb,
            )
            page_result = self._build_page_result(
                page_index=page_index,
                image_rgb=preprocessed.base_rgb,
                lines=ocr_page.lines,
                stats={
                    **ocr_page.stats,
                    "deskew_angle": preprocessed.deskew_angle,
                    "scale_factor": preprocessed.scale_factor,
                },
                source_metadata=metadata,
            )
            page_results.append(page_result)
            page_images.append(preprocessed.base_rgb)

        page_results = self._cleanup_document_pages(page_results)

        page_markdown: list[str] = []
        page_text: list[str] = []
        for page_index, page_result in enumerate(page_results, start=1):
            page_text.append(page_result["text"])
            page_markdown.append(page_result["markdown"])
            for warning in page_result.get("warnings", []):
                document_warnings.append(f"Page {page_index}: {warning}")

        text = "\n\n".join(part for part in page_text if part).strip()
        markdown = "\n\n".join(part for part in page_markdown if part).strip()
        if document_warnings:
            metadata["warnings"] = document_warnings
            metadata["quality_status"] = "degraded"
        else:
            metadata["quality_status"] = "ok"
        return ParsedDocument(
            text=text,
            markdown=markdown,
            pages=page_results,
            metadata=metadata,
            page_images=page_images,
        )

    def _build_page_result(
        self,
        *,
        page_index: int,
        image_rgb: np.ndarray,
        lines: list[OCRLine],
        stats: dict,
        source_metadata: dict,
    ) -> dict:
        page_height, page_width = image_rgb.shape[:2]
        blocks = self._group_blocks(lines)
        column_split = self._detect_column_split(blocks, page_width, page_height)
        ordered_blocks = self._sort_blocks(blocks, column_split, page_width)
        median_line_height = self._median_line_height(lines)

        rendered_blocks: list[dict] = []
        for block in ordered_blocks:
            rendered = self._render_block(
                block=block,
                page_width=page_width,
                page_height=page_height,
                column_split=column_split,
                median_line_height=median_line_height,
            )
            rendered_blocks.append(rendered)
        rendered_blocks = self._merge_list_marker_blocks(
            rendered_blocks,
            page_width=page_width,
            median_line_height=median_line_height,
        )

        raw_text, raw_markdown = self._compose_page_output(
            page_index=page_index,
            blocks=rendered_blocks,
        )
        warnings = self._assess_page_quality(
            source_metadata=source_metadata,
            stats=stats,
            page_width=page_width,
            page_height=page_height,
            text=raw_text,
        )
        final_text = raw_text
        final_markdown = raw_markdown
        quality_status = "ok"
        if warnings:
            quality_status = "unusable"
            warning_intro = (
                f"OCR output for page {page_index} is too low quality to trust."
            )
            warning_details = " ".join(warnings)
            final_text = f"{warning_intro} {warning_details}".strip()
            final_markdown = f"## Page {page_index}\n\n> {final_text}"

        return {
            "page": page_index,
            "width": page_width,
            "height": page_height,
            "column_split": column_split,
            "stats": stats,
            "quality_status": quality_status,
            "warnings": warnings,
            "blocks": rendered_blocks,
            "text": final_text,
            "markdown": final_markdown,
            "raw_text": raw_text if warnings else "",
            "raw_markdown": raw_markdown if warnings else "",
        }

    def _assess_page_quality(
        self,
        *,
        source_metadata: dict,
        stats: dict,
        page_width: int,
        page_height: int,
        text: str,
    ) -> list[str]:
        warnings: list[str] = []
        primary_score = float(stats.get("primary_score", -100.0))
        merged_line_count = int(stats.get("merged_line_count", 0))
        scale_factor = float(stats.get("scale_factor", 1.0))
        text = normalize_text(text)

        original_width = int(source_metadata.get("original_width", page_width) or page_width)
        original_height = int(source_metadata.get("original_height", page_height) or page_height)
        original_pixels = original_width * original_height

        if source_metadata.get("source_type") == "image":
            if original_pixels < 400_000 or min(original_width, original_height) < 700:
                warnings.append(
                    f"Source image is only {original_width}x{original_height} pixels, which is too small for reliable document OCR."
                )
            elif original_pixels < 1_000_000 and primary_score < 55:
                warnings.append(
                    f"Source image is only {original_width}x{original_height} pixels and the OCR confidence is low."
                )

        if scale_factor >= 3.0 and primary_score < 50:
            warnings.append(
                f"The page had to be upscaled {scale_factor:.1f}x before OCR, which usually means the scan is too low resolution."
            )

        if primary_score < 35:
            warnings.append(
                f"The OCR quality score is {primary_score:.1f}, which is far below the trust threshold for document extraction."
            )
        elif primary_score < 45 and merged_line_count <= 12:
            warnings.append(
                f"The OCR quality score is {primary_score:.1f} and the page yielded too little reliable text."
            )

        if text:
            alpha = sum(ch.isalpha() for ch in text)
            weird = sum(ch in "[]{}~|@^_<>`" for ch in text)
            if alpha < max(25, len(text) * 0.35):
                warnings.append(
                    "The extracted text contains too little alphabetic content to trust as a readable document."
                )
            if weird > max(12, len(text) * 0.08):
                warnings.append(
                    "The extracted text is dominated by OCR artifacts and symbol noise."
                )
        else:
            warnings.append("The page did not yield readable text.")

        return _dedupe_messages(warnings)

    def _group_blocks(self, lines: list[OCRLine]) -> list[dict]:
        blocks: dict[str, dict] = {}
        for line in sorted(lines, key=lambda item: (item.bbox[1], item.bbox[0])):
            block = blocks.setdefault(
                line.block_key,
                {"id": line.block_key, "lines": []},
            )
            block["lines"].append(line)

        grouped: list[dict] = []
        for block in blocks.values():
            block_lines = sorted(block["lines"], key=lambda item: (item.bbox[1], item.bbox[0]))
            grouped.append(
                {
                    "id": block["id"],
                    "bbox": self._merge_bbox([line.bbox for line in block_lines]),
                    "lines": block_lines,
                }
            )
        return grouped

    def _detect_column_split(
        self,
        blocks: list[dict],
        page_width: int,
        page_height: int,
    ) -> int | None:
        candidates = [
            block
            for block in blocks
            if (
                page_width * 0.14
                <= (block["bbox"][2] - block["bbox"][0])
                < page_width * 0.72
            )
            and page_height * 0.1 <= block["bbox"][1] <= page_height * 0.82
            and sum(len(normalize_text(line.text)) for line in block["lines"]) >= 18
        ]
        if len(candidates) < 6:
            return None

        centres = sorted((block["bbox"][0] + block["bbox"][2]) / 2.0 for block in candidates)
        largest_gap = 0.0
        split = None
        for left, right in zip(centres, centres[1:]):
            gap = right - left
            if gap > largest_gap:
                largest_gap = gap
                split = int((left + right) / 2.0)

        if split is None or largest_gap < page_width * 0.14:
            return None

        left_count = sum(
            1 for block in candidates if ((block["bbox"][0] + block["bbox"][2]) / 2.0) < split
        )
        right_count = len(candidates) - left_count
        if min(left_count, right_count) < 3:
            return None

        left_blocks = [
            block for block in candidates
            if ((block["bbox"][0] + block["bbox"][2]) / 2.0) < split
        ]
        right_blocks = [block for block in candidates if block not in left_blocks]
        if not self._has_column_mass(left_blocks, page_width, page_height):
            return None
        if not self._has_column_mass(right_blocks, page_width, page_height):
            return None
        return split

    def _sort_blocks(self, blocks: list[dict], column_split: int | None, page_width: int) -> list[dict]:
        if column_split is None:
            for block in blocks:
                block["region"] = "main"
            return sorted(blocks, key=lambda block: (block["bbox"][1], block["bbox"][0]))

        left: list[dict] = []
        right: list[dict] = []
        wide: list[dict] = []
        split_margin = int(page_width * 0.05)

        for block in blocks:
            left_edge, _, right_edge, _ = block["bbox"]
            width = right_edge - left_edge
            centre = (left_edge + right_edge) / 2.0
            crosses_split = left_edge < (column_split - split_margin) and right_edge > (column_split + split_margin)
            if width > page_width * 0.82 or (
                crosses_split and abs(centre - column_split) <= page_width * 0.12
            ):
                block["region"] = "wide"
                wide.append(block)
            elif centre < column_split:
                block["region"] = "left"
                left.append(block)
            else:
                block["region"] = "right"
                right.append(block)

        substantive_columns = [
            block
            for block in left + right
            if len(block["lines"]) >= 3
            or sum(len(normalize_text(line.text)) for line in block["lines"]) >= 120
        ]
        body_top = min(
            (block["bbox"][1] for block in substantive_columns),
            default=10**9,
        )

        preface = [
            block for block in blocks
            if block["bbox"][1] < body_top
        ]
        body_left = [
            block for block in left
            if block["bbox"][1] >= body_top
        ]
        body_right = [
            block for block in right
            if block["bbox"][1] >= body_top
        ]
        body_wide = [
            block for block in wide
            if block["bbox"][1] >= body_top
        ]

        return (
            sorted(preface, key=lambda block: (block["bbox"][1], block["bbox"][0]))
            + sorted(body_left, key=lambda block: (block["bbox"][1], block["bbox"][0]))
            + sorted(body_right, key=lambda block: (block["bbox"][1], block["bbox"][0]))
            + sorted(body_wide, key=lambda block: (block["bbox"][1], block["bbox"][0]))
        )

    def _has_column_mass(
        self,
        blocks: list[dict],
        page_width: int,
        page_height: int,
    ) -> bool:
        substantive = [
            block
            for block in blocks
            if (
                (block["bbox"][2] - block["bbox"][0]) >= page_width * 0.18
                or sum(len(normalize_text(line.text)) for line in block["lines"]) >= 45
            )
        ]
        if len(substantive) < 2:
            return False

        text_mass = sum(
            sum(len(normalize_text(line.text)) for line in block["lines"])
            for block in substantive
        )
        top = min(block["bbox"][1] for block in substantive)
        bottom = max(block["bbox"][3] for block in substantive)
        vertical_span = bottom - top
        return text_mass >= 160 and vertical_span >= page_height * 0.18

    def _cleanup_document_pages(self, pages: list[dict]) -> list[dict]:
        if not pages:
            return pages

        self._mark_repeated_margin_text(pages)
        self._mark_low_quality_margin_artifacts(pages)

        for page in pages:
            if page.get("quality_status") == "unusable":
                continue
            text, markdown = self._compose_page_output(
                page_index=page["page"],
                blocks=page["blocks"],
            )
            page["text"] = text
            page["markdown"] = markdown

        return pages

    def _mark_repeated_margin_text(self, pages: list[dict]) -> None:
        groups: dict[tuple[str, str], list[tuple[dict, dict]]] = {}

        for page in pages:
            for block in page.get("blocks", []):
                zone = self._margin_zone(block, page["height"])
                if zone is None:
                    continue
                fingerprint = self._margin_text_fingerprint(block.get("text", ""))
                if len(fingerprint) < 20:
                    continue
                key = (zone, fingerprint)
                groups.setdefault(key, []).append((page, block))

        threshold = max(2, math.ceil(len(pages) * 0.5))
        for (zone, _), items in groups.items():
            if len(items) < threshold:
                continue
            for _, block in items:
                block["ignored"] = True
                block["ignore_reason"] = f"repeated_{zone}"

    def _mark_low_quality_margin_artifacts(self, pages: list[dict]) -> None:
        groups: dict[tuple[float, float, float, float], list[tuple[dict, dict]]] = {}

        for page in pages:
            for block in page.get("blocks", []):
                if self._margin_zone(block, page["height"]) != "top":
                    continue
                x1, y1, x2, y2 = block["bbox"]
                width = x2 - x1
                height = y2 - y1
                if width > page["width"] * 0.45 or height > page["height"] * 0.1:
                    continue
                key = (
                    round(x1 / max(1, page["width"]), 2),
                    round(y1 / max(1, page["height"]), 2),
                    round(x2 / max(1, page["width"]), 2),
                    round(y2 / max(1, page["height"]), 2),
                )
                groups.setdefault(key, []).append((page, block))

        for items in groups.values():
            if len(items) < 2:
                continue

            confidences = [self._block_confidence(block) for _, block in items]
            scores = [
                text_quality_score(block.get("text", ""), self._block_confidence(block))
                for _, block in items
            ]
            texts = [normalize_text(block.get("text", "")) for _, block in items]
            if (
                max(len(text) for text in texts) <= 32
                and sum(confidences) / len(confidences) < 55
                and sum(scores) / len(scores) < 45
            ):
                for _, block in items:
                    block["ignored"] = True
                    block["ignore_reason"] = "margin_artifact"

    def _compose_page_output(self, *, page_index: int, blocks: list[dict]) -> tuple[str, str]:
        kept_blocks = [block for block in blocks if not block.get("ignored")]
        text_parts = [
            block.get("text", "").strip()
            for block in kept_blocks
            if block.get("text", "").strip()
        ]
        markdown_parts = [f"## Page {page_index}"]
        for block in kept_blocks:
            markdown = block.get("markdown", "").strip()
            if markdown:
                markdown_parts.append(markdown)
        return (
            "\n\n".join(text_parts).strip(),
            "\n\n".join(part for part in markdown_parts if part).strip(),
        )

    def _merge_list_marker_blocks(
        self,
        blocks: list[dict],
        *,
        page_width: int,
        median_line_height: float,
    ) -> list[dict]:
        merged: list[dict] = []
        index = 0
        while index < len(blocks):
            block = blocks[index]

            if merged and self._is_orphan_marker(block, merged[-1], page_width, median_line_height):
                index += 1
                continue

            marker = self._list_marker_value(block, page_width)
            next_block = blocks[index + 1] if index + 1 < len(blocks) else None
            if marker and next_block and self._should_attach_marker(
                marker_block=block,
                content_block=next_block,
                page_width=page_width,
                median_line_height=median_line_height,
            ):
                merged.append(self._attach_list_marker(marker, block, next_block))
                index += 2
                continue

            merged.append(block)
            index += 1

        return merged

    def _list_marker_value(self, block: dict, page_width: int) -> str | None:
        text = normalize_text(block.get("text", ""))
        if not text or len(text) > 3:
            return None

        x1, _, x2, _ = block["bbox"]
        width = x2 - x1
        if width > page_width * 0.06:
            return None

        if text in {"e", ">", "o", "0", "*", "-", "•"}:
            return "-"
        if re.fullmatch(r"[0-9]+[.)]", text):
            return text
        if re.fullmatch(r"[A-Za-z][.)]", text):
            return text.lower()
        if re.fullmatch(r"[ivxlcdmIVXLCDM]+[.)]", text):
            return text.lower()
        return None

    def _should_attach_marker(
        self,
        *,
        marker_block: dict,
        content_block: dict,
        page_width: int,
        median_line_height: float,
    ) -> bool:
        marker_x1, marker_y1, marker_x2, marker_y2 = marker_block["bbox"]
        content_x1, content_y1, content_x2, content_y2 = content_block["bbox"]
        marker_mid_y = (marker_y1 + marker_y2) / 2.0
        content_overlap = content_y1 <= marker_mid_y <= content_y2
        close_row = abs(content_y1 - marker_y1) <= max(20.0, median_line_height * 2.5)
        close_gap = 0 <= (content_x1 - marker_x2) <= page_width * 0.09
        return (
            close_gap
            and (content_overlap or close_row)
            and len(normalize_text(content_block.get("text", ""))) >= 6
            and content_x2 > marker_x2
        )

    def _is_orphan_marker(
        self,
        block: dict,
        previous_block: dict,
        page_width: int,
        median_line_height: float,
    ) -> bool:
        marker_value = self._list_marker_value(block, page_width)
        if marker_value is None:
            return False

        _, y1, _, y2 = block["bbox"]
        prev_x1, prev_y1, _, prev_y2 = previous_block["bbox"]
        marker_mid_y = (y1 + y2) / 2.0
        if previous_block.get("kind") == "list":
            return (
                abs(block["bbox"][0] - prev_x1) <= page_width * 0.05
                and prev_y1 - median_line_height <= marker_mid_y <= prev_y2 + median_line_height
            )

        previous_text = normalize_text(previous_block.get("text", ""))
        return (
            abs(block["bbox"][0] - prev_x1) <= page_width * 0.05
            and prev_y1 <= marker_mid_y <= prev_y2
            and len(previous_text) >= 40
            and marker_value == "-"
        )

    def _attach_list_marker(self, marker: str, marker_block: dict, content_block: dict) -> dict:
        marker_text = self._marker_text(marker)
        content_text = normalize_text(content_block.get("text", ""))
        text = f"{marker_text} {content_text}".strip()
        merged = dict(content_block)
        merged["bbox"] = self._merge_bbox([marker_block["bbox"], content_block["bbox"]])
        merged["kind"] = "list"
        merged["text"] = text
        merged["markdown"] = self._list_markdown_text(marker_text, content_text)
        merged["lines"] = marker_block.get("lines", []) + content_block.get("lines", [])
        merged["list_marker"] = marker_text
        return merged

    def _marker_text(self, marker: str) -> str:
        if marker == "-":
            return "-"
        return marker

    def _list_markdown_text(self, marker_text: str, content_text: str) -> str:
        if marker_text == "-":
            return f"- {content_text}"
        if re.fullmatch(r"([0-9]+)[)]", marker_text):
            return re.sub(r"([0-9]+)[)]", r"\1.", marker_text) + f" {content_text}"
        if re.fullmatch(r"([A-Za-z])[)]", marker_text):
            return re.sub(r"([A-Za-z])[)]", r"\1.", marker_text) + f" {content_text}"
        return f"{marker_text} {content_text}".strip()

    def _render_block(
        self,
        *,
        block: dict,
        page_width: int,
        page_height: int,
        column_split: int | None,
        median_line_height: float,
    ) -> dict:
        lines = block["lines"]
        table_rows = self._detect_table(lines, page_width)

        if table_rows is not None:
            kind = "table"
            text = self._table_to_text(table_rows)
            markdown = self._table_to_markdown(table_rows)
        else:
            paragraphs = self._merge_paragraphs(lines)
            text = "\n\n".join(paragraphs).strip()
            kind = self._infer_block_kind(
                block,
                paragraphs,
                median_line_height=median_line_height,
                page_height=page_height,
            )
            markdown = self._block_to_markdown(kind, text, lines, median_line_height)

        return {
            "id": block["id"],
            "region": block.get("region", "main"),
            "bbox": [int(value) for value in block["bbox"]],
            "kind": kind,
            "text": text,
            "markdown": markdown,
            "table_rows": table_rows or [],
            "lines": [self._serialize_line(line) for line in lines],
            "column_split": column_split,
        }

    def _detect_table(self, lines: list[OCRLine], page_width: int) -> list[list[str]] | None:
        row_cells: list[list[dict]] = []
        for line in lines:
            cells = self._split_line_into_cells(line)
            if len(cells) >= 2:
                row_cells.append(cells)

        if len(row_cells) < 3:
            return None

        tolerance = max(28.0, page_width * 0.018)
        column_starts: list[float] = []
        for row in row_cells:
            for cell in row:
                start = float(cell["bbox"][0])
                nearest = self._nearest_index(column_starts, start, tolerance)
                if nearest is None:
                    column_starts.append(start)
                    column_starts.sort()

        if len(column_starts) < 2 or len(column_starts) > 6:
            return None

        aligned_rows = 0
        rows: list[list[str]] = []
        for row in row_cells:
            values = [""] * len(column_starts)
            filled = 0
            for cell in row:
                nearest = self._nearest_index(column_starts, float(cell["bbox"][0]), tolerance)
                if nearest is None or values[nearest]:
                    continue
                values[nearest] = cell["text"]
                filled += 1
            if filled >= 2:
                aligned_rows += 1
            rows.append(values)

        if aligned_rows < 3:
            return None
        return rows

    def _split_line_into_cells(self, line: OCRLine) -> list[dict]:
        if not line.words:
            return []

        words = sorted(line.words, key=lambda word: word.bbox[0])
        gap_threshold = max(36.0, median((word.bbox[3] - word.bbox[1]) for word in words) * 2.6)

        cells: list[list] = []
        current: list = []
        prev_right = None

        for word in words:
            if prev_right is not None and (word.bbox[0] - prev_right) > gap_threshold:
                if current:
                    cells.append(current)
                current = [word]
            else:
                current.append(word)
            prev_right = word.bbox[2]

        if current:
            cells.append(current)

        cell_results: list[dict] = []
        for cell_words in cells:
            text = " ".join(word.text for word in cell_words).strip()
            if not text:
                continue
            cell_results.append(
                {
                    "text": text,
                    "bbox": self._merge_bbox([word.bbox for word in cell_words]),
                }
            )
        return cell_results

    def _merge_paragraphs(self, lines: list[OCRLine]) -> list[str]:
        ordered = sorted(lines, key=lambda line: (line.bbox[1], line.bbox[0]))
        if not ordered:
            return []

        heights = [line.bbox[3] - line.bbox[1] for line in ordered]
        median_height = median(heights) if heights else 24.0

        paragraphs: list[str] = []
        current_lines: list[str] = []
        previous_line: OCRLine | None = None

        for line in ordered:
            text = normalize_text(line.text)
            if not text:
                continue

            if previous_line is None:
                current_lines = [text]
                previous_line = line
                continue

            gap = line.bbox[1] - previous_line.bbox[3]
            indent_shift = abs(line.bbox[0] - previous_line.bbox[0])
            previous_text = current_lines[-1] if current_lines else ""

            starts_list = bool(LIST_PATTERN.match(text))
            ends_list = bool(LIST_PATTERN.match(previous_text))

            new_paragraph = (
                gap > median_height * 1.3
                or (indent_shift > median_height * 1.8 and len(previous_text) > 35)
                or starts_list
                or ends_list
            )

            if new_paragraph:
                paragraph = self._join_paragraph_lines(current_lines)
                if paragraph:
                    paragraphs.append(paragraph)
                current_lines = [text]
            else:
                current_lines.append(text)

            previous_line = line

        paragraph = self._join_paragraph_lines(current_lines)
        if paragraph:
            paragraphs.append(paragraph)
        expanded: list[str] = []
        for paragraph in paragraphs:
            expanded.extend(self._split_embedded_sections(paragraph))
        return expanded

    def _join_paragraph_lines(self, lines: list[str]) -> str:
        if not lines:
            return ""

        paragraph = lines[0]
        for line in lines[1:]:
            if paragraph.endswith("-") and line[:1].islower():
                paragraph = paragraph[:-1] + line
            else:
                paragraph = f"{paragraph} {line}"
        return normalize_text(paragraph)

    def _infer_block_kind(
        self,
        block: dict,
        paragraphs: list[str],
        *,
        median_line_height: float,
        page_height: int,
    ) -> str:
        text = " ".join(paragraphs).strip()
        if not text:
            return "empty"

        if paragraphs and all(LIST_PATTERN.match(part) for part in paragraphs[: min(3, len(paragraphs))]):
            return "list"

        heights = [line.bbox[3] - line.bbox[1] for line in block["lines"]]
        block_height = median(heights) if heights else median_line_height
        word_count = len(text.split())
        uppercase_ratio = sum(ch.isupper() for ch in text) / max(1, sum(ch.isalpha() for ch in text))

        if (
            len(paragraphs) == 1
            and word_count <= 8
            and not text.endswith(".")
            and not text.endswith(";")
            and not re.match(r"^(on|and|or|the|for|to|of)\b", text.lower())
            and (
                text.endswith(":")
                or word_count <= 3
                or block_height >= median_line_height * 1.05
                or uppercase_ratio > 0.45
            )
        ):
            return "heading"

        if (
            word_count <= 18
            and (
                block_height >= median_line_height * 1.35
                or uppercase_ratio > 0.75
                or block["bbox"][1] < page_height * 0.2
            )
        ):
            return "heading"

        return "paragraph"

    def _block_to_markdown(
        self,
        kind: str,
        text: str,
        lines: list[OCRLine],
        median_line_height: float,
    ) -> str:
        if not text:
            return ""

        if kind == "list":
            if re.match(r"^[0-9]+[.)]\s+", text):
                return re.sub(r"^([0-9]+)[)]\s+", r"\1. ", text)
            if re.match(r"^[A-Za-z][.)]\s+", text):
                return re.sub(r"^([A-Za-z])[)]\s+", r"\1. ", text)
            return text if text.startswith("- ") else f"- {text.lstrip('- ').strip()}"

        if kind == "heading":
            heights = [line.bbox[3] - line.bbox[1] for line in lines]
            block_height = median(heights) if heights else median_line_height
            level = 1 if block_height >= median_line_height * 1.75 else 2
            return f"{'#' * level} {text}"
        return text

    def _split_embedded_sections(self, paragraph: str) -> list[str]:
        paragraph = normalize_text(paragraph)
        matches = list(INLINE_SECTION_PATTERN.finditer(paragraph))
        if len(matches) < 2:
            return [paragraph] if paragraph else []

        parts: list[str] = []
        start = 0
        for match in matches[1:]:
            part = paragraph[start:match.start()].strip()
            if part:
                parts.append(part)
            start = match.start()
        tail = paragraph[start:].strip()
        if tail:
            parts.append(tail)
        return parts or ([paragraph] if paragraph else [])

    def _table_to_markdown(self, rows: list[list[str]]) -> str:
        if not rows:
            return ""
        header = rows[0]
        body = rows[1:] if len(rows) > 1 else []
        separator = ["---"] * len(header)
        all_rows = [header, separator] + body
        return "\n".join(
            "| " + " | ".join(cell.strip() for cell in row) + " |"
            for row in all_rows
        )

    def _table_to_text(self, rows: list[list[str]]) -> str:
        if not rows:
            return ""
        widths = [0] * len(rows[0])
        for row in rows:
            for index, cell in enumerate(row):
                widths[index] = max(widths[index], len(cell))

        rendered_rows = []
        for row in rows:
            rendered_rows.append(
                "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)).rstrip()
            )
        return "\n".join(rendered_rows)

    def _serialize_line(self, line: OCRLine) -> dict:
        return {
            "text": line.text,
            "confidence": round(float(line.confidence), 3),
            "bbox": [int(value) for value in line.bbox],
            "source": line.source,
            "words": [
                {
                    "text": word.text,
                    "confidence": round(float(word.confidence), 3),
                    "bbox": [int(value) for value in word.bbox],
                }
                for word in line.words
            ],
        }

    def _margin_zone(self, block: dict, page_height: int) -> str | None:
        _, y1, _, y2 = block["bbox"]
        if y2 <= page_height * 0.14:
            return "top"
        if y1 >= page_height * 0.86:
            return "bottom"
        return None

    def _margin_text_fingerprint(self, text: str) -> str:
        text = normalize_text(text).lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^a-z0-9 ]+", "", text)
        return text.strip()

    def _block_confidence(self, block: dict) -> float:
        lines = block.get("lines", [])
        if not lines:
            return 0.0
        confidences = [float(line.get("confidence", 0.0)) for line in lines]
        return sum(confidences) / max(1, len(confidences))

    @staticmethod
    def _merge_bbox(bboxes: list[list[int]]) -> list[int]:
        return [
            int(min(bbox[0] for bbox in bboxes)),
            int(min(bbox[1] for bbox in bboxes)),
            int(max(bbox[2] for bbox in bboxes)),
            int(max(bbox[3] for bbox in bboxes)),
        ]

    @staticmethod
    def _median_line_height(lines: list[OCRLine]) -> float:
        if not lines:
            return 24.0
        return float(median(line.bbox[3] - line.bbox[1] for line in lines))

    @staticmethod
    def _nearest_index(values: list[float], target: float, tolerance: float) -> int | None:
        if not values:
            return None
        nearest = min(range(len(values)), key=lambda index: abs(values[index] - target))
        if abs(values[nearest] - target) <= tolerance:
            return nearest
        return None


def write_outputs(result: ParsedDocument, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    pages_dir = output_dir / "pages"
    overlays_dir = output_dir / "overlays"

    output_dir.mkdir(parents=True, exist_ok=True)
    if result.page_images:
        pages_dir.mkdir(parents=True, exist_ok=True)
        overlays_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "document.txt").write_text(result.text, encoding="utf-8")
    (output_dir / "document.md").write_text(result.markdown, encoding="utf-8")
    (output_dir / "result.json").write_text(
        json.dumps(
            {
                "metadata": result.metadata,
                "text": result.text,
                "markdown": result.markdown,
                "pages": result.pages,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    for page_index, image_rgb in enumerate(result.page_images, start=1):
        page = result.pages[page_index - 1]
        page_path = pages_dir / f"page_{page_index:03d}.png"
        overlay_path = overlays_dir / f"page_{page_index:03d}_overlay.png"
        _write_rgb_image(page_path, image_rgb)
        _write_rgb_image(overlay_path, _draw_overlay(image_rgb, page))


def _draw_overlay(image_rgb: np.ndarray, page: dict) -> np.ndarray:
    image_bgr = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    block_colors = {
        "heading": (42, 82, 190),
        "paragraph": (40, 160, 40),
        "list": (0, 170, 170),
        "table": (200, 120, 30),
        "empty": (180, 180, 180),
    }

    for block in page.get("blocks", []):
        if block.get("ignored"):
            continue
        x1, y1, x2, y2 = block["bbox"]
        kind = block.get("kind", "paragraph")
        color = block_colors.get(kind, (120, 120, 120))
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
        label = f"{block.get('region', 'main')}:{kind}"
        cv2.putText(
            image_bgr,
            label[:36],
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )

        for line in block.get("lines", []):
            lx1, ly1, lx2, ly2 = line["bbox"]
            cv2.rectangle(image_bgr, (lx1, ly1), (lx2, ly2), (235, 235, 235), 1)

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _write_rgb_image(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), image_bgr)


def _dedupe_messages(messages: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for message in messages:
        if not message or message in seen:
            continue
        deduped.append(message)
        seen.add(message)
    return deduped
