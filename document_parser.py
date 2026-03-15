from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re

import cv2

from document_loader import NativeDocument, RasterDocument, load_document
from ocr_backends import EasyOCRBackend, EnsembleOCR, OCRLine, SuryaBackend


@dataclass
class ParsedDocument:
    text: str
    markdown: str
    lines: list[dict]
    fields: dict
    pages: list[dict]
    metadata: dict


class DocumentParser:
    def __init__(self, lang: str = "en", use_surya: bool = True):
        backends = [EasyOCRBackend([lang])]
        if use_surya:
            try:
                backends.append(SuryaBackend([lang]))
            except Exception as exc:
                print(f"[OCR] Surya disabled: {exc}")
        self.ocr = EnsembleOCR(backends)

    def parse(self, path: str | Path) -> ParsedDocument:
        loaded = load_document(path)
        if isinstance(loaded, NativeDocument):
            return self._parse_native(loaded)
        return self._parse_raster(loaded)

    def _parse_native(self, document: NativeDocument) -> ParsedDocument:
        text = document.text.strip()
        markdown = text
        return ParsedDocument(
            text=text,
            markdown=markdown,
            lines=[],
            fields={},
            pages=document.pages,
            metadata=document.metadata,
        )

    def _parse_raster(self, document: RasterDocument) -> ParsedDocument:
        all_pages: list[dict] = []
        all_text_parts: list[str] = []
        all_lines: list[dict] = []
        all_fields: dict = {}

        for page_index, image_rgb in enumerate(document.pages, start=1):
            lines = self.ocr.recognize(image_rgb)
            ordered = self._sort_lines(lines, image_rgb.shape[1])
            grouped = self._group_rows(ordered)
            fields = self._extract_fields(grouped)
            text = self._rows_to_text(grouped)

            all_fields.update({f"page_{page_index}.{k}": v for k, v in fields.items()})
            all_text_parts.append(text)
            all_lines.extend(
                [
                    {
                        "page": page_index,
                        "text": line.text,
                        "confidence": line.confidence,
                        "bbox": line.bbox,
                        "source": line.source,
                    }
                    for line in ordered
                ]
            )
            all_pages.append(
                {
                    "page": page_index,
                    "rows": [
                        [
                            {
                                "text": line.text,
                                "confidence": line.confidence,
                                "bbox": line.bbox,
                                "source": line.source,
                            }
                            for line in row
                        ]
                        for row in grouped
                    ],
                    "fields": fields,
                    "text": text,
                }
            )

        text = "\n\n".join(part.strip() for part in all_text_parts if part.strip())
        markdown = self._to_markdown(all_pages)
        return ParsedDocument(
            text=text,
            markdown=markdown,
            lines=all_lines,
            fields=all_fields,
            pages=all_pages,
            metadata=document.metadata,
        )

    def _sort_lines(self, lines: list[OCRLine], page_width: int) -> list[OCRLine]:
        if not lines:
            return []
        columns = self._detect_columns(lines, page_width)
        sorted_lines: list[OCRLine] = []
        for left, right in columns:
            column_lines = [
                line for line in lines
                if self._line_centre_x(line) >= left and self._line_centre_x(line) <= right
            ]
            column_lines.sort(key=lambda line: (line.bbox[1], line.bbox[0]))
            sorted_lines.extend(column_lines)
        return sorted_lines

    def _detect_columns(self, lines: list[OCRLine], page_width: int) -> list[tuple[int, int]]:
        if len(lines) < 8:
            return [(0, page_width)]
        centres = sorted(int(self._line_centre_x(line)) for line in lines)
        largest_gap = 0
        split = None
        for left, right in zip(centres, centres[1:]):
            gap = right - left
            if gap > largest_gap:
                largest_gap = gap
                split = (left + right) // 2
        if split is None or largest_gap < page_width * 0.12:
            return [(0, page_width)]

        left_count = sum(1 for line in lines if self._line_centre_x(line) < split)
        right_count = len(lines) - left_count
        if min(left_count, right_count) < max(3, len(lines) * 0.15):
            return [(0, page_width)]
        return [(0, split), (split + 1, page_width)]

    def _group_rows(self, lines: list[OCRLine]) -> list[list[OCRLine]]:
        rows: list[list[OCRLine]] = []
        for line in lines:
            if not rows:
                rows.append([line])
                continue
            prev_row = rows[-1]
            prev_mid_y = sum((item.bbox[1] + item.bbox[3]) / 2 for item in prev_row) / len(prev_row)
            line_mid_y = (line.bbox[1] + line.bbox[3]) / 2
            threshold = max(12.0, self._median_height(prev_row) * 0.9)
            if abs(line_mid_y - prev_mid_y) <= threshold:
                prev_row.append(line)
                prev_row.sort(key=lambda item: item.bbox[0])
            else:
                rows.append([line])
        return rows

    def _extract_fields(self, rows: list[list[OCRLine]]) -> dict:
        fields: dict[str, str] = {}
        fallback_index = 1
        for row in rows:
            texts = [line.text.strip() for line in row if line.text.strip()]
            if not texts:
                continue

            joined = " ".join(texts)
            if ":" in joined:
                left, right = joined.split(":", 1)
                key = self._normalize_key(left)
                value = self._clean_value(right)
                if key and value:
                    fields[key] = value
                    continue

            if len(texts) >= 2 and len(texts[0]) <= 40:
                key = self._normalize_key(texts[0])
                value = self._clean_value(" ".join(texts[1:]))
                if key and value:
                    fields[key] = value
                    continue

            if len(joined) > 20:
                fields[f"line_{fallback_index:03d}"] = self._clean_value(joined)
                fallback_index += 1
        return fields

    def _rows_to_text(self, rows: list[list[OCRLine]]) -> str:
        paragraphs: list[str] = []
        for row in rows:
            texts = [self._clean_value(line.text) for line in row if line.text.strip()]
            if not texts:
                continue
            if len(texts) >= 2 and (texts[0].endswith(":") or len(texts[0]) <= 40):
                paragraph = f"{texts[0]} {' '.join(texts[1:])}".strip()
            else:
                paragraph = " ".join(texts).strip()
            if paragraph:
                paragraphs.append(paragraph)
        return "\n".join(paragraphs)

    def _to_markdown(self, pages: list[dict]) -> str:
        parts: list[str] = []
        for page in pages:
            parts.append(f"## Page {page['page']}")
            if page["fields"]:
                for key, value in page["fields"].items():
                    parts.append(f"- **{key}**: {value}")
            elif page["text"]:
                parts.append(page["text"])
            parts.append("")
        return "\n".join(parts).strip()

    @staticmethod
    def _line_centre_x(line: OCRLine) -> float:
        return (line.bbox[0] + line.bbox[2]) / 2

    @staticmethod
    def _median_height(row: list[OCRLine]) -> float:
        heights = sorted((line.bbox[3] - line.bbox[1]) for line in row)
        return float(heights[len(heights) // 2]) if heights else 12.0

    @staticmethod
    def _normalize_key(text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip(" :-|")
        text = text.lower()
        text = text.replace("/", " ")
        text = re.sub(r"[^a-z0-9 ]+", "", text)
        return re.sub(r"\s+", "_", text).strip("_")

    @staticmethod
    def _clean_value(text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip(" |:-")
        text = text.replace(" ,", ",")
        return text


def write_outputs(result: ParsedDocument, output_dir: str | Path) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "document.txt").write_text(result.text, encoding="utf-8")
    (out_dir / "document.md").write_text(result.markdown, encoding="utf-8")
    (out_dir / "result.json").write_text(
        json.dumps(
            {
                "metadata": result.metadata,
                "fields": result.fields,
                "lines": result.lines,
                "pages": result.pages,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    if result.pages and result.lines:
        image_path = result.metadata.get("path")
        if image_path and Path(image_path).suffix.lower() in {
            ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"
        }:
            image_bgr = cv2.imread(image_path)
            if image_bgr is not None:
                for line in result.lines:
                    x1, y1, x2, y2 = line["bbox"]
                    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 180, 0), 2)
                cv2.imwrite(str(out_dir / "layout_detection.png"), image_bgr)
