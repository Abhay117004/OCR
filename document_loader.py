from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pypdfium2 as pdfium


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
TEXT_SUFFIXES = {".txt", ".md"}
SUPPORTED_SUFFIXES = IMAGE_SUFFIXES | TEXT_SUFFIXES | {".pdf"}


@dataclass
class NativeDocument:
    text: str
    pages: list[dict]
    metadata: dict


@dataclass
class RasterDocument:
    pages: list[np.ndarray]
    metadata: dict


def iter_input_files(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported input file: {path}")
        return [path]

    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")

    if not path.is_dir():
        raise ValueError(f"Input path must be a file or folder: {path}")

    files = [
        item
        for item in sorted(path.iterdir())
        if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return files


def load_document(
    path: str | Path,
    pdf_scale: float = 4.0,
    max_pages: int | None = None,
) -> NativeDocument | RasterDocument:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in IMAGE_SUFFIXES:
        image_bgr = cv2.imread(str(path))
        if image_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return RasterDocument(
            pages=[image_rgb],
            metadata={
                "source_type": "image",
                "path": str(path),
                "page_count": 1,
                "original_width": int(image_rgb.shape[1]),
                "original_height": int(image_rgb.shape[0]),
            },
        )

    if suffix == ".pdf":
        return _load_pdf(path, pdf_scale=pdf_scale, max_pages=max_pages)

    if suffix in TEXT_SUFFIXES:
        text = path.read_text(encoding="utf-8")
        return NativeDocument(
            text=text,
            pages=[{"page": 1, "text": text}],
            metadata={"source_type": "text", "path": str(path), "page_count": 1},
        )

    raise ValueError(
        f"Unsupported file type: {suffix}. Supported: PDF, images, TXT, MD."
    )


def _load_pdf(
    path: Path,
    pdf_scale: float = 4.0,
    max_pages: int | None = None,
) -> RasterDocument:
    pdf_doc = pdfium.PdfDocument(str(path))
    page_total = len(pdf_doc)
    if max_pages is not None:
        page_total = min(page_total, max_pages)

    pages: list[np.ndarray] = []
    for page_index in range(page_total):
        bitmap = pdf_doc[page_index].render(scale=pdf_scale).to_pil()
        pages.append(np.array(bitmap.convert("RGB")))

    return RasterDocument(
        pages=pages,
        metadata={
            "source_type": "pdf-raster",
            "path": str(path),
            "page_count": page_total,
            "pdf_scale": pdf_scale,
        },
    )
