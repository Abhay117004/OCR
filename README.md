# Scanned Document Extractor

This repository is a practical OCR pipeline for scanned PDFs and document images.

It is built around four ideas:

- rasterize every PDF page at high resolution
- normalize the page before OCR
- use Tesseract for page structure and reading order
- use Surya to repair the lines that Tesseract is most likely to get wrong

The goal is not just to dump text, but to preserve the document's structure well enough to produce readable text, markdown, per-page JSON, page images, and layout overlays.

## What It Supports

- scanned PDFs
- document images: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`, `.webp`
- plain text / markdown passthrough: `.txt`, `.md`

## Current Architecture

| Layer | File | Role |
|---|---|---|
| CLI | `main.py` | Processes a file or folder and writes one output folder per input file |
| Loading | `document_loader.py` | Rasterizes PDFs and loads images / text inputs |
| Preprocessing | `image_preprocessing.py` | Upscaling, denoising, contrast normalization, deskewing, binarization |
| OCR | `ocr_backends.py` | Tesseract page OCR + optional Surya line refinement |
| Reconstruction | `document_parser.py` | Block grouping, column ordering, paragraph rebuilding, markdown / JSON output |

## Setup

### 1. Create a virtual environment

```powershell
python -m venv OCR
OCR\Scripts\Activate.ps1
```

### 2. Install PyTorch first

CPU:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

CUDA 12.1:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Python dependencies

```powershell
pip install -r requirements.txt
```

### 4. Install Tesseract OCR

Windows with `winget`:

```powershell
winget install -e --id tesseract-ocr.tesseract --accept-package-agreements --accept-source-agreements
```

If `tesseract.exe` is not on `PATH`, pass it explicitly:

```powershell
python main.py --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

## Usage

Put files in `input/` and run:

```powershell
python main.py
```

Or point it at a specific file:

```powershell
python main.py "input\scan.pdf"
```

Useful flags:

```powershell
python main.py "input\scan.pdf" --max-pages 3
python main.py "input\scan.pdf" --disable-surya
python main.py "input\scan.pdf" --output output_run
python main.py "input\scan.pdf" --pdf-scale 4.5
```

## Output Layout

Each input file gets its own folder under `output/`:

```text
output/
  some_document/
    document.txt
    document.md
    result.json
    pages/
      page_001.png
    overlays/
      page_001_overlay.png
```

## Notes

- `--disable-surya` keeps the pipeline much faster. Use it when you want structure-first OCR without line repair.
- The first Surya run may download model weights.
- PDFs are always rasterized in this version, which keeps the OCR path consistent for scanned documents.
- The parser uses deterministic heuristics for columns, headings, lists, and simple table-like rows. It is practical, but not perfect.

## Verified Locally

The current pipeline was run against:

- `input/Private_Car_Long_Term_Policy_Policy_Wording_new_01_fb033caacd.pdf`

The strongest results on the sample came from:

- PDF rasterization at `--pdf-scale 4.0`
- Tesseract for page segmentation and reading order
- optional Surya repair for suspicious lines
