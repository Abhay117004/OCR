# General Document Parser

This repository now contains a single document parsing system built around:

- native extraction for digital documents
- OCR for scanned documents and images
- multi-backend OCR using EasyOCR plus optional Surya
- reading-order reconstruction and form-style field extraction

The old Detectron2, LayoutParser, PaddleOCR, and LayoutLM pipeline has been removed.

## Supported Inputs

- Images: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`, `.webp`
- PDFs: native text when available, OCR fallback when needed
- DOCX: paragraphs and tables
- DOC: converted through Microsoft Word automation on Windows if Word is installed
- Text: `.txt`, `.md`

## Current Architecture

| Layer | File | Purpose |
|---|---|---|
| Entry point | `main.py` | CLI and output writing |
| Document loading | `document_loader.py` | Load images, PDFs, DOCX, DOC, TXT, MD |
| OCR backends | `ocr_backends.py` | EasyOCR backend, optional Surya backend, OCR result merging |
| Parsing | `document_parser.py` | Reading order, row grouping, field extraction, final text/markdown/json |

## Setup

### 1. Create and activate a virtual environment

**Windows PowerShell**
```powershell
python -m venv OCR
OCR\Scripts\Activate.ps1
```

**Linux / macOS**
```bash
python3 -m venv OCR
source OCR/bin/activate
```

### 2. Install PyTorch first

**CPU**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**CUDA 12.1**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install project dependencies

```bash
pip install -r requirements.txt
```

## First Run

The first run may download OCR model weights used by:

- EasyOCR
- Surya

After the first download, subsequent runs use the local cache.

## Usage

```bash
# default input/output
python main.py

# image
python main.py input/document.png --output output_final

# PDF
python main.py input/report.pdf --output parsed_pdf

# DOCX
python main.py input/letter.docx --output parsed_docx

# EasyOCR only
python main.py input/document.png --output output_easy --disable-surya
```

## Outputs

| File | Description |
|---|---|
| `document.txt` | Plain text extraction |
| `document.md` | Markdown rendering of extracted fields/text |
| `result.json` | Structured metadata, fields, lines, and page content |
| `layout_detection.png` | OCR line overlay for raster inputs |

## Notes

- `--disable-surya` is useful when you want the faster, more stable EasyOCR-only path.
- `.doc` support depends on Microsoft Word being installed on the machine.
- OCR quality will still vary by scan quality, handwriting, blur, skew, and template complexity.
- The parser is designed so stronger OCR backends or postprocessors can be added without changing the CLI.

## File Map

```text
main.py
document_loader.py
document_parser.py
ocr_backends.py
requirements.txt
input/
```
