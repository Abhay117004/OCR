from __future__ import annotations

import argparse
from pathlib import Path
import re

from document_loader import iter_input_files
from document_parser import DocumentParser, write_outputs


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_root = Path(args.output)

    try:
        files = iter_input_files(input_path)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return

    if not files:
        print(f"ERROR: No supported input files found in {input_path}")
        return

    parser = DocumentParser(
        use_surya=not args.disable_surya,
        pdf_scale=args.pdf_scale,
        max_pages=args.max_pages,
        max_surya_lines=args.max_surya_lines,
        tesseract_cmd=args.tesseract_cmd,
    )

    print(f"\nInput path     : {input_path.resolve()}")
    print(f"Output root    : {output_root.resolve()}")
    print(f"Use Surya      : {not args.disable_surya}")
    print(f"PDF scale      : {args.pdf_scale}")
    if args.max_pages is not None:
        print(f"Max pages/file : {args.max_pages}")
    print("")

    used_names: set[str] = set()
    failures: list[tuple[Path, str]] = []

    for index, file_path in enumerate(files, start=1):
        output_dir = output_root / unique_output_name(file_path, used_names)
        print(f"[{index}/{len(files)}] Processing {file_path.name}")
        print(f"      Output -> {output_dir}")

        try:
            result = parser.parse(file_path)
            write_outputs(result, output_dir)
            warnings = result.metadata.get("warnings", [])
            if warnings:
                print("      Warning:")
                for warning in warnings:
                    print(f"      {warning}")

            preview = result.text[:700].strip()
            if preview:
                label = "      Preview:" if not warnings else "      Result:"
                print(label)
                print(f"      {preview[:700].replace(chr(10), ' ')}")
            else:
                print("      Preview: [No text extracted]")
        except Exception as exc:
            failures.append((file_path, str(exc)))
            print(f"      ERROR: {exc}")

        print("")

    print("Finished.")
    if failures:
        print("\nFailures:")
        for file_path, error in failures:
            print(f"- {file_path.name}: {error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scanned-document OCR and structure reconstruction"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="input",
        help="Input file or folder. PDFs are rasterized page-by-page before OCR.",
    )
    parser.add_argument("--output", "-o", default="output")
    parser.add_argument(
        "--disable-surya",
        action="store_true",
        help="Use Tesseract only and skip Surya line refinement.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Limit the number of pages processed per PDF.",
    )
    parser.add_argument(
        "--max-surya-lines",
        type=int,
        default=20,
        help="Maximum low-confidence lines per page to repair with Surya.",
    )
    parser.add_argument(
        "--pdf-scale",
        type=float,
        default=4.0,
        help="Rasterization scale for PDFs. 4.0 is roughly 288 DPI.",
    )
    parser.add_argument(
        "--tesseract-cmd",
        default=None,
        help="Explicit path to tesseract.exe if it is not on PATH.",
    )
    return parser.parse_args()


def unique_output_name(file_path: Path, used_names: set[str]) -> str:
    base = sanitize_name(file_path.stem)
    name = base or "document"
    counter = 2
    while name in used_names:
        name = f"{base}_{counter}"
        counter += 1
    used_names.add(name)
    return name


def sanitize_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
    return name or "document"


if __name__ == "__main__":
    main()
