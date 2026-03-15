import argparse
from pathlib import Path

from document_parser import DocumentParser, write_outputs


def main(document_path: str, output_dir: str, lang: str, disable_surya: bool):
    print(f"\nInput         : {document_path}")
    print(f"Output        : {output_dir}")
    print(f"Language      : {lang}")
    print(f"Use Surya     : {not disable_surya}\n")

    parser = DocumentParser(lang=lang, use_surya=not disable_surya)
    result = parser.parse(document_path)
    write_outputs(result, output_dir)

    print("Preview:\n")
    preview = result.text[:3000].strip()
    print(preview if preview else "[No text extracted]")
    if len(result.text) > 3000:
        print(f"\n... [{len(result.text) - 3000} more chars]")

    print(f"\nAll outputs -> {Path(output_dir).resolve()}")
    print("  document.txt")
    print("  document.md")
    print("  result.json")
    if result.lines:
        print("  layout_detection.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="General document parser")
    parser.add_argument(
        "document",
        nargs="?",
        default="input/document.png",
        help="Path to document (image, PDF, DOCX, TXT, MD)",
    )
    parser.add_argument("--output", "-o", default="output")
    parser.add_argument("--lang", "-l", default="en")
    parser.add_argument(
        "--disable-surya",
        action="store_true",
        help="Use EasyOCR only. Helpful if Surya is unavailable or unstable.",
    )
    args = parser.parse_args()

    if not Path(args.document).exists():
        print(f"ERROR: Document not found: {args.document}")
    else:
        main(args.document, args.output, args.lang, args.disable_surya)
