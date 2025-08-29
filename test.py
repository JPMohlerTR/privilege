#!/usr/bin/env python3
# Combine specific images into a single multi-page PDF.
# Loss of image quality is fine.

from pathlib import Path
from PIL import Image, ImageOps

# ---- Edit only if you want a different output path/name ----
OUTPUT_PDF = Path("/Users/johnmohler/Downloads/Exhibits_8-16_combined.pdf")

# The exact files you provided, in order
IMAGE_PATHS = [
    "/Users/johnmohler/Downloads/Exhibit 8.jpeg",
    "/Users/johnmohler/Downloads/Exhibit 9.jpeg",
    "/Users/johnmohler/Downloads/Exhibit 10.jpeg",
    "/Users/johnmohler/Downloads/Exhibit 11.jpeg",
    "/Users/johnmohler/Downloads/Exhibit 12.jpeg",
    "/Users/johnmohler/Downloads/Exhibit 13.jpeg",
    "/Users/johnmohler/Downloads/Exhibit 14.jpeg",
    "/Users/johnmohler/Downloads/Exhibit 15.jpeg",
    "/Users/johnmohler/Downloads/Exhibit 16.jpg",
]

def load_image_as_pdf_page(p: Path) -> Image.Image:
    """Open image, fix EXIF orientation, convert to RGB for PDF."""
    im = Image.open(p)
    im = ImageOps.exif_transpose(im)  # auto-rotate if camera added EXIF orientation
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    else:
        # Pillow sometimes embeds 'L' (grayscale) fine in PDFs,
        # but converting to RGB keeps things consistent.
        im = im.convert("RGB")
    return im

def main():
    paths = [Path(p) for p in IMAGE_PATHS]
    existing = [p for p in paths if p.exists()]

    if not existing:
        raise SystemExit("None of the specified image files were found.")

    missing = [p for p in paths if not p.exists()]
    if missing:
        print("Warning: the following files were not found and will be skipped:")
        for m in missing:
            print(f"  - {m}")

    pages = [load_image_as_pdf_page(p) for p in existing]

    first, rest = pages[0], pages[1:]
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    # Save as multi-page PDF. optimize=True can reduce size a bit.
    first.save(
        OUTPUT_PDF,
        format="PDF",
        save_all=True,
        append_images=rest,
        optimize=True,
        resolution=72.0,  # standard; PDF viewers scale anyway
    )

    print(f"Combined {len(pages)} image(s) into:\n{OUTPUT_PDF}")

if __name__ == "__main__":
    main()
