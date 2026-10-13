#!/usr/bin/env python3
"""Prepare one photo for the existing benchmark-based inference pipeline."""

import argparse
import json
import re
import tempfile
from pathlib import Path

from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[2]


def prepare(image_path, bench_name, repo_root=REPO_ROOT):
    if not re.fullmatch(r"demo_[A-Za-z0-9_-]+", bench_name):
        raise ValueError("Benchmark name must start with demo_ and contain only letters, digits, _ or -.")
    source = Path(image_path).expanduser().resolve()
    destination = Path(repo_root) / "bench" / bench_name
    if destination.exists():
        raise FileExistsError(f"{destination} already exists; choose a new demo name to preserve prior inputs/results.")
    # Normalize EXIF orientation and remove metadata before copying into the bench.
    # Build in a temporary directory so invalid images cannot leave half-written input.
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as original:
        normalized = ImageOps.exif_transpose(original).convert("RGB")
        clean = Image.new("RGB", normalized.size)
        clean.paste(normalized)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=destination.parent) as staging:
        stage = Path(staging)
        (stage / "img").mkdir()
        clean.save(stage / "img" / "input.png")
        record = [{"index": 1, "original_img_path": "img/input.png", "save_name": "input.png"}]
        (stage / "metadata.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        stage.rename(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Local input photo (read only)")
    parser.add_argument("--name", default="demo_my_photo", help="New demo_* benchmark name")
    args = parser.parse_args()
    try:
        destination = prepare(args.image, args.name)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Prepared {destination}")
    print(f"GPU_MODE=single bash scripts/inference/run_vlm_infer.sh vlm {args.name}")
    print("See docs/quickstart.md for the full pipeline and fine-tuned DiT checkpoint setting.")


if __name__ == "__main__":
    main()
