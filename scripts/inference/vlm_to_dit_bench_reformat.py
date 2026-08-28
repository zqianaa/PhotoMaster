"""Convert model result JSON to BOOGU bench JSONL format.

Usage:
    python convert_to_boogu.py input.json -o output.jsonl
    python convert_to_boogu.py input.json  # writes to input_boogu.jsonl
"""

import argparse
import json
from pathlib import Path


def convert(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    lines = []
    for r in results:
        original_img_path = r["original_img_path"]
        save_name = r.get("save_name")
        if not save_name and isinstance(original_img_path, str):
            save_name = original_img_path.removeprefix("img/")
        item = {
            "query_id": str(r["index"]),
            "caption": [r["Image_Editing_Prompt"]],
            "image": [original_img_path],
            "save_name": save_name,
            "task": r.get("task", ""),
        }
        lines.append(json.dumps(item, ensure_ascii=False))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Converted {len(lines)} items: {input_path} -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert model result JSON to BOOGU bench JSONL")
    parser.add_argument("input", help="Input JSON file (with 'results' array)")
    parser.add_argument("-o", "--output", help="Output JSONL path (default: <input>_boogu.jsonl)")
    args = parser.parse_args()

    if args.output:
        out = args.output
    else:
        p = Path(args.input)
        out = str(p.with_name(p.stem + "_boogu.jsonl"))

    convert(args.input, out)
