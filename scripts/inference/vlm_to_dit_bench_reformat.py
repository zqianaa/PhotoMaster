"""Convert successful PhotoMaster VLM results to renderer JSONL inputs.

Usage:
    python vlm_to_dit_bench_reformat.py input.json -o dit_input.jsonl
"""

import argparse
import json
from pathlib import Path


def convert(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list) or not results:
        raise ValueError("Input must contain a non-empty 'results' array.")
    lines = []
    seen_ids, seen_names = set(), set()
    for position, r in enumerate(results, 1):
        if not isinstance(r, dict):
            raise ValueError(f"Result {position} must be an object.")
        sample_id = r.get("index")
        if not isinstance(sample_id, int) or isinstance(sample_id, bool) or sample_id in seen_ids:
            raise ValueError(f"Result {position} requires a unique integer index.")
        seen_ids.add(sample_id)
        prompt = r.get("Image_Editing_Prompt")
        raw = r.get("model_output_raw", "")
        if ("parse_error" in r or r.get("error") or
                (isinstance(raw, str) and raw.startswith("ERROR")) or
                not isinstance(prompt, str) or not prompt.strip()):
            raise ValueError(f"Sample {sample_id} has a failed, malformed or empty VLM response. Inspect vlm_output.json and rerun VLM inference.")
        original_img_path = r.get("original_img_path")
        if not isinstance(original_img_path, str) or not original_img_path.strip():
            raise ValueError(f"Sample {sample_id} has no input image path.")
        save_name = r.get("save_name")
        if not save_name and isinstance(original_img_path, str):
            save_name = Path(original_img_path).name
        if not isinstance(save_name, str) or not save_name or Path(save_name).name != save_name or save_name in (".", ".."):
            raise ValueError(f"Sample {sample_id} save_name must be a filename without directories.")
        if save_name in seen_names:
            raise ValueError(f"Duplicate output filename: {save_name}")
        seen_names.add(save_name)
        item = {
            "query_id": str(r["index"]),
            "caption": [prompt],
            "image": [original_img_path],
            "save_name": save_name,
            "task": r.get("task", ""),
        }
        lines.append(json.dumps(item, ensure_ascii=False))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Converted {len(lines)} items: {input_path} -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Input JSON file (with 'results' array)")
    parser.add_argument("-o", "--output", help="Output JSONL path (default: <input>_dit.jsonl)")
    args = parser.parse_args()

    if args.output:
        out = args.output
    else:
        p = Path(args.input)
        out = str(p.with_name(p.stem + "_dit.jsonl"))

    try:
        convert(args.input, out)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")
