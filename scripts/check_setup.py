#!/usr/bin/env python3
"""Check local inference inputs and imports without downloading or loading weights."""

import argparse
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_data(data_dir):
    errors = []
    try:
        samples = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"Cannot read metadata.json: {exc}"]
    if not isinstance(samples, list) or not samples:
        return ["metadata.json must contain a non-empty list."]
    ids, names = set(), set()
    for number, sample in enumerate(samples, 1):
        if not isinstance(sample, dict):
            errors.append(f"Record {number} must be an object.")
            continue
        index = sample.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index in ids:
            errors.append(f"Record {number}: index must be a unique integer.")
        else:
            ids.add(index)
        source = sample.get("original_img_path")
        if not isinstance(source, str) or not source or not (data_dir / source).is_file():
            errors.append(f"Record {number}: input image is missing.")
        name = sample.get("save_name") or (Path(source).name if isinstance(source, str) else "")
        if not isinstance(name, str) or not name or name in names or Path(name).name != name or name in (".", ".."):
            errors.append(f"Record {number}: save_name must be a unique filename without directories.")
        else:
            names.add(name)
    print(f"Data: {len(samples)} records in {data_dir}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "bench/photomaster_bench")
    parser.add_argument("--data-only", action="store_true", help="CPU-only metadata/path validation; no model checks")
    parser.add_argument("--vlm-path", type=Path, default=ROOT / "models/ft_ckpts/vlm")
    parser.add_argument("--base-vlm", type=Path, default=ROOT / "models/Qwen3-VL-32B-Instruct")
    parser.add_argument("--dit-path", type=Path, default=ROOT / "models/Qwen-Image-Edit-2511")
    parser.add_argument("--ft-ckpt", type=Path, help="Fine-tuned DiT checkpoint; required unless --allow-base-dit")
    parser.add_argument("--allow-base-dit", action="store_true", help="Explicitly check a base-renderer experiment")
    args = parser.parse_args()
    errors = check_data(args.data_dir)
    if not args.data_only:
        if not (args.vlm_path / "adapter_config.json").is_file() and not (args.vlm_path / "config.json").is_file():
            errors.append(f"VLM config/adapter_config missing: {args.vlm_path}")
        if (args.vlm_path / "adapter_config.json").is_file() and not (args.base_vlm / "config.json").is_file():
            errors.append(f"LoRA base VLM config missing: {args.base_vlm}")
        if not (args.dit_path / "model_index.json").is_file():
            errors.append(f"DiT model_index.json missing: {args.dit_path}")
        if args.ft_ckpt is not None:
            if not args.ft_ckpt.is_file():
                errors.append(f"Fine-tuned DiT checkpoint missing: {args.ft_ckpt}")
        elif not args.allow_base_dit:
            errors.append("Pass --ft-ckpt for PhotoMaster, or --allow-base-dit for a base-renderer experiment.")
        for package in ("torch", "torchvision", "transformers", "diffusers", "accelerate", "peft", "safetensors", "Pillow", "numpy", "huggingface_hub"):
            try:
                print(f"{package}: {importlib.metadata.version(package)}")
            except importlib.metadata.PackageNotFoundError:
                errors.append(f"Dependency missing: {package}")
        try:
            import torch
            if not torch.cuda.is_available():
                errors.append("CUDA is unavailable; current inference workers require NVIDIA CUDA.")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {props.name}, {props.total_memory / 2**30:.1f} GiB")
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor  # noqa: F401
            from peft import PeftModel  # noqa: F401
            sys.path.insert(0, str(ROOT / "scripts/inference"))
            worker = importlib.import_module("qwen_edit_infer")
            worker.load_qwen_image_edit_plus_pipeline_class()
        except Exception as exc:
            errors.append(f"Required inference import failed: {type(exc).__name__}: {exc}")
    for error in errors:
        print(f"FAIL: {error}", file=sys.stderr)
    if errors:
        return 1
    print("Data checks passed." if args.data_only else "Preflight passed. Weight completeness, VRAM fit and generated outputs still require a GPU smoke run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
