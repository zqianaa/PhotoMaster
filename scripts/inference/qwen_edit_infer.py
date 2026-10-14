#!/usr/bin/env python3
"""
Inference script for Qwen-Image-Edit-2511 on dit_input.jsonl benchmarks.

Loads QwenImageEditPlusPipeline and runs image editing inference.
Supports single-GPU, multi-GPU (torchrun), and multi-node distributed execution.

Reproducibility: A fresh torch.Generator(device="cpu") is created per sample
with the same seed, ensuring identical output regardless of GPU sharding.

Usage:
    # Single GPU
    python qwen_edit_infer.py \
        --model-path /path/to/Qwen-Image-Edit-2511 \
        --jsonl-path /path/to/dit_input.jsonl \
        --output-dir /path/to/output/

    # Multi-GPU (torchrun)
    torchrun --nproc_per_node=8 qwen_edit_infer.py \
        --model-path /path/to/model \
        --jsonl-path /path/to/dit_input.jsonl \
        --output-dir /path/to/output/

    # Multi-node
    torchrun --nnodes=2 --nproc_per_node=8 \
        --rdzv_id=JOB_ID --rdzv_backend=c10d --rdzv_endpoint=MASTER:29500 \
        qwen_edit_infer.py \
        --model-path /path/to/model \
        --jsonl-path /path/to/dit_input.jsonl \
        --output-dir /path/to/output/
"""

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageOps


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen Image Edit 2511 inference")
    parser.add_argument("--model-path", required=True, help="Path to model weights")
    parser.add_argument("--jsonl-path", required=True, help="Path to dit_input.jsonl")
    parser.add_argument("--output-dir", required=True, help="Output directory for generated images")
    parser.add_argument("--true-cfg-scale", type=float, default=4.0, help="True CFG scale")
    parser.add_argument("--inference-steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--negative-prompt", type=str, default=" ", help="Negative prompt")
    parser.add_argument("--guidance-scale", type=float, default=1.0, help="Guidance scale")
    parser.add_argument("--ckpt-path", type=str, default="", help="Optional fine-tuned DIT .safetensors checkpoint")
    parser.add_argument(
        "--full-transformer-checkpoint",
        action="store_true",
        help="Load --ckpt-path as a complete 2511 transformer, requiring all keys/shapes to match; skips base transformer weights.",
    )
    parser.add_argument(
        "--output-resolution",
        type=int,
        choices=(1024, 1328),
        default=1328,
        help="Square-equivalent target resolution; aspect ratio is copied from the first input image.",
    )
    parser.add_argument(
        "--vae-input-resolution",
        type=int,
        choices=(1024, 1328),
        default=1328,
        help="Square-equivalent VAE input resolution; aspect ratio is copied from each input image.",
    )
    args = parser.parse_args()
    if args.full_transformer_checkpoint and not args.ckpt_path.endswith(".safetensors"):
        parser.error("--full-transformer-checkpoint requires --ckpt-path to a complete .safetensors file")
    return args


def init_distributed():
    """Initialize distributed process group if launched via torchrun."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        print(f"[Rank {rank}/{world_size}] Initialized on GPU {local_rank}")
    else:
        torch.cuda.set_device(0)
        print(f"[Single GPU] Running on GPU 0")

    return rank, world_size, local_rank


def load_samples(jsonl_path: str) -> list[dict]:
    """Load samples from dit_input.jsonl."""
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def get_sample_id(sample: dict) -> str:
    """Extract sample ID from various possible field names."""
    for key in ("query_id", "id", "sample_id"):
        if key in sample:
            return str(sample[key])
    return str(hash(json.dumps(sample, sort_keys=True)) % 10**8)


def get_output_path(sample: dict, output_dir: str) -> str:
    """Return the output path, preferring metadata save_name when present."""
    if sample.get("save_name"):
        output_name = str(sample["save_name"])
    else:
        output_name = f"{get_sample_id(sample)}.png"
    return os.path.join(output_dir, output_name)


def load_images(sample: dict, jsonl_dir: str) -> list:
    """Load input images from sample, resolving paths relative to JSONL or bench dir."""
    image_field = sample.get("image", [])
    if isinstance(image_field, str):
        image_field = [image_field]

    images = []
    bench_dir = str(Path(jsonl_dir).parent.parent)
    for img_path in image_field:
        if os.path.isabs(img_path):
            full_path = img_path
        else:
            full_path = os.path.join(jsonl_dir, img_path)
            if not os.path.exists(full_path):
                full_path = os.path.join(bench_dir, img_path)
        if not os.path.exists(full_path):
            print(f"  [WARN] Image not found: {full_path}, skipping")
            continue
        images.append(ImageOps.exif_transpose(Image.open(full_path)).convert("RGB"))
    return images


def round_to_multiple(value: float, multiple: int = 16) -> int:
    """Round image dimensions to a model-friendly multiple."""
    return max(multiple, int(round(value / multiple)) * multiple)


def get_output_size(images: list, target_resolution: int) -> tuple[int, int]:
    """Compute output size at target_resolution^2 pixels using the first image aspect ratio."""
    if not images:
        return target_resolution, target_resolution

    input_width, input_height = images[0].size
    if input_width <= 0 or input_height <= 0:
        return target_resolution, target_resolution

    aspect_ratio = input_width / input_height
    target_area = target_resolution * target_resolution
    output_width = round_to_multiple(math.sqrt(target_area * aspect_ratio))
    output_height = round_to_multiple(math.sqrt(target_area / aspect_ratio))

    return output_width, output_height


def get_prompt(sample: dict) -> str:
    """Extract prompt from sample caption field."""
    caption = sample.get("caption", "")
    if isinstance(caption, list):
        return caption[0] if caption else ""
    return caption


def _load_checkpoint_state_dict(ckpt_path: str) -> dict:
    """Load a PyTorch/safetensors checkpoint onto CPU."""
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(ckpt_path, device="cpu")
    data = torch.load(ckpt_path, map_location="cpu")
    if isinstance(data, dict) and "state_dict" in data and isinstance(data["state_dict"], dict):
        return data["state_dict"]
    return data


def _strip_state_dict_prefix(state_dict: dict, prefix: str) -> dict:
    if not state_dict:
        return state_dict
    if all(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict


def _validate_full_transformer_shapes(checkpoint_shapes: dict, expected_shapes: dict) -> None:
    """Reject partial checkpoints and mismatched tensors before loading their data."""
    missing = sorted(set(expected_shapes) - set(checkpoint_shapes))
    unexpected = sorted(set(checkpoint_shapes) - set(expected_shapes))
    mismatched = sorted(
        key for key in set(expected_shapes) & set(checkpoint_shapes)
        if tuple(checkpoint_shapes[key]) != tuple(expected_shapes[key])
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            "Full transformer checkpoint must match every model key and shape: "
            f"missing={len(missing)} {missing[:5]}, "
            f"unexpected={len(unexpected)} {unexpected[:5]}, "
            f"shape_mismatches={len(mismatched)} {mismatched[:5]}"
        )


def _assert_no_meta_tensors(model) -> None:
    """Include unregistered tensor attributes such as Qwen's complex RoPE tables."""
    meta_names = [
        name for name, value in list(model.named_parameters()) + list(model.named_buffers())
        if value.is_meta
    ]
    for module_name, module in model.named_modules():
        for name, value in vars(module).items():
            if torch.is_tensor(value) and value.is_meta:
                meta_names.append(f"{module_name}.{name}".lstrip("."))
    if meta_names:
        raise RuntimeError(f"Full transformer still contains meta tensors: {meta_names[:10]}")


def load_full_transformer(model_path: str, ckpt_path: str, rank: int):
    """Load a complete PhotoMaster transformer without materializing base weights."""
    from accelerate import init_empty_weights
    from diffusers import QwenImageTransformer2DModel
    from safetensors import safe_open

    if not ckpt_path.endswith(".safetensors"):
        raise ValueError("A full transformer checkpoint must be a .safetensors file")
    config = QwenImageTransformer2DModel.load_config(
        str(Path(model_path) / "transformer"), local_files_only=True
    )
    if config.get("zero_cond_t") is not True:
        raise ValueError("A full PhotoMaster transformer requires the Qwen-Image-Edit-2511 zero_cond_t=true config")
    # Keep unregistered RoPE constants on CPU; a global meta-device context
    # would strand them on meta because they are absent from the state dict.
    with init_empty_weights(include_buffers=False):
        transformer = QwenImageTransformer2DModel.from_config(config)
    if getattr(transformer, "zero_cond_t", None) is not True:
        raise RuntimeError("Installed diffusers does not implement zero_cond_t; use diffusers 0.37.0 or a compatible version")

    prefixes = ("module.", "pipe.dit.", "dit.", "transformer.")
    with safe_open(ckpt_path, framework="pt", device="cpu") as checkpoint:
        shapes = {key: checkpoint.get_slice(key).get_shape() for key in checkpoint.keys()}
    for prefix in prefixes:
        shapes = _strip_state_dict_prefix(shapes, prefix)
    expected = {key: value.shape for key, value in transformer.state_dict().items()}
    _validate_full_transformer_shapes(shapes, expected)

    state_dict = _load_checkpoint_state_dict(ckpt_path)
    for prefix in prefixes:
        state_dict = _strip_state_dict_prefix(state_dict, prefix)
    # assign=True materializes meta parameters from the checkpoint without a
    # second initialized transformer allocation; strict=True forbids partial load.
    transformer.load_state_dict(state_dict, strict=True, assign=True)
    del state_dict
    _assert_no_meta_tensors(transformer)
    transformer.to(dtype=torch.bfloat16).eval().requires_grad_(False)
    if rank == 0:
        print(f"[Rank {rank}] Loaded full FT transformer: {len(expected)} keys/shapes matched, BF16, zero_cond_t=True; base transformer weights skipped")
    return transformer


def apply_ft_checkpoint(pipeline, ckpt_path: str, rank: int) -> None:
    """Apply a fine-tuned trainable-state checkpoint to the pipeline transformer."""
    if not ckpt_path:
        return
    if rank == 0:
        print(f"[Rank {rank}] Loading fine-tuned checkpoint: {ckpt_path}")

    state_dict = _load_checkpoint_state_dict(ckpt_path)
    state_dict = _strip_state_dict_prefix(state_dict, "module.")
    state_dict = _strip_state_dict_prefix(state_dict, "pipe.dit.")
    state_dict = _strip_state_dict_prefix(state_dict, "dit.")
    state_dict = _strip_state_dict_prefix(state_dict, "transformer.")

    target = getattr(pipeline, "transformer", None) or getattr(pipeline, "dit", None)
    if target is None:
        raise AttributeError("Qwen Image Edit pipeline has no transformer/dit component for FT checkpoint loading")

    incompatible = target.load_state_dict(state_dict, strict=False)
    missing = len(getattr(incompatible, "missing_keys", []) or [])
    unexpected = len(getattr(incompatible, "unexpected_keys", []) or [])
    if unexpected == len(state_dict):
        raise RuntimeError(
            "Fine-tuned checkpoint did not match the pipeline transformer. "
            "All checkpoint keys were unexpected."
        )
    if rank == 0:
        print(
            f"[Rank {rank}] Applied FT checkpoint "
            f"(missing keys: {missing}, unexpected keys: {unexpected})"
        )


def load_qwen_image_edit_plus_pipeline_class():
    """Load the vendored Qwen Edit Plus pipeline patch instead of env diffusers."""
    default_path = (
        Path(__file__).resolve().parent
        / "vendor"
        / "qwenimage"
        / "pipeline_qwenimage_edit_plus.py"
    )
    pipeline_path = Path(os.environ.get("PHOTOMASTER_QWEN_EDIT_PIPELINE", default_path)).resolve()
    if not pipeline_path.is_file():
        raise FileNotFoundError(f"Vendored Qwen Edit pipeline not found: {pipeline_path}")

    # The vendored file keeps diffusers' relative imports, so load it under the
    # installed diffusers qwenimage package context.
    import diffusers.pipelines.qwenimage  # noqa: F401

    module_name = "diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus_photomaster"
    spec = importlib.util.spec_from_file_location(module_name, pipeline_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load vendored Qwen Edit pipeline: {pipeline_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.QwenImageEditPlusPipeline


def main():
    args = parse_args()
    rank, world_size, local_rank = init_distributed()
    is_main = rank == 0

    # Load samples and shard
    samples = load_samples(args.jsonl_path)
    jsonl_dir = str(Path(args.jsonl_path).parent)
    my_samples = [s for i, s in enumerate(samples) if i % world_size == rank]

    if is_main:
        print(f"Total samples: {len(samples)}, this rank: {len(my_samples)}")

    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Check which samples already have output (resume support)
    pending = []
    for s in my_samples:
        sid = get_sample_id(s)
        out_path = get_output_path(s, args.output_dir)
        if os.path.exists(out_path):
            if is_main or rank == 0:
                pass  # silently skip
        else:
            pending.append(s)

    skipped = len(my_samples) - len(pending)
    if skipped > 0:
        print(f"[Rank {rank}] Skipping {skipped} already-completed samples, {len(pending)} remaining")

    if not pending:
        print(f"[Rank {rank}] All samples already completed")
        if world_size > 1:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
        return

    # Load pipeline
    print(f"[Rank {rank}] Loading pipeline from {args.model_path}...")
    t0 = time.time()

    QwenImageEditPlusPipeline = load_qwen_image_edit_plus_pipeline_class()
    if is_main:
        print(f"Using vendored Qwen Edit pipeline: {QwenImageEditPlusPipeline.__module__}")
    pipeline_kwargs = {}
    if args.full_transformer_checkpoint:
        pipeline_kwargs["transformer"] = load_full_transformer(args.model_path, args.ckpt_path, rank)
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, **pipeline_kwargs
    )
    if not args.full_transformer_checkpoint:
        apply_ft_checkpoint(pipeline, args.ckpt_path, rank)
    pipeline.to(f"cuda:{local_rank}")
    pipeline.set_progress_bar_config(disable=True)

    t1 = time.time()
    print(f"[Rank {rank}] Pipeline loaded in {t1 - t0:.1f}s")

    # Run inference
    success = 0
    failed = 0
    for idx, sample in enumerate(pending):
        sid = get_sample_id(sample)
        out_path = get_output_path(sample, args.output_dir)
        prompt = get_prompt(sample)
        images = load_images(sample, jsonl_dir)

        if not images:
            print(f"[Rank {rank}] [{idx+1}/{len(pending)}] Sample {sid}: no valid images, skipping")
            failed += 1
            continue

        if not prompt:
            print(f"[Rank {rank}] [{idx+1}/{len(pending)}] Sample {sid}: empty prompt, skipping")
            failed += 1
            continue

        try:
            generator = torch.Generator(device="cpu").manual_seed(args.seed)
            output_width, output_height = get_output_size(images, args.output_resolution)
            with torch.inference_mode():
                output = pipeline(
                    image=images,
                    prompt=prompt,
                    width=output_width,
                    height=output_height,
                    vae_input_resolution=args.vae_input_resolution,
                    generator=generator,
                    true_cfg_scale=args.true_cfg_scale,
                    num_inference_steps=args.inference_steps,
                    guidance_scale=args.guidance_scale,
                    negative_prompt=args.negative_prompt,
                    num_images_per_prompt=1,
                )
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            output.images[0].save(out_path)
            success += 1
            print(
                f"[Rank {rank}] [{idx+1}/{len(pending)}] Sample {sid}: "
                f"{output_width}x{output_height}, "
                f"vae_input_resolution={args.vae_input_resolution}, saved to {out_path}"
            )
        except Exception as e:
            failed += 1
            print(f"[Rank {rank}] [{idx+1}/{len(pending)}] Sample {sid}: ERROR - {e}")

    local_counts = torch.tensor([success, failed], dtype=torch.long, device=f"cuda:{local_rank}")
    if world_size > 1:
        torch.distributed.all_reduce(local_counts, op=torch.distributed.ReduceOp.SUM)
    total_success = int(local_counts[0].item())
    total_failed = int(local_counts[1].item())

    print(f"[Rank {rank}] Done. Success: {success}, Failed: {failed}")

    exit_code = 1 if total_failed > 0 else 0
    if is_main:
        output_exts = {".png", ".jpg", ".jpeg", ".webp"}
        total_output = sum(
            1 for f in os.listdir(args.output_dir)
            if os.path.splitext(f)[1].lower() in output_exts
        )
        print(f"Total output images: {total_output}/{len(samples)}")
        print(f"Total reported samples: success={total_success}, failed={total_failed}")
        if total_output < len(samples):
            print(
                f"[ERROR] Qwen Edit output incomplete: "
                f"{total_output}/{len(samples)} images were produced"
            )
            exit_code = 1

    # Sync all ranks before exiting so torchrun observes a clean distributed shutdown.
    if world_size > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
