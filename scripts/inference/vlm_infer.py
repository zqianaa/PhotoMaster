#!/usr/bin/env python3
"""
Inference script for Qwen3-VL on the photography critique benchmark.

Loads a Qwen3-VL model and runs inference on samples defined in a metadata.json file.
Each sample contains an original image.

The analysis framework is defined in an external [REDACTED] file
(default: prompts/system_prompt.txt, resolved next to this script).
The script generates JSON output with three fields: Issues_Analysis, Professional_Suggestions,
and Image_Editing_Prompt.

Supports both full-finetuned checkpoints and LoRA-finetuned checkpoints. LoRA
checkpoints are auto-detected by the presence of adapter_config.json.

Usage:
    # Single GPU – output auto-derived to <data-dir>/results/<model-basename>/vlm_output.json
    python vlm_infer.py \
        --model-path /path/to/full-finetuned-ckpt \
        --data-dir /path/to/bench

    # Single GPU – explicit output path
    python vlm_infer.py \
        --model-path /path/to/full-finetuned-ckpt \
        --data-dir /path/to/bench \
        --output-json /path/to/output/results.json

    # Single GPU – custom [REDACTED] file
    python vlm_infer.py \
        --model-path /path/to/model \
        --data-dir /path/to/bench \
        --output-json /path/to/output/results.json \
        --system-prompt-file /path/to/custom_prompt.txt

    # Multi-node with torchrun
    torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
        --master_addr=master_node_ip --master_port=29500 \
        vlm_infer.py \
        --model-path /path/to/Qwen3-VL-32B-Instruct \
        --data-dir /path/to/bench \
        --max-new-tokens 2048 \
        --temperature 0.7

Multi-node note: ranks exchange results through the filesystem, so
PHOTOMASTER_GATHER_DIR must point at storage shared by every node.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


# ---------------------------------------------------------------------------
# User prompt template
# ---------------------------------------------------------------------------

USER_PROMPT = (
    "Provide your analysis in JSON format with the following structure:\n"
    "{{\n"
    '  "Issues_Analysis": [\n'
    '    {{\n'
    '      "Dimension": "<Dimension Name>",\n'
    '      "Detail": "<Direct critique or \'None\' if acceptable>"\n'
    '    }}\n'
    '  ],\n'
    '  "Professional_Suggestions": [\n'
    '    {{\n'
    '      "Dimension": "<Dimension Name>",\n'
    '      "Detail": "<Specific advice or \'None\' if acceptable>"\n'
    '    }}\n'
    '  ],\n'
    '  "Image_Editing_Prompt": "<Detailed text-to-image prompt>"\n'
    "}}\n\n"
    "Return ONLY the JSON object, no additional text."
)

DEFAULT_SYSTEM_PROMPT = Path(__file__).resolve().parent / "prompts" / "system_prompt.txt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Qwen3-VL inference on photography critique benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the Qwen3-VL model directory (full-finetuned) or LoRA adapter directory.",
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default=None,
        help="Path to the base model for LoRA adapters. If not specified, uses "
             "base_model_name_or_path from adapter_config.json.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the benchmark data directory (containing metadata.json and img/).",
    )
    parser.add_argument(
        "--metadata-file",
        type=str,
        default="metadata.json",
        help="Name of the metadata JSON file (default: metadata.json).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to the output JSON file. If not specified, auto-derives as "
             "<data-dir>/results/<model-basename>/vlm_output.json.",
    )
    parser.add_argument(
        "--system-prompt-file",
        type=str,
        default=str(DEFAULT_SYSTEM_PROMPT),
        help=f"Path to the [REDACTED] file (default: {DEFAULT_SYSTEM_PROMPT}).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum number of new tokens to generate (default: 2048).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7). Set to 0 for greedy decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Top-p (nucleus) sampling (default: 0.8).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k sampling (default: 20).",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=1003520,
        help="Maximum number of pixels for image resizing (default: 1003520).",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=3136,
        help="Minimum number of pixels for image resizing (default: 3136).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: 'auto', 'cuda', 'cpu' (default: auto).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype (default: bfloat16).",
    )
    parser.add_argument(
        "--sample-indices",
        type=str,
        default=None,
        help="Comma-separated list of sample indices to run (1-based). "
             "If not specified, runs all samples. E.g. '1,3,5' or '1-10'.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="If set, only process the first N samples from metadata before any sharding/index filtering.",
    )
    parser.add_argument(
        "--save_raw_outputs",
        action="store_true",
        default=False,
        help="If set, save full generated outputs with special tokens and token-level diagnostics.",
    )
    return parser.parse_args()


def load_system_prompt(file_path: str) -> str:
    """Load [REDACTED] from file."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    return content


def parse_sample_indices(indices_str: str, total: int) -> list[int]:
    """Parse sample indices string like '1,3,5' or '1-10' or '1,3-5,8'."""
    if indices_str is None:
        return list(range(total))

    result = set()
    for part in indices_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            start_idx = int(start) - 1  # Convert 1-based to 0-based
            end_idx = int(end) - 1
            result.update(range(start_idx, end_idx + 1))
        else:
            result.add(int(part) - 1)  # Convert 1-based to 0-based

    # Filter valid indices
    return sorted(i for i in result if 0 <= i < total)


def setup_distributed():
    """Initialize distributed training environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        # Initialize process group
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        # Single GPU/CPU mode
        return 0, 1, 0


def cleanup_distributed():
    """Clean up distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def _is_lora_checkpoint(model_path: str) -> bool:
    """Check if the model path contains a LoRA adapter (has adapter_config.json)."""
    return os.path.isfile(os.path.join(model_path, "adapter_config.json"))


def _resolve_base_model_path(model_path: str, base_model_path: str | None) -> str:
    """Resolve the base model path for a LoRA adapter."""
    if base_model_path is not None:
        return base_model_path
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    with open(adapter_config_path, "r", encoding="utf-8") as f:
        adapter_config = json.load(f)
    base_path = adapter_config.get("base_model_name_or_path")
    if not base_path:
        raise ValueError(
            f"adapter_config.json in {model_path} has no 'base_model_name_or_path'. "
            f"Please specify --base-model-path explicitly."
        )
    return base_path


def load_model_and_processor(
    model_path: str, device: str, dtype: str,
    min_pixels: int, max_pixels: int, local_rank: int = 0,
    base_model_path: str | None = None,
):
    """Load the Qwen3-VL model and processor.

    Supports both full-finetuned checkpoints and LoRA adapter checkpoints.
    LoRA checkpoints are detected by the presence of adapter_config.json.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    is_lora = _is_lora_checkpoint(model_path)

    if rank == 0:
        if is_lora:
            resolved_base = _resolve_base_model_path(model_path, base_model_path)
            print(f"Detected LoRA checkpoint: {model_path}")
            print(f"  Base model: {resolved_base}")
        else:
            print(f"Loading full-finetuned model from: {model_path}")
        print(f"  Device: {device}, Dtype: {dtype}")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[dtype]

    # For distributed training, use local_rank as device
    if dist.is_initialized():
        device_map = {"": local_rank}
    else:
        device_map = device

    if is_lora:
        from peft import PeftModel

        resolved_base = _resolve_base_model_path(model_path, base_model_path)
        # Load base model on CPU, merge LoRA on CPU, then move to GPU
        # This avoids OOM from having both base weights + adapter on GPU simultaneously
        if rank == 0:
            print("Loading base model on CPU for LoRA merge...")
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            resolved_base,
            torch_dtype=torch_dtype,
            device_map="cpu",
            attn_implementation="flash_attention_2",
        )
        # Apply and merge the LoRA adapter on CPU
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
        if rank == 0:
            print("LoRA adapter merged into base model. Moving to GPU...")
        # Move merged model to the target device
        if dist.is_initialized():
            model = model.to(f"cuda:{local_rank}")
        else:
            target_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
            model = model.to(target_device)
        # Load processor from base model (LoRA dir may not have all processor files)
        processor_path = resolved_base
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation="flash_attention_2",
        )
        processor_path = model_path

    model.eval()

    processor = AutoProcessor.from_pretrained(
        processor_path,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    if rank == 0:
        print("Model and processor loaded successfully.")
    return model, processor


def build_messages(
    sample: dict,
    data_dir: Path,
    system_prompt: str,
) -> tuple[list[dict], list[Image.Image]]:
    """
    Build the chat messages and load images for a single sample.

    Returns:
        messages: list of message dicts for the chat template
        images: list of PIL images (in the order they appear in the messages)
    """
    # Load images
    images = []
    user_content = []

    # Original image
    orig_img_path = data_dir / sample["original_img_path"]
    if orig_img_path.exists():
        orig_img = Image.open(orig_img_path).convert("RGB")
        images.append(orig_img)
        user_content.append({"type": "image", "image": orig_img_path.as_posix()})
    else:
        print(f"  WARNING: Original image not found: {orig_img_path}")

    # Add user prompt
    user_content.append({"type": "text", "text": "\n" + USER_PROMPT})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    return messages, images


def run_inference(
    model,
    processor,
    messages: list[dict],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> dict:
    """Run inference on a single sample and return generation details."""
    # Apply chat template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Process inputs (handles images automatically)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    image_grid_thw = (
        inputs["image_grid_thw"].cpu().tolist()
        if "image_grid_thw" in inputs
        else None
    )
    inputs = inputs.to(model.device)

    # Generation config
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
    }
    if temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
        gen_kwargs["top_k"] = top_k
    else:
        gen_kwargs["do_sample"] = False

    # Generate
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    # Decode only the generated part (exclude input tokens)
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_len:]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    output_text_with_special = processor.batch_decode(
        generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )[0]

    # Diagnose stop reason: eos reached vs max_new_tokens cap.
    eos_token_id = model.generation_config.eos_token_id
    if eos_token_id is None:
        eos_token_ids = set()
    elif isinstance(eos_token_id, int):
        eos_token_ids = {eos_token_id}
    else:
        eos_token_ids = set(eos_token_id)

    generated_token_ids = generated_ids[0].tolist() if generated_ids.numel() > 0 else []
    ended_with_eos = bool(generated_token_ids) and (generated_token_ids[-1] in eos_token_ids)
    generated_token_count = len(generated_token_ids)
    hit_max_new_tokens = generated_token_count >= max_new_tokens

    return {
        "output_text": output_text.strip(),
        "output_text_with_special_tokens": output_text_with_special,
        "generated_token_ids": generated_token_ids,
        "generated_token_count": generated_token_count,
        "ended_with_eos_token": ended_with_eos,
        "hit_max_new_tokens": hit_max_new_tokens,
        "eos_token_ids": sorted(eos_token_ids),
        "image_grid_thw": image_grid_thw,
    }


def process_vision_info(messages):
    """
    Extract image and video inputs from messages for the processor.
    Compatible with qwen_vl_utils if available, otherwise uses a simple fallback.
    """
    try:
        from qwen_vl_utils import process_vision_info as _process_vision_info
        return _process_vision_info(messages)
    except ImportError:
        # Fallback: manually collect images from messages
        images = []
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, str):
                continue
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "image":
                        img_path = item.get("image", "")
                        if img_path and os.path.exists(img_path):
                            images.append(Image.open(img_path).convert("RGB"))
        return images if images else None, None


def format_output_to_string(output_value) -> str:
    """
    Convert the output value (list, dict, or string) into a single formatted string.

    Args:
        output_value: Can be a list, dict, or string from the model output

    Returns:
        A formatted string representation of the output
    """
    if isinstance(output_value, str):
        return output_value
    elif isinstance(output_value, list):
        if not output_value:
            return ""
        formatted_items = []
        for item in output_value:
            if isinstance(item, dict):
                dimension = item.get("Dimension", "")
                detail = item.get("Detail", "")
                if detail and detail.lower() != "none":
                    formatted_items.append(f"【{dimension}】{detail}")
            else:
                formatted_items.append(str(item))
        return "\n".join(formatted_items)
    else:
        return str(output_value)


def shard_data(data: list, rank: int, world_size: int) -> list:
    """Shard data across GPUs. Each GPU gets a contiguous chunk."""
    total_samples = len(data)
    samples_per_rank = total_samples // world_size
    remainder = total_samples % world_size
    
    # Distribute remainder samples to first few ranks
    if rank < remainder:
        start_idx = rank * (samples_per_rank + 1)
        end_idx = start_idx + samples_per_rank + 1
    else:
        start_idx = rank * samples_per_rank + remainder
        end_idx = start_idx + samples_per_rank
    
    return data[start_idx:end_idx]


def gather_results(results: list, rank: int, world_size: int) -> list:
    """Gather results from all ranks to rank 0 via filesystem to avoid NCCL issues."""
    if not dist.is_initialized():
        return results

    # Each rank writes its results to a shared filesystem dir (not /tmp which is node-local)
    tmp_dir = Path(os.environ.get("PHOTOMASTER_GATHER_DIR", "/tmp/infer_gather"))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_file = tmp_dir / f"rank_{rank}.json"

    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

    # Barrier to ensure all ranks have written their files
    dist.barrier()

    if rank == 0:
        all_results = []
        for r in range(world_size):
            rfile = tmp_dir / f"rank_{r}.json"
            with open(rfile, "r", encoding="utf-8") as f:
                all_results.extend(json.load(f))
            rfile.unlink()

        # Sort by original index to maintain order
        all_results.sort(key=lambda x: x.get("index", 0))
        return all_results
    else:
        return []


def main():
    args = parse_args()

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()

    try:
        # Load [REDACTED]
        try:
            system_prompt = load_system_prompt(args.system_prompt_file)
            if rank == 0:
                print(f"Loaded [REDACTED] from: {args.system_prompt_file}")
        except FileNotFoundError:
            if rank == 0:
                print(f"ERROR: [REDACTED] file not found: {args.system_prompt_file}")
            sys.exit(1)

        data_dir = Path(args.data_dir)
        metadata_path = data_dir / args.metadata_file

        # Auto-derive output path if not specified
        if args.output_json is None:
            model_basename = Path(args.model_path.rstrip("/")).name
            output_json_path = data_dir / "results" / model_basename / "vlm_output.json"
        else:
            output_json_path = Path(args.output_json)

        # Validate paths (only rank 0 prints)
        if not data_dir.exists():
            if rank == 0:
                print(f"ERROR: Data directory not found: {data_dir}")
            sys.exit(1)
        if not metadata_path.exists():
            if rank == 0:
                print(f"ERROR: Metadata file not found: {metadata_path}")
            sys.exit(1)

        # Create output directory if it doesn't exist (only rank 0)
        if rank == 0:
            output_dir = output_json_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)

        # Load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        if args.num_samples is not None:
            if args.num_samples <= 0:
                if rank == 0:
                    print("ERROR: --num_samples must be > 0 when provided.")
                sys.exit(1)
            metadata = metadata[:args.num_samples]

        if rank == 0:
            print(f"Loaded {len(metadata)} samples from {metadata_path}")
            print(f"Running distributed inference with {world_size} GPUs")

        # Parse sample indices
        sample_indices = parse_sample_indices(args.sample_indices, len(metadata))

        # Shard data across GPUs
        local_indices = shard_data(sample_indices, rank, world_size)

        if rank == 0:
            print(f"Total samples to process: {len(sample_indices)}")
        print(f"[Rank {rank}] Processing {len(local_indices)} samples")

        # Load model (each GPU gets its own instance)
        model, processor = load_model_and_processor(
            args.model_path, args.device, args.dtype,
            args.min_pixels, args.max_pixels, local_rank,
            base_model_path=args.base_model_path,
        )

        # Run inference on local shard
        results = []
        total = len(local_indices)

        for step, idx in enumerate(local_indices, 1):
            sample = metadata[idx]
            sample_index = sample.get("index", idx + 1)
            print(f"\n[Rank {rank}] {'='*60}")
            print(f"[Rank {rank}] [{step}/{total}] Processing sample {sample_index} "
                  f"(original_img_path: {sample.get('original_img_path', 'N/A')})")

            # Build messages
            messages, images = build_messages(
                sample, data_dir, system_prompt,
            )

            # Run inference
            start_time = time.time()
            inference_info = None
            try:
                inference_info = run_inference(
                    model, processor, messages,
                    args.max_new_tokens, args.temperature, args.top_p, args.top_k,
                )
                output_text = inference_info["output_text"]
                elapsed = time.time() - start_time
                print(f"[Rank {rank}]   Generated {len(output_text)} chars in {elapsed:.1f}s")
                print(f"[Rank {rank}]   Preview: {output_text[:200]}...")
                if inference_info["hit_max_new_tokens"]:
                    print(
                        f"[Rank {rank}]   WARNING: generation reached max_new_tokens "
                        f"({args.max_new_tokens}); output may be truncated."
                    )
            except Exception as e:
                output_text = f"ERROR: {str(e)}"
                elapsed = time.time() - start_time
                print(f"[Rank {rank}]   ERROR during inference: {e}")

            # Build result record - start with all fields from original metadata
            result = dict(sample)  # Copy all fields from metadata
            result["Issues_Analysis"] = ""
            result["Professional_Suggestions"] = ""
            result["Image_Editing_Prompt"] = ""

            parsed_output = {}

            if not output_text.startswith("ERROR"):
                try:
                    # Try to extract JSON from the output.
                    json_text = output_text.strip()
                    if json_text.startswith("```"):
                        lines = json_text.split("\n")
                        json_text = "\n".join(lines[1:-1]) if len(lines) > 2 else json_text
                        json_text = json_text.replace("```json", "").replace("```", "").strip()

                    parsed_output = json.loads(json_text)
                    print(f"[Rank {rank}]   Successfully parsed JSON output")
                except json.JSONDecodeError as e:
                    print(f"[Rank {rank}]   WARNING: Failed to parse JSON output: {e}")
                    print(f"[Rank {rank}]   Raw output: {output_text[:200]}...")
                    parsed_output = {"parse_error": str(e), "raw_output": output_text}

                # Extract the three required fields
                issues_analysis = parsed_output.get("Issues_Analysis", "")
                professional_suggestions = parsed_output.get("Professional_Suggestions", "")
                image_editing_prompt = parsed_output.get("Image_Editing_Prompt", "")

                result["Issues_Analysis"] = format_output_to_string(issues_analysis)
                result["Professional_Suggestions"] = format_output_to_string(professional_suggestions)
                result["Image_Editing_Prompt"] = image_editing_prompt if isinstance(image_editing_prompt, str) else str(image_editing_prompt)
            else:
                parsed_output = {"error": output_text}

            result["model_output_raw"] = output_text
            result["inference_time_s"] = round(elapsed, 2)
            result["processed_by_rank"] = rank

            if inference_info is not None:
                result["generated_token_count"] = inference_info["generated_token_count"]
                result["ended_with_eos_token"] = inference_info["ended_with_eos_token"]
                result["hit_max_new_tokens"] = inference_info["hit_max_new_tokens"]
                result["eos_token_ids"] = inference_info["eos_token_ids"]
                if inference_info["image_grid_thw"] is not None:
                    result["image_grid_thw"] = inference_info["image_grid_thw"]

                if args.save_raw_outputs:
                    result["model_output_with_special_tokens"] = inference_info["output_text_with_special_tokens"]
                    result["generated_token_ids"] = inference_info["generated_token_ids"]

            # Add parse error if exists
            if "parse_error" in parsed_output:
                result["parse_error"] = parsed_output["parse_error"]
            if "error" in parsed_output:
                result["error"] = parsed_output["error"]

            results.append(result)

        # Gather results from all ranks
        if rank == 0:
            print(f"\n[Rank {rank}] Gathering results from all ranks...")

        all_results = gather_results(results, rank, world_size)

        # Only rank 0 saves and prints final results
        if rank == 0:
            # Final save
            output_path = _save_results(all_results, output_json_path, args, world_size)
            print(f"\n{'='*60}")
            print(f"Done! Processed {len(all_results)} samples across {world_size} GPUs.")
            print(f"Results saved to: {output_path}")

            # Print summary statistics
            times = [r["inference_time_s"] for r in all_results if not r.get("model_output_raw", "").startswith("ERROR")]
            if times:
                print(f"\nInference time stats:")
                print(f"  Total:   {sum(times):.1f}s")
                print(f"  Average: {sum(times)/len(times):.1f}s per sample")
                print(f"  Min:     {min(times):.1f}s")
                print(f"  Max:     {max(times):.1f}s")

            errors = [r for r in all_results if r.get("model_output_raw", "").startswith("ERROR")]
            if errors:
                print(f"\n{len(errors)} samples had errors:")
                for r in errors:
                    print(f"  Sample {r['index']}: {r['model_output_raw'][:100]}")

            parse_errors = [r for r in all_results if "parse_error" in r]
            if parse_errors:
                print(f"\n{len(parse_errors)} samples had JSON parsing errors:")
                for r in parse_errors:
                    print(f"  Sample {r['index']}: {r['parse_error'][:100]}")

    finally:
        # Clean up distributed environment
        cleanup_distributed()


def _save_results(results: list[dict], output_json_path: Path, args, world_size: int = 1) -> Path:
    """Save results to JSON file."""
    output_data = {
        "model_path": args.model_path,
        "data_dir": args.data_dir,
        "system_prompt_file": args.system_prompt_file,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "num_samples_limit": args.num_samples,
        "save_raw_outputs": args.save_raw_outputs,
        "num_samples": len(results),
        "world_size": world_size,
        "results": results,
    }
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    return output_json_path


if __name__ == "__main__":
    main()
