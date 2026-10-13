# Run PhotoMaster on a photo

The inference pipeline runs three processes in sequence: VLM guidance, JSON conversion, and reference rendering. Models are released between GPU stages. Multi-GPU mode distributes samples across GPUs; it does not imply that each model is sharded to fit smaller GPUs.

## Environment

Use a dedicated Linux NVIDIA CUDA environment with Python 3.10 or newer. Install a CUDA-compatible PyTorch build, then install `requirements-inference.txt`. This file lists inference dependencies; it is not a tested version lock. Training has separate dependencies and should use a separate environment.

`python scripts/check_setup.py` checks installed package versions, required Qwen3-VL imports, the vendored DiT pipeline import, CUDA, and local input/config paths without loading weights. It does not prove weight integrity or memory fit. A fresh-environment GPU smoke run is still required before declaring a supported setup.

## Download and place weights

Use these resource pages:

- [PhotoMaster checkpoints](https://huggingface.co/WilsonCahu/PhotoMaster)
- [Qwen3-VL-32B-Instruct base](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct)
- [Qwen-Image-Edit-2511 base](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)

The inference scripts expect the following local layout:

```text
models/
├── Qwen3-VL-32B-Instruct/    # base model, processor and tokenizer
├── Qwen-Image-Edit-2511/     # full base diffusion pipeline
└── ft_ckpts/
    └── vlm/                # PhotoMaster VLM adapter or full model
```

A LoRA VLM directory must contain `adapter_config.json` and adapter weights. Its recorded base path may belong to the training machine: the launcher prefers `models/Qwen3-VL-32B-Instruct`, or you can explicitly set `BASE_MODEL`.

The renderer loads a base pipeline plus a fine-tuned transformer `.safetensors` file through `FT_CKPT_PATH`. This is distinct from a VLM LoRA adapter. **Exact filenames and subdirectories in the hosted PhotoMaster repository still need to be mapped to this layout.** Do not assume that downloading its root into `ft_ckpts/vlm` produces a valid VLM adapter.

## First run

After placing the checkpoints:

```bash
export FT_CKPT_PATH=/absolute/path/to/photomaster-dit.safetensors
python scripts/check_setup.py --ft-ckpt "$FT_CKPT_PATH"
GPU_MODE=single NUM_SAMPLES=1 RUN_NAME=smoke_01 \
  bash scripts/inference/run_pipeline.sh vlm photomaster_bench
```

Inspect `bench/photomaster_bench/results/smoke_01/vlm_output.json` for a successful diagnosis and `qwen_edit_images/` for a generated reference. A successful process alone is not evidence that the advice or image is correct.

Use a **new `RUN_NAME` whenever you change checkpoints, prompts, or generation parameters**. Stage 2 preserves an existing JSONL unless `FORCE=1`, and stage 3 skips existing images. Reusing a result directory for a different configuration can combine stale results with a new VLM run.

For the recorded inference parameters:

```bash
set -a
source config/experiments/photomaster_bench_infer.env
set +a
GPU_MODE=single RUN_NAME=photomaster_examples \
  bash scripts/inference/run_pipeline.sh vlm photomaster_bench
```

`FT_CKPT_PATH` must still be set. The experiment file specifies generation parameters, not the fine-tuned renderer checkpoint.

## Your own image

```bash
python scripts/inference/prepare_input.py /path/to/photo.jpg --name demo_portrait
python scripts/check_setup.py --data-only --data-dir bench/demo_portrait
GPU_MODE=single RUN_NAME=first_run \
  bash scripts/inference/run_pipeline.sh vlm demo_portrait
```

The preparation command writes a new benchmark directory, normalizes EXIF orientation, and saves an RGB PNG without the source metadata. It preserves the original photo and refuses to overwrite an existing demo. Only supported Pillow image formats can be opened; convert unsupported phone formats to JPEG or PNG first.

## Text guidance only

```bash
GPU_MODE=single RUN_NAME=text_only \
  bash scripts/inference/run_vlm_infer.sh vlm photomaster_bench
```

This runs only the Decision Reasoner. Neither the base DiT nor the PhotoMaster DiT checkpoint is needed for this command.

## Checkpoints in other locations

`MODEL_PATH` has different meanings in stage 1 and stage 3. Set it separately:

```bash
MODEL_PATH=/models/photomaster-vlm BASE_MODEL=/models/qwen3-vl \
  GPU_MODE=single RUN_NAME=custom \
  bash scripts/inference/run_vlm_infer.sh vlm photomaster_bench
bash scripts/inference/run_vlm_to_dit.sh custom photomaster_bench
MODEL_PATH=/models/qwen-image-edit FT_CKPT_PATH=/models/photomaster-dit.safetensors \
  GPU_MODE=single bash scripts/inference/run_qwen_edit_infer.sh custom photomaster_bench
```

Do not export one shared `MODEL_PATH` for the entire pipeline.

## Troubleshooting

| Symptom | Action |
|---|---|
| Missing Qwen3-VL class or vendored pipeline import | Run the setup checker and record package versions. The inference and training environments may have different compatibility constraints. |
| Missing VLM base path | Set `BASE_MODEL` or populate the in-repo base directory. |
| Eight workers launch unexpectedly | Set `GPU_MODE=single`; the legacy renderer launcher defaults to multi-GPU. |
| Empty or malformed VLM response | Inspect `model_output_raw`, `parse_error`, and `hit_max_new_tokens`; retry under a new run name. The converter rejects failed or blank-prompt records. |
| Renderer looks like the base model | Check `FT_CKPT_PATH` and the printed checkpoint loading report. |
| Old images after changing settings | Use a new run name. `FORCE=1` only regenerates the intermediate JSONL, not images. |
| CUDA out of memory | Check available GPU memory and other workloads. No tested quantized or consumer-GPU configuration is supplied. |

The paper's timing on an 80 GB GPU is not a universal performance guarantee. Record GPU type, memory, model revisions, dependency versions, and generation settings when reporting results.
