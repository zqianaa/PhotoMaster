# Run PhotoMaster on a photo

The inference pipeline runs three processes in sequence: VLM guidance, JSON conversion, and reference rendering. Models are released between GPU stages. Multi-GPU mode distributes samples across GPUs; it does not imply that each model is sharded to fit smaller GPUs.

## Environment

Start with the [recorded Linux x86_64 / Python 3.12 / CUDA 12.8 environment recipe](../environments/inference-h20-cu128/README.md). It pins the observed direct dependencies and the compatible FlashAttention wheel. `requirements-inference.txt` alone is an unpinned dependency list and does not install the mandatory FlashAttention binary. Other Python/CUDA combinations have not been validated. Training has separate dependencies and should use a separate environment.

`python scripts/check_setup.py --ft-ckpt models/ft_ckpts/i2i/photomaster_i2i.safetensors` checks installed package versions, required Qwen3-VL imports, the vendored DiT pipeline import, FlashAttention and its native CUDA extension import, CUDA availability, and local input/config paths without loading weights. It does not prove weight integrity or memory fit. One-image and five-image runs completed on a single H20; see the [GPU execution report](gpu-validation.md) for the tested path, measurements and remaining verification.

### Observed single-GPU environment

The October 13, 2026 runs used a fresh AutoDL environment with one NVIDIA H20, Ubuntu 22.04, Python 3.12 and CUDA 12.8. The following versions were recorded for that environment. The [recorded setup recipe, direct dependency constraints, exact wheel hashes and environment inventory](../environments/inference-h20-cu128/README.md) are available. The inventory includes inherited image packages and is not a complete portable lock; the consolidated recipe has not been independently rerun on a clean machine.

| Component | Recorded version |
|---|---|
| PyTorch / torchvision | 2.8.0+cu128 / 0.23.0+cu128 |
| Transformers | 4.57.1 |
| Diffusers | 0.37.0 |
| Accelerate / PEFT | 1.10.1 / 0.17.1 |
| huggingface_hub | 0.36.2 |
| FlashAttention | 2.8.3, with a wheel matching Python, PyTorch, CUDA and the C++ ABI |
| NumPy / Pillow / safetensors | 2.3.2 / 11.3.0 / 0.8.0 |

The VLM worker explicitly uses `flash_attention_2`; FlashAttention must be installed separately from the generic inference dependency list. Use the matching wheel from the [official 2.8.3 release](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3). Diffusers 0.37.0 supports the `zero_cond_t=true` architecture in Qwen-Image-Edit-2511; the 0.36.0 release does not implement this behavior and is unsuitable for this checkpoint.

Both runs used `GPU_MODE=single` and `FT_CKPT_IS_FULL=1`, loaded the complete 1,933-tensor BF16 checkpoint, and reported zero renderer failures. The five-image run produced five VLM records, five renderer inputs and five readable images. The retrieved archive passed all 51 payload hashes and 59 offline consistency checks. [Saved outputs](examples.md) are supplementary; these runs verify execution, not paper benchmark quality. The [published evidence](validation/h20-20261013/README.md) documents the raw outputs and source-proof limits. Other GPU memory capacities and distributed execution remain untested; the [report](gpu-validation.md) records the observed whole-card memory usage and source revision.

## Download and place weights

Use these resource pages:

- [PhotoMaster checkpoints](https://huggingface.co/WilsonCahu/PhotoMaster)
- [Qwen3-VL-32B-Instruct base](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct)
- [Qwen-Image-Edit-2511 base](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)

Run the following commands from the repository root after installing the inference dependencies. The `hf` command is provided by `huggingface_hub`.

```bash
hf download WilsonCahu/PhotoMaster \
  --revision ba78362f91a8c6856a1a4605a9fef04e78c06e63 \
  --local-dir models/ft_ckpts
hf download Qwen/Qwen3-VL-32B-Instruct \
  --revision 0cfaf48183f594c314753d30a4c4974bc75f3ccb \
  --local-dir models/Qwen3-VL-32B-Instruct
hf download Qwen/Qwen-Image-Edit-2511 \
  --revision 6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9 \
  --exclude 'transformer/*.safetensors' 'transformer/*.bin' \
  --local-dir models/Qwen-Image-Edit-2511
```

The [PhotoMaster file inventory at this revision](https://huggingface.co/WilsonCahu/PhotoMaster/tree/ba78362f91a8c6856a1a4605a9fef04e78c06e63) was verified on October 13, 2026. The hosted `vlm/` and `i2i/` subdirectories are preserved by downloading into `models/ft_ckpts`; do not download the repository root into `models/ft_ckpts/vlm`. All three commands pin the revisions used by the completed single-GPU runs.

This produces the layout expected by the inference scripts:

```text
models/
├── Qwen3-VL-32B-Instruct/    # base model, processor and tokenizer
├── Qwen-Image-Edit-2511/     # base pipeline components; PhotoMaster supplies transformer
└── ft_ckpts/
    ├── vlm/                # PhotoMaster VLM LoRA adapter
    │   ├── adapter_config.json
    │   └── adapter_model.safetensors
    └── i2i/
        └── photomaster_i2i.safetensors
```

A LoRA VLM directory must contain `adapter_config.json` and adapter weights. Its recorded base path may belong to the training machine: the launcher prefers `models/Qwen3-VL-32B-Instruct`, or you can explicitly set `BASE_MODEL`.

The renderer uses `models/ft_ckpts/i2i/photomaster_i2i.safetensors` through `FT_CKPT_PATH`. This file is distinct from the VLM LoRA adapter. The current published file is a complete transformer checkpoint: 1,933 BF16 tensors, approximately 40.86 GB. It was loaded successfully in both recorded H20 runs using the complete-transformer mode below.

### Checkpoint loading modes

With no checkpoint override, the launcher selects the published `photomaster_i2i.safetensors` and uses **`FT_CKPT_IS_FULL=1`**. It checks every key and tensor shape and supplies the complete PhotoMaster transformer to the base pipeline. The download above excludes the unused base transformer weights, saving approximately 40.86 GB. The three downloaded model components total approximately 125.5 GB; allow additional space for caches and outputs.

Keep `transformer/config.json`, including `zero_cond_t=true`, plus the text encoder, VAE, tokenizer, processor, scheduler and `model_index.json`. Do not exclude the entire `transformer/` directory or modify the model index.

For an **explicit custom `FT_CKPT_PATH`**, the launcher preserves the partial-checkpoint default (`FT_CKPT_IS_FULL=0`). Set `FT_CKPT_IS_FULL=1` alongside an explicit path to a complete checkpoint, including a relocated copy of the published PhotoMaster checkpoint. Partial checkpoints require the full base pipeline: rerun the base download command without `--exclude`. Full mode never silently falls back to partial loading.

Base-renderer experiments require `ALLOW_BASE_DIT=1`, an unset `FT_CKPT_PATH`, `FT_CKPT_IS_FULL=0`, and the full base pipeline. A missing PhotoMaster checkpoint is an error rather than an automatic base-model fallback. The direct Python equivalent of complete-checkpoint loading is `--full-transformer-checkpoint` together with `--ckpt-path`.

## First run

After placing the checkpoints:

```bash
# The pipeline preflights both models, FlashAttention and CUDA before generation.
# Default model locations above need no path or loading-mode overrides.
NUM_SAMPLES=1 RUN_NAME=smoke_01 \
  bash scripts/inference/run_pipeline.sh vlm photomaster_bench
```

Inspect `bench/photomaster_bench/results/smoke_01/vlm_output.json` for a successful diagnosis and `qwen_edit_images/` for a generated reference. A successful process alone is not evidence that the advice or image is correct.

Use a **new `RUN_NAME` for every whole-pipeline invocation**, including when changing `NUM_SAMPLES` from 1 to 5. The pipeline refuses any existing run directory before starting a stage, even with `FORCE=1`. This prevents new VLM results from being combined with cached JSONL or images. To recover an interrupted run, invoke the unfinished stage directly only while its inputs, model files and settings remain unchanged. Stage 2 reuses its JSONL and stage 3 skips existing expected images; neither validates a configuration fingerprint.

For the recorded inference parameters:

```bash
unset NUM_SAMPLES
set -a
source config/experiments/photomaster_bench_infer.env
set +a
GPU_MODE=single RUN_NAME=photomaster_examples \
  bash scripts/inference/run_pipeline.sh vlm photomaster_bench
```

This runs all five bundled inputs when `NUM_SAMPLES` is unset. If you exported `NUM_SAMPLES` for a smoke test, unset it first. The experiment file specifies generation parameters; the launcher supplies the default PhotoMaster model paths and full-checkpoint mode.

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
  FT_CKPT_IS_FULL=1 GPU_MODE=single \
  bash scripts/inference/run_qwen_edit_infer.sh custom photomaster_bench
```

This example assumes the complete published PhotoMaster transformer. For a partial training checkpoint, set `FT_CKPT_IS_FULL=0` and retain all base weights. The whole-pipeline entry point rejects a shared `MODEL_PATH`, since that variable would select two different model types. Use the individual stages above for custom locations.

## Troubleshooting

| Symptom | Action |
|---|---|
| Missing Qwen3-VL class or vendored pipeline import | Run the setup checker and record package versions. The inference and training environments may have different compatibility constraints. |
| Missing FlashAttention or a CUDA extension import error | Install a FlashAttention wheel matching the active Python, PyTorch, CUDA and C++ ABI; the VLM worker requests `flash_attention_2`. |
| Full-transformer checkpoint key or shape mismatch | Confirm the exact PhotoMaster revision and 2511 config. A partial checkpoint requires `FT_CKPT_IS_FULL=0` and complete base weights. |
| Missing base transformer weight files | If they were deliberately excluded, keep `FT_CKPT_IS_FULL=1` with the complete PhotoMaster checkpoint. Otherwise download the complete base pipeline. |
| Missing VLM base path | Set `BASE_MODEL` or populate the in-repo base directory. |
| Unexpected worker count | Both launchers default to `single`. Check exported `GPU_MODE` and `GPUS_PER_NODE`; multi-GPU is explicitly selected. |
| Empty or malformed VLM response | Inspect `model_output_raw`, `parse_error`, and `hit_max_new_tokens`; retry under a new run name. The converter rejects failed or blank-prompt records. |
| Missing PhotoMaster checkpoint | Download to the documented path or provide `FT_CKPT_PATH` and the correct loading mode. Base rendering requires explicit `ALLOW_BASE_DIT=1`. |
| Result directory already exists | Choose a new `RUN_NAME`. The whole pipeline deliberately refuses reuse; `FORCE=1` applies only to direct stage 2 invocation. |
| CUDA out of memory | Check available GPU memory and other workloads. No tested quantized or consumer-GPU configuration is supplied. |

The paper's timing on an 80 GB GPU is not a universal performance guarantee. Record GPU type, memory, model revisions, dependency versions, and generation settings when reporting results.
