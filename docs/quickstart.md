# Run PhotoMaster on a photo

The inference pipeline runs three processes in sequence: VLM guidance, JSON conversion, and reference rendering. Models are released between GPU stages. Multi-GPU mode distributes samples across GPUs; it does not imply that each model is sharded to fit smaller GPUs.

## Environment

Use a dedicated Linux NVIDIA CUDA environment with Python 3.10 or newer. Install a CUDA-compatible PyTorch build, then install `requirements-inference.txt`. This file lists inference dependencies; it is not a tested version lock. Training has separate dependencies and should use a separate environment.

`python scripts/check_setup.py` checks installed package versions, required Qwen3-VL imports, the vendored DiT pipeline import, CUDA, and local input/config paths without loading weights. It does not prove weight integrity or memory fit. One-image and five-image runs completed on a single H20; see the [GPU execution report](gpu-validation.md) for the tested path, measurements and remaining verification.

### Observed single-GPU environment

The October 13, 2026 runs used a fresh AutoDL environment with one NVIDIA H20, Ubuntu 22.04, Python 3.12 and CUDA 12.8. The following versions were recorded for that environment. The exact installation recipe and dependency lock documenting this tested environment remain to be published.

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

Both runs used `GPU_MODE=single` and `FT_CKPT_IS_FULL=1`, loaded the complete 1,933-tensor BF16 checkpoint, and reported zero renderer failures. The five-image run produced five VLM records, five renderer inputs and five readable images. Execution was verified through remote logs; image content and paper benchmark metrics have not been reviewed. The raw evidence archive still awaits retrieval and local verification. Other GPU memory capacities and distributed execution remain untested; the [report](gpu-validation.md) records the observed whole-card memory usage and source revision.

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
  --local-dir models/Qwen-Image-Edit-2511
```

The [PhotoMaster file inventory at this revision](https://huggingface.co/WilsonCahu/PhotoMaster/tree/ba78362f91a8c6856a1a4605a9fef04e78c06e63) was verified on October 13, 2026. The hosted `vlm/` and `i2i/` subdirectories are preserved by downloading into `models/ft_ckpts`; do not download the repository root into `models/ft_ckpts/vlm`. All three commands pin the revisions used by the completed single-GPU runs.

This produces the layout expected by the inference scripts:

```text
models/
├── Qwen3-VL-32B-Instruct/    # base model, processor and tokenizer
├── Qwen-Image-Edit-2511/     # full base diffusion pipeline
└── ft_ckpts/
    ├── vlm/                # PhotoMaster VLM LoRA adapter
    │   ├── adapter_config.json
    │   └── adapter_model.safetensors
    └── i2i/
        └── photomaster_i2i.safetensors
```

A LoRA VLM directory must contain `adapter_config.json` and adapter weights. Its recorded base path may belong to the training machine: the launcher prefers `models/Qwen3-VL-32B-Instruct`, or you can explicitly set `BASE_MODEL`.

The renderer uses `models/ft_ckpts/i2i/photomaster_i2i.safetensors` through `FT_CKPT_PATH`. This file is distinct from the VLM LoRA adapter. The current published file is a complete transformer checkpoint: 1,933 BF16 tensors, approximately 40.86 GB. It was loaded successfully in both recorded H20 runs using the complete-transformer mode below.

### Optional: skip the base transformer weights

For the pinned `WilsonCahu/PhotoMaster` checkpoint above, set **`FT_CKPT_IS_FULL=1`** to load the complete PhotoMaster transformer directly. This checks every key and tensor shape, rejects incomplete checkpoints, and supplies the loaded transformer to the base pipeline. It avoids downloading and loading the additional approximately 40.86 GB of base transformer weights.

To use this mode, replace the final base-model download command above with:

```bash
hf download Qwen/Qwen-Image-Edit-2511 \
  --revision 6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9 \
  --exclude 'transformer/*.safetensors' 'transformer/*.bin' \
  --local-dir models/Qwen-Image-Edit-2511
export FT_CKPT_IS_FULL=1
```

Keep `transformer/config.json`, including `zero_cond_t=true`, as well as the pipeline's text encoder, VAE, tokenizer, processor, scheduler and `model_index.json`. Do not exclude the entire `transformer/` directory or modify the model index. Keep `FT_CKPT_IS_FULL=1` exported for every renderer run when the base transformer weights are absent. The direct Python equivalent is `--full-transformer-checkpoint` together with `--ckpt-path`.

The default remains `FT_CKPT_IS_FULL=0`: the renderer loads the full base transformer and overlays a potentially partial training checkpoint using the original loading path. Use that mode for partial checkpoints and download the complete base pipeline. The full mode does not silently fall back to partial loading when a key or shape differs.

## First run

After placing the checkpoints:

```bash
export FT_CKPT_PATH="$PWD/models/ft_ckpts/i2i/photomaster_i2i.safetensors"
# Explicitly select the complete-transformer mode for the published checkpoint.
export FT_CKPT_IS_FULL=1
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

`FT_CKPT_PATH` must still be set. Keep `FT_CKPT_IS_FULL=1` for the complete-checkpoint path above. The experiment file specifies generation parameters, not the fine-tuned renderer checkpoint or loading mode.

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

This example assumes the complete published PhotoMaster transformer. For a partial training checkpoint, set `FT_CKPT_IS_FULL=0` and retain all base weights. Do not export one shared `MODEL_PATH` for the entire pipeline.

## Troubleshooting

| Symptom | Action |
|---|---|
| Missing Qwen3-VL class or vendored pipeline import | Run the setup checker and record package versions. The inference and training environments may have different compatibility constraints. |
| Missing FlashAttention or a CUDA extension import error | Install a FlashAttention wheel matching the active Python, PyTorch, CUDA and C++ ABI; the VLM worker requests `flash_attention_2`. |
| Full-transformer checkpoint key or shape mismatch | Confirm the exact PhotoMaster revision and 2511 config. A partial checkpoint requires `FT_CKPT_IS_FULL=0` and complete base weights. |
| Missing base transformer weight files | If they were deliberately excluded, keep `FT_CKPT_IS_FULL=1` with the complete PhotoMaster checkpoint. Otherwise download the complete base pipeline. |
| Missing VLM base path | Set `BASE_MODEL` or populate the in-repo base directory. |
| Eight workers launch unexpectedly | Set `GPU_MODE=single`; the legacy renderer launcher defaults to multi-GPU. |
| Empty or malformed VLM response | Inspect `model_output_raw`, `parse_error`, and `hit_max_new_tokens`; retry under a new run name. The converter rejects failed or blank-prompt records. |
| Renderer looks like the base model | Check `FT_CKPT_PATH` and the printed checkpoint loading report. |
| Old images after changing settings | Use a new run name. `FORCE=1` only regenerates the intermediate JSONL, not images. |
| CUDA out of memory | Check available GPU memory and other workloads. No tested quantized or consumer-GPU configuration is supplied. |

The paper's timing on an 80 GB GPU is not a universal performance guarantee. Record GPU type, memory, model revisions, dependency versions, and generation settings when reporting results.
