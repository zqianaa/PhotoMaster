# Single-GPU execution report

The one-image smoke run and five-image demo run completed on **one NVIDIA H20** on October 13, 2026. Both executed VLM guidance, JSON conversion and reference rendering with `GPU_MODE=single` and `FT_CKPT_IS_FULL=1`. The run sequence reported `succeeded`, exit code `0`, and completion at **19:49:56 Asia/Shanghai (UTC+08:00)**.

This report is based on status, logs and validation summaries inspected through the remote UI. The raw logs and results have been packaged for retrieval, but the archive has not yet been retrieved and verified locally or published. Image content has not been reviewed, and no PhotoMaster-Bench metric reproduction is claimed.

## Execution results

| Run | Renderer successes / failures | Whole-pipeline wall time | Renderer script total |
|---|---:|---:|---:|
| `smoke_01` — one image | 1 / 0 | 10 min 58.182 s | 6 min 38 s |
| `demo_05` — five images | 5 / 0 | 36 min 28.534 s | 29 min 4 s |

The five-image validation reported **five VLM records, five renderer input records and five readable output images**, with no missing outputs. Both runs reported successful loading of the full fine-tuned transformer, with all **1,933 tensor keys and shapes matching** and `zero_cond_t=true`.

Whole-pipeline wall time covers the VLM, conversion and renderer stages, including model loading. The renderer script total also includes pipeline loading and output handling. These measurements do not isolate denoising latency and should not be compared directly with the paper's per-stage inference timings.

## GPU memory observations

The device reported **97,871 MiB total GPU memory**. Memory usage was sampled through `nvidia-smi` at approximately one-second intervals over each run.

| Run | CSV samples | Highest observed whole-card used memory |
|---|---:|---:|
| `smoke_01` | 659 | 65,561 MiB |
| `demo_05` | 2,188 | 67,781 MiB |

These values are sampled whole-card used memory, including memory outside the model's PyTorch allocator. Sampling can miss shorter peaks. They are not allocator peak measurements or a minimum VRAM requirement. This H20 had sufficient memory for the tested settings; other GPU models, smaller memory capacities and distributed execution have not been verified.

## Source, weights and settings

The executed source was commit **`01f7eed` plus the complete-transformer loader patch**. It was not an unmodified release commit. The patch loads the full PhotoMaster transformer directly after strict key/shape checks, avoiding the additional base transformer weights. The default partial-checkpoint loading mode was not exercised by these GPU runs.

| Resource | Pinned revision |
|---|---|
| [Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct/tree/0cfaf48183f594c314753d30a4c4974bc75f3ccb) | `0cfaf48183f594c314753d30a4c4974bc75f3ccb` |
| [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511/tree/6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9) | `6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9` |
| [PhotoMaster checkpoints](https://huggingface.co/WilsonCahu/PhotoMaster/tree/ba78362f91a8c6856a1a4605a9fef04e78c06e63) | `ba78362f91a8c6856a1a4605a9fef04e78c06e63` |

The PhotoMaster files were `vlm/adapter_config.json`, `vlm/adapter_model.safetensors` and `i2i/photomaster_i2i.safetensors`. The complete DiT checkpoint contains 1,933 BF16 tensors; the 2511 transformer configuration retains `zero_cond_t=true`.

| Setting | Value |
|---|---|
| GPU mode / transformer loading | `single` / `FT_CKPT_IS_FULL=1` |
| Maximum new VLM tokens | 2,048 |
| VLM temperature / top-p / top-k | 0.7 / 0.8 / 20 |
| Renderer steps | 50 |
| True CFG scale | 4.0 |
| Seed | 0 |
| Output / VAE input resolution | 1328 / 1328 square-equivalent; aspect ratio follows the input |

## Recorded environment

The fresh AutoDL environment used Ubuntu 22.04, Python 3.12 and CUDA 12.8.

| Package | Recorded version |
|---|---|
| torch | 2.8.0+cu128 |
| torchvision | 0.23.0+cu128 |
| transformers | 4.57.1 |
| diffusers | 0.37.0 |
| accelerate | 1.10.1 |
| peft | 0.17.1 |
| huggingface_hub | 0.36.2 |
| flash-attn | 2.8.3 |
| numpy | 2.3.2 |
| Pillow | 11.3.0 |
| safetensors | 0.8.0 |

These are observed versions for the completed runs. The exact installation recipe and dependency lock remain to be published with their tested hardware and inference scope; these runs do not establish compatibility with other environments. FlashAttention also requires a wheel matching the Python, PyTorch, CUDA and C++ ABI versions; a package version alone does not identify that binary.

## Remaining acceptance work

- Retrieve the raw evidence archive, verify its contents and checksums locally, and publish the appropriate reproducibility artifacts.
- Inspect the generated advice and images for quality, instruction following and artifacts. File readability and successful execution establish only that outputs were produced.
- Match the recorded executed-source hashes to the release commit and patch, then publish the exact installation recipe and dependency lock with the scope verified by these runs.
- Release and evaluate the full 377-case benchmark before claiming reproduction of paper metrics. The five bundled inference examples do not provide that evaluation.

The [quick start](quickstart.md) documents the tested complete-checkpoint mode. The [release checklist](release-status.md) retains the remaining licensing, data, training and publication work.
