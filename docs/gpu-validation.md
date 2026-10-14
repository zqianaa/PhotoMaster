# Single-GPU execution report

The one-image smoke run and five-image demo run completed on **one NVIDIA H20** on October 13, 2026. Both executed VLM guidance, JSON conversion and reference rendering with `GPU_MODE=single` and `FT_CKPT_IS_FULL=1`. The run sequence reported `succeeded`, exit code `0`, and completion at **19:49:56 Asia/Shanghai (UTC+08:00)**.

The archive was retrieved on October 14, 2026. All **51 payload files** match their recorded sizes and SHA-256 hashes, with zero missing files; **59 offline consistency checks** passed. The [selected raw evidence and audit](validation/h20-20261013/README.md) and [all five generated comparisons](examples.md) are published in this branch. Qualitative inspection found partial instruction following and visible failures. No PhotoMaster-Bench metric reproduction is claimed.

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

The executed source was commit **`01f7eed` plus the complete-transformer loader patch**. The recorded patch exactly matches the two inference-file changes from `01f7eed` to **`e7c2e15c080396a1cb04e09fac0ae7a870eba261`**. All three recorded source hashes (worker, launcher and full-transformer test) also match that commit. The helper copied pre-existing provenance records; this verifies their correspondence, not an independent snapshot of every remotely executed file. The remote system prompt and complete source tree were not archived. The patch loads the full PhotoMaster transformer directly after strict key/shape checks, avoiding the additional base transformer weights. The partial-checkpoint loading mode was not exercised by these GPU runs.

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

The [environment record](../environments/inference-h20-cu128/README.md) supplies the observed inventory, direct package constraints, exact official wheel URLs/hashes and a consolidated installation recipe. The successful runs reused the image's PyTorch through a `--system-site-packages` virtual environment. The inventory contains inherited packages and image-local build references; it is not a complete portable lock, and the consolidated clean-machine alternative has not been rerun. FlashAttention requires the matching Python 3.12 / PyTorch 2.8 / CUDA 12 / C++ ABI TRUE binary.

## Output inspection and limitations

All six VLM responses across the two runs parse as JSON and supply the exact nonempty prompts used by the renderer. They end with EOS without reaching the token cap. Four responses contain five inactive `Attribution: "None"` string fields where the reference prompt specifies arrays. The current converter accepts them; parsing success is not strict schema compliance.

All six output images decode successfully, and all five input files match the repository examples. The [five-image gallery](examples.md) shows both useful changes and errors: the motorcycle output invents a helmet-like object and changes mechanical details; the cafe output does not follow the arm-placement instruction; the stone-wall output crops closer despite advice to widen the shot. Faces and scene details can be regenerated. These observations are assistant qualitative inspection, not a blinded evaluation or a measured success rate.

The original archive SHA-256 is `06e4798d6aea219146fa1f1b4ac0e89765f6946ec894a90525eec30183735ade` (9,265,781 bytes). Internal manifest hashes establish file consistency, not independent authenticity. The published directory contains a selected evidence subset and derived audit; model tensors and an independently captured complete remote source tree are absent.

## Subsequent workflow changes

After retrieving the evidence, this branch changed launcher defaults to single GPU and the complete published checkpoint, added automatic preflight and fresh-directory enforcement, checked FlashAttention's extension import, rejected non-object/non-string VLM responses, and made empty renderer ranks join the same collectives. Renderer completion now checks every expected output path. These changes passed **60 CPU regression tests**, including mocked launchers and distributed calls, plus shell syntax and documentation command checks. The six recorded VLM responses retain identical parsed fields under the new parser.

These workflow changes were **not rerun on a GPU**. The full-transformer loading implementation remains unchanged from the GPU-tested path; the historical logs and hashes above identify the earlier source. CPU mocks do not validate NCCL, actual model generation or a clean-machine installation.

## Remaining acceptance work

- Independently GPU-validate the subsequent workflow revision if claiming that exact revision as end-to-end tested; strict PDT schema validation and real distributed execution remain outside this run.
- Provide a complete portable dependency lock and independently test the consolidated installation recipe when claiming clean-machine reproducibility.
- Release and evaluate the full 377-case benchmark before claiming reproduction of paper metrics. The five bundled inference examples do not provide that evaluation.

The [quick start](quickstart.md) documents the tested complete-checkpoint mode. The [release checklist](release-status.md) retains the remaining licensing, data, training and publication work.
