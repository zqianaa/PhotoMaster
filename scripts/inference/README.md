# Inference — PDT guidance and visual references

Start with the [quick start](../../docs/quickstart.md) for the recorded installation recipe, pinned weights and single-photo input preparation. The five bundled photos are inference examples, not the 377-case paper benchmark.

## Data flow

| Stage | Launcher | Reads | Writes |
|---|---|---|---|
| 1 | `run_vlm_infer.sh` | `bench/<BENCH>/metadata.json` and images | `results/<RUN_NAME>/vlm_output.json` |
| 2 | `run_vlm_to_dit.sh` | `vlm_output.json` | `dit_input.jsonl` |
| 3 | `run_qwen_edit_infer.sh` | `dit_input.jsonl` and input images | `<SAVE_MODEL_NAME>/` images |

The whole-pipeline entry point runs these sequentially; models are released between GPU processes. Relative image paths in JSONL resolve against the benchmark root, so preserve the layout `bench/<BENCH>/results/<RUN_NAME>/dit_input.jsonl`.

## Whole pipeline

After installing dependencies and downloading weights to the quick-start locations:

```bash
NUM_SAMPLES=1 RUN_NAME=smoke_01 \
  bash scripts/inference/run_pipeline.sh vlm photomaster_bench

# A new run name is required for the five-image run.
unset NUM_SAMPLES
set -a
source config/experiments/photomaster_bench_infer.env
set +a
RUN_NAME=demo_05 \
  bash scripts/inference/run_pipeline.sh vlm photomaster_bench
```

Both positional arguments are optional (`vlm`, `photomaster_bench`). `RUN_NAME` defaults to the model name. The entry point preflights both models' configuration paths, input metadata, package imports, FlashAttention's native extension and CUDA availability before VLM generation. This does not check every weight shard or prove memory fit.

The pipeline **requires a new result directory** and atomically reserves it. Reusing a run name fails before any stage starts, even if `FORCE=1`. This prevents a fresh VLM run from inheriting older JSONL or rendered images. Directory names must start with a letter/digit and contain only letters, digits, `_`, `-`, or `.`.

## Model locations and selection

| Component | Default location | Override |
|---|---|---|
| VLM checkpoint | `<repo>/models/ft_ckpts/<MODEL>` | `MODEL_ROOT`, or stage-1 `MODEL_PATH` |
| Base VLM for LoRA | `<repo>/models/Qwen3-VL-32B-Instruct` | `BASE_MODEL` |
| Base diffusion components | `<repo>/models/Qwen-Image-Edit-2511` | Stage-3 `MODEL_PATH` |
| PhotoMaster transformer | `<repo>/models/ft_ckpts/i2i/photomaster_i2i.safetensors` | `FT_CKPT_PATH` |
| System prompt | `scripts/inference/prompts/system_prompt.txt` | `SYSTEM_PROMPT` |
| Vendored pipeline | `scripts/inference/vendor/qwenimage/pipeline_qwenimage_edit_plus.py` | `PHOTOMASTER_QWEN_EDIT_PIPELINE` |

With no fine-tuned checkpoint override, stage 3 loads the published **complete** PhotoMaster transformer (`FT_CKPT_IS_FULL=1`). Its strict loader validates all tensor keys and shapes; the base transformer weight files may be omitted, but `transformer/config.json` and the other base pipeline components are required.

An explicit `FT_CKPT_PATH` defaults to **partial** loading (`FT_CKPT_IS_FULL=0`) for compatibility with training checkpoints. For a complete checkpoint at a custom path, also set `FT_CKPT_IS_FULL=1`. Partial loading requires all base weights. A missing fine-tuned checkpoint is an error; it never silently selects the base renderer.

For a deliberate base-renderer experiment, download all base weights, unset `FT_CKPT_PATH`, and set `ALLOW_BASE_DIT=1 FT_CKPT_IS_FULL=0`. Base mode conflicts with an explicit fine-tuned path or full-checkpoint mode.

LoRA adapters are detected by `adapter_config.json` and merged on CPU before moving to GPU. The stage-1 launcher prefers the local base VLM; if absent, it falls back to the adapter's recorded path. For whole-pipeline preflight, supply `BASE_MODEL` explicitly when the base is elsewhere.

`MODEL_PATH` has different meanings in the two GPU stages. The whole-pipeline entry point rejects it; use the individual stages for custom locations as shown in the quick start.

## Individual stages and interrupted runs

For a fresh named run, these commands are equivalent to the three processing stages:

```bash
RUN_NAME=staged_demo bash scripts/inference/run_vlm_infer.sh vlm photomaster_bench
bash scripts/inference/run_vlm_to_dit.sh staged_demo photomaster_bench
bash scripts/inference/run_qwen_edit_infer.sh staged_demo photomaster_bench
```

The individual launchers do not perform the whole-pipeline preflight or enforce fresh directories. Resume an unfinished stage only while its inputs, checkpoint files and settings are unchanged. Stage 1 rewrites its VLM JSON; stage 2 preserves an existing JSONL unless `FORCE=1`; stage 3 skips existing expected output files. No configuration fingerprint is checked. Use a new run name when changing settings, prompts, sample counts or weights; `FORCE=1` does not regenerate images.

## Parameters

| Variable | Default | Scope / meaning |
|---|---|---|
| `GPU_MODE` | `single` | Both GPU launchers; explicit `multi_gpu` / `multi_node` available |
| `GPUS_PER_NODE` | 8 | Only used when a distributed mode is selected |
| `NUM_NODES` | 1 | Multi-node only |
| `MASTER_ADDR` / `MASTER_PORT` | `localhost` / 29500 | Multi-node rendezvous |
| `RUN_NAME` | model name | Result directory under the benchmark |
| `MAX_NEW_TOKENS` | 2048 | VLM generation limit |
| `TEMPERATURE` | 0.7 | VLM sampling; 0 selects greedy decoding |
| `TOP_P` / `TOP_K` | 0.8 / 20 | VLM sampling |
| `NUM_SAMPLES` | all | VLM processes the first N samples |
| `PHOTOMASTER_GATHER_DIR` | `<result dir>/.gather` | Stage-1 shard exchange; shared across nodes |
| `FORCE` | 0 | Direct stage 2 only: 1 overwrites existing JSONL |
| `FT_CKPT_IS_FULL` | 1 for default checkpoint; 0 for explicit custom path | Complete transformer versus partial overlay |
| `ALLOW_BASE_DIT` | 0 | Explicit opt-in to rendering without PhotoMaster weights |
| `SAVE_MODEL_NAME` | `qwen_edit_images` | Output subdirectory |
| `TRUE_CFG_SCALE` | 4.0 | Renderer true CFG |
| `INFERENCE_STEPS` | 50 | Denoising steps |
| `SEED` | 0 | Renderer seed, recreated for each sample |
| `QWEN_OUTPUT_RESOLUTION` | 1328 | Square-equivalent output size, 1024 or 1328 |
| `QWEN_VAE_INPUT_RESOLUTION` | 1328 | Square-equivalent VAE input size, 1024 or 1328 |

Only single-GPU execution has been validated on hardware. Multi-GPU mode distributes samples, not model tensors: each worker must fit the model on its GPU. The empty-rank collective ordering has CPU mock coverage, not real NCCL validation.

`run_pipeline.sh` supports a single host (`single` or `multi_gpu`). Multi-node jobs require launching each GPU stage on all participating nodes with shared model/data/output storage and the same rendezvous configuration. The VLM exchanges result shards through the filesystem, so `PHOTOMASTER_GATHER_DIR` must be visible to every node. A node-local `/tmp` is unsuitable.

## Output and failure handling

The VLM keeps per-sample generation and parsing errors in its JSON. A valid renderer prompt must come from a top-level JSON object and be a nonempty string; null, arrays and non-string prompts are rejected without interrupting the rest of the batch. Invalid records retain `model_output_raw` and `parse_error` or `error`. This is not full PDT schema validation. Token-cap hits are recorded as `hit_max_new_tokens`.

The converter rejects failed records, malformed/blank prompts and duplicate output names before writing its JSONL. The renderer exits nonzero on any reported generation failure or missing expected output path. Unrelated images in the output directory cannot satisfy that check. Existing output files are not decoded or content-validated by the resume logic.

Output area approximates `resolution²`; aspect ratio follows the first input and dimensions round to multiples of 16. Filenames follow `save_name`, falling back to `<query_id>.png`. Stage 3 seeds each sample separately, but identical pixels across hardware or dependency versions are not guaranteed. VLM generation is sampled by default; `TEMPERATURE=0` selects greedy decoding.

See the [execution report](../../docs/gpu-validation.md) for the exact GPU-tested source scope and the subsequent CPU-tested workflow fixes.
