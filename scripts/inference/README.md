# Inference — PDT guidance and visual references

Three-stage benchmark pipeline. A Qwen3-VL model critiques a photograph and writes an
editing prompt; that prompt is reformatted into DiT input; Qwen-Image-Edit-2511 renders
a visual reference for reshooting.

For installation, checkpoints, and a single-photo run, start with the
[quick start](../../docs/quickstart.md). The bundled benchmark directory contains
five inference examples, not the complete 377-case paper evaluation set.

## Layout

```
scripts/inference/
├── run_pipeline.sh                  # runs all three stages for one (MODEL, BENCH)
├── run_vlm_infer.sh                 # stage 1 — VLM critique
├── run_vlm_to_dit.sh                # stage 2 — format conversion
├── run_qwen_edit_infer.sh           # stage 3 — image editing
├── vlm_infer.py                     # stage 1 worker
├── vlm_to_dit_bench_reformat.py     # stage 2 worker
├── qwen_edit_infer.py               # stage 3 worker
├── prompts/system_prompt.txt        # analysis framework
└── vendor/qwenimage/                # vendored diffusers pipeline patch
config/experiments/*.env             # per-run hyperparameters
bench/<BENCH>/                       # benchmark data and results
```

## Data flow

| Stage | Script | Reads | Writes |
|---|---|---|---|
| 1 | `run_vlm_infer.sh` | `bench/<BENCH>/metadata.json`, `bench/<BENCH>/img/` | `results/<RUN_NAME>/vlm_output.json` |
| 2 | `run_vlm_to_dit.sh` | `results/<RUN_NAME>/vlm_output.json` | `results/<RUN_NAME>/dit_input.jsonl` |
| 3 | `run_qwen_edit_infer.sh` | `results/<RUN_NAME>/dit_input.jsonl` | `results/<RUN_NAME>/<SAVE_MODEL_NAME>/` |

All three stages key off one result directory, `bench/<BENCH>/results/<RUN_NAME>/`. That
directory is the only handle shared between them, so the chain can be re-entered at any
stage by calling that stage's script directly.

`RUN_NAME` defaults to `MODEL` in stage 1 and is the first positional argument to stages 2
and 3. `run_pipeline.sh` resolves it once and exports it so all three agree.

### Layout constraint

`dit_input.jsonl` records image paths relative to the bench root (`img/foo.jpg`). Stage 3
resolves them two parents above the JSONL-containing directory, so the result directory must stay
exactly two levels below the bench root. A flatter layout breaks image resolution.

## Prerequisites

| Requirement | Default location | Override |
|---|---|---|
| VLM LoRA adapter | `<repo>/models/ft_ckpts/<MODEL>` | `MODEL_ROOT`, or `MODEL_PATH` for a full path |
| Base model for LoRA adapters | `<repo>/models/Qwen3-VL-32B-Instruct`, then `base_model_name_or_path` from the adapter's `adapter_config.json` | `BASE_MODEL` |
| Qwen-Image-Edit-2511 weights | `<repo>/models/Qwen-Image-Edit-2511` | `MODEL_PATH` |
| Fine-tuned DiT checkpoint | none — stage 3 runs the base model | `FT_CKPT_PATH` |
| System prompt | `prompts/system_prompt.txt` | `SYSTEM_PROMPT` |
| Qwen Edit pipeline | `vendor/qwenimage/pipeline_qwenimage_edit_plus.py` | `PHOTOMASTER_QWEN_EDIT_PIPELINE` |

Every default resolves inside `<repo>/models/`, so a populated `models/` needs no
variables at all:

```
models/
├── Qwen3-VL-32B-Instruct/          base VLM, stage 1
├── Qwen-Image-Edit-2511/           base DiT, stage 3
└── ft_ckpts/
    ├── vlm/                        LoRA adapter, the default MODEL
    └── i2i/                        I2I finetune, pass one via FT_CKPT_PATH
```

`models/` is gitignored — it is a layout you populate, not one the repository ships.

LoRA checkpoints are detected automatically by the presence of `adapter_config.json`. The
adapter is merged into the base model on CPU before the merged model moves to GPU, which
avoids holding both copies in device memory at once. `adapter_config.json` records the base
model path from the machine that trained it, so stage 1 prefers the in-repo base model when
`models/Qwen3-VL-32B-Instruct` exists and only falls back to the recorded path otherwise.

Python dependencies: `torch`, `transformers`, `diffusers`, `peft` (LoRA only),
`safetensors`, `Pillow`, and optionally `qwen_vl_utils` (stage 1 falls back to a built-in
image collector when it is absent).

## Usage

Whole pipeline with the recorded generation settings (set `FT_CKPT_PATH` to
the PhotoMaster checkpoint and `GPU_MODE=single` for a single-GPU run):

```bash
set -a; . config/experiments/photomaster_bench_infer.env; set +a
bash scripts/inference/run_pipeline.sh vlm photomaster_bench
```

Both arguments are optional and already default to those values, so bare
`bash scripts/inference/run_pipeline.sh` does the same thing.

Individual stages:

```bash
# Stage 1 — 8 GPUs on one node
GPU_MODE=multi_gpu bash scripts/inference/run_vlm_infer.sh vlm photomaster_bench

# Stage 2 — cheap, CPU only
bash scripts/inference/run_vlm_to_dit.sh vlm photomaster_bench

# Stage 3 — resumes by skipping images that already exist
bash scripts/inference/run_qwen_edit_infer.sh vlm photomaster_bench

# Stage 3 against an I2I finetune instead of the base DiT
FT_CKPT_PATH=models/ft_ckpts/i2i/epoch-2.safetensors \
    bash scripts/inference/run_qwen_edit_infer.sh vlm photomaster_bench
```

Checkpoints elsewhere on disk are reached with `MODEL_ROOT` (a directory of VLM
checkpoints) or `MODEL_PATH` (one full path). `MODEL_PATH` means different things to the two
GPU stages — the VLM checkpoint in stage 1, the DiT weights in stage 3 — so set it on a
stage script directly rather than on `run_pipeline.sh`, which passes it to both.

Smoke test on two samples before committing a GPU allocation:

```bash
NUM_SAMPLES=2 bash scripts/inference/run_vlm_infer.sh vlm photomaster_bench
```

## Environment reference

Stage 1 — `run_vlm_infer.sh`:

| Variable | Default | Purpose |
|---|---|---|
| `MODEL` | `vlm` | Checkpoint directory name; also the first positional argument |
| `MODEL_ROOT` | `<repo>/models/ft_ckpts` | Directory holding VLM checkpoints |
| `MODEL_PATH` | `$MODEL_ROOT/$MODEL` | Full checkpoint path, bypassing `MODEL_ROOT` |
| `BASE_MODEL` | `<repo>/models/Qwen3-VL-32B-Instruct`, else from `adapter_config.json` | Base model for a LoRA adapter |
| `RUN_NAME` | `$MODEL` | Result directory name |
| `SYSTEM_PROMPT` | `system_prompt.txt` | Relative names resolve against `prompts/` |
| `MAX_NEW_TOKENS` | 2048 | Generation cap; the script warns when a sample hits it |
| `TEMPERATURE` | 0.7 | 0 selects greedy decoding |
| `TOP_P` / `TOP_K` | 0.8 / 20 | Sampling, ignored when `TEMPERATURE=0` |
| `NUM_SAMPLES` | all | Process only the first N samples |
| `PHOTOMASTER_GATHER_DIR` | `<result dir>/.gather` | Where ranks exchange shards |

Stage 2 — `run_vlm_to_dit.sh`:

| Variable | Default | Purpose |
|---|---|---|
| `FORCE` | 0 | Set to 1 to overwrite an existing `dit_input.jsonl` |

Stage 3 — `run_qwen_edit_infer.sh`:

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `<repo>/models/Qwen-Image-Edit-2511` | Qwen-Image-Edit-2511 weights |
| `FT_CKPT_PATH` | none — base model runs | Fine-tuned DiT `.safetensors`, e.g. `models/ft_ckpts/i2i/epoch-2.safetensors` |
| `SAVE_MODEL_NAME` | `qwen_edit_images` | Output subfolder inside the result directory |
| `TRUE_CFG_SCALE` | 4.0 | True CFG scale |
| `INFERENCE_STEPS` | 50 | Denoising steps |
| `SEED` | 0 | Per-sample seed |
| `QWEN_OUTPUT_RESOLUTION` | 1328 | Square-equivalent output size: 1024 or 1328 |
| `QWEN_VAE_INPUT_RESOLUTION` | 1328 | Square-equivalent VAE input size: 1024 or 1328 |

Both GPU-bound stages take `GPU_MODE` (`single` / `multi_gpu` / `multi_node`),
`GPUS_PER_NODE`, `NUM_NODES` and `MASTER_ADDR`. Stage 1 defaults to `single`, stage 3 to
`multi_gpu`.

### Multi-node gather

Stage 1 ranks exchange their result shards through the filesystem rather than NCCL, so
`PHOTOMASTER_GATHER_DIR` must point at storage every node can see. The default puts it
inside the result directory, which satisfies this whenever the bench itself is on shared
storage. Pointing it at a node-local path such as `/tmp` silently loses the shards of every
non-master node.

## Reproducibility

Stage 3 builds a fresh `torch.Generator(device="cpu")` per sample from the same seed, so a
given sample renders identically regardless of how many GPUs the run is sharded across.

Stage 1 is sampled (`TEMPERATURE=0.7` by default) and is therefore not bit-reproducible.
Set `TEMPERATURE=0` for greedy decoding when comparing checkpoints.

Output sizes are derived per sample: the target area is `resolution²`, and the aspect ratio
is copied from the first input image, then rounded to a multiple of 16.

Each output file takes its name from the sample's `save_name`, so the image format follows
the source filename — a benchmark mixing JPEG and PNG sources yields a mix of both. Samples
without a `save_name` fall back to `<query_id>.png`.

## Failure handling

Stage 1 records per-sample errors in the output JSON rather than aborting the run: a sample
whose generation raised carries `model_output_raw` beginning with `ERROR`, and one whose
output would not parse as JSON carries `parse_error` alongside the raw text. The run
summary counts both. Samples that hit `MAX_NEW_TOKENS` are flagged with
`hit_max_new_tokens` — their JSON is usually truncated and will show up as a parse error.

Stage 3 skips samples whose output image already exists, so re-running the same command
resumes an interrupted run. It exits non-zero if any sample failed or if the final image
count is short of the input sample count.

Stage 2 rejects failed VLM records, parse errors, empty rendering prompts and
duplicate output names before writing its JSONL. Use a new `RUN_NAME` whenever
model checkpoints or generation settings change to avoid stale cached outputs.
