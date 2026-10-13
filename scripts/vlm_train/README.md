# VLM Training — Qwen3-VL LoRA SFT

LoRA fine-tuning of Qwen3-VL-32B-Instruct via LLaMA-Factory, using ShareGPT-format
image-text datasets.

## Layout

```
scripts/vlm_train/
├── launch_qwen3_vl_train.sh    # launch script — env setup + torchrun invocation
└── LLaMA-Factory/              # vendored trainer (hiyouga/LLaMA-Factory @ 1317057)
    ├── src/train.py            # trainer entry point
    └── src/llamafactory/       # full trainer library
config/
├── experiments/*.env           # hyperparameters per experiment
training_data/
└── dataset_info.json           # LLaMA-Factory dataset registry
```

`launch_qwen3_vl_train.sh` activates the conda environment, generates a DeepSpeed ZeRO-2
config, and launches the trainer via `torchrun`.

> This is the research training setup. See [release status](../../docs/release-status.md)
> for missing dataset assets and machine-specific launcher assumptions.

## Prerequisites

| Requirement | Default location | Override |
|---|---|---|
| Conda environment | `<repo>/../../env/photomaster_final` | `PHOTOMASTER_CONDA_ENV` |
| Base model `Qwen3-VL-32B-Instruct` | `<repo>/models/Qwen3-VL-32B-Instruct` | `MODEL_PATH` |
| Trainer | `scripts/vlm_train/LLaMA-Factory/src/train.py` (in this repo) | `SCRIPT_PATH` |
| Dataset registry | `data/dataset_info.json` (must be supplied) | `DATASET_DIR` |

The script verifies that `SCRIPT_PATH` exists and exits early with the missing path if not.

LLaMA-Factory is vendored at `scripts/vlm_train/LLaMA-Factory/` at commit `1317057` with
five local patches (see Vendored trainer section). It must be installed into the conda
environment before first use:

```bash
pip install -e scripts/vlm_train/LLaMA-Factory/
```

## Running

Source an experiment file to set hyperparameters, then run the launch script:

```bash
# Single node (8 GPUs)
set -a; source config/experiments/photomaster_vlm_train.env; set +a
bash scripts/vlm_train/launch_qwen3_vl_train.sh

# Multi-node — set rendezvous vars before launching on each node
set -a; source config/experiments/photomaster_vlm_train.env; set +a
export NNODES=4 NODE_RANK=<rank> MASTER_ADDR=<node0-ip>
bash scripts/vlm_train/launch_qwen3_vl_train.sh
```

Source and export the experiment settings first; override values on the launch command:

```bash
OUTPUT_PATH=/path/to/output \
DATASET_NAME=lingbot_captionV3 \
bash scripts/vlm_train/launch_qwen3_vl_train.sh
```

## Experiment configuration

An experiment file is plain shell assignments, sourced before the launch script. See
`config/experiments/photomaster_vlm_train.env`.

| Variable | Default | Meaning |
|---|---|---|
| `DATASET_NAME` | `lingbot_captionV3,forwardPipeline_captionV3` | Comma-separated names from `dataset_info.json`. Repeat a name to oversample. |
| `OUTPUT_PATH` | — (required) | Checkpoint and log directory. Relative paths resolve against the repo root. |
| `LEARNING_RATE` | `1e-5` | |
| `NUM_TRAIN_EPOCHS` | `3` | |
| `PER_DEVICE_TRAIN_BATCH_SIZE` | `1` | Per-GPU micro-batch. |
| `GRADIENT_ACCUMULATION_STEPS` | `1` | |
| `SAVE_STEPS` | `1000` | Steps between LoRA adapter checkpoints. |
| `CUTOFF_LEN` | `2048` | Token sequence cutoff. |
| `IMAGE_MAX_PIXELS` | `262144` | Max pixels per image (512×512). |
| `MASTER_PORT` | `39987` | Change only on a port clash. |
| `NGPUS_PER_NODE` | `8` | GPUs per node. |

Global batch size is `PER_DEVICE_TRAIN_BATCH_SIZE × GRADIENT_ACCUMULATION_STEPS × total_gpus`.

Dataset ratios: repeating a name in `DATASET_NAME` oversamples it. For example,
`DATASET_NAME=lingbot_captionV3,lingbot_captionV3,forwardPipeline_captionV3` gives a 2:1
mix.

## Dataset configuration

`dataset_info.json` is the LLaMA-Factory dataset registry. It is not included in
this checkout and must be prepared under `DATASET_DIR`. It maps dataset
names to label JSON paths using ShareGPT column mappings:

```json
"lingbot_captionV3": {
  "file_name": "lingbot/v0_20260330/clean_labels_v1_train.json",
  "formatting": "sharegpt",
  "tags": { "role_tag": "from", "content_tag": "value", ... },
  "columns": { "messages": "conversations", "images": "image" }
}
```

Paths in `dataset_info.json` are relative to `--dataset_dir`, which defaults to
`$PHOTOMASTER_DIR/data` in the launcher. Repoint `DATASET_DIR` for a different environment.

Available datasets:

| Name | Description |
|---|---|
| `lingbot_captionV3` | Lingbot data, caption version 3 |
| `lingbot_captionV3_clean` | Filtered subset of above |
| `forwardPipeline_captionV3` | Forward pipeline data, caption version 3 |
| `forwardPipeline_mti1` / `mti2` | Forward pipeline MTI variants |
| `bananaMirror_captionV1` / `captionV2` | BananaMirror data |
| `forwardPipeline_v*_ablation_set` | Ablation subsets |

## Training configuration

The script trains with:
- LoRA rank 64, applied to all linear layers (`--lora_target all`)
- Vision tower and multimodal projector frozen
- Template `qwen3_vl_nothink` (no chain-of-thought)
- DeepSpeed ZeRO stage 2 (generated inline into `$OUTPUT_PATH/ds_z2_config.json`)
- bf16 mixed precision
- Sequence packing enabled (`--packing True`)

## Output

```
$OUTPUT_PATH/
├── ds_z2_config.json              # generated DeepSpeed config
├── training_loss.png              # loss curve (from --plot_loss)
├── trainer_log.jsonl              # per-step metrics
├── checkpoint-<N>/                # every SAVE_STEPS — full LoRA adapter
└── runs/                          # TensorBoard event files
```

To load a checkpoint:

```python
from peft import PeftModel
from transformers import Qwen3VLForConditionalGeneration

base = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_PATH)
model = PeftModel.from_pretrained(base, f"{OUTPUT_PATH}/checkpoint-{N}")
```

## Vendored trainer

LLaMA-Factory is vendored at `scripts/vlm_train/LLaMA-Factory/` from commit `1317057`
(`[feat] support megatron-LM training by mcore_adapter`) of
`https://github.com/hiyouga/LLaMA-Factory`.

Five files carry local patches on top of that commit:

- `src/llamafactory/data/loader.py`
- `src/llamafactory/data/mm_plugin.py`
- `src/llamafactory/data/processor/supervised.py`
- `src/llamafactory/model/model_utils/moe.py`
- `src/train.py`

The built-in `data/` directory (LLaMA-Factory's own dataset templates) is not vendored —
it is not needed when `--dataset_dir` points to this repo's `training_data/`.

Because LLaMA-Factory is vendored, this repository is the source of truth for trainer
behaviour and will not pick up upstream changes automatically.

## GPU memory

LoRA training with ZeRO stage 2 is significantly lighter than full fine-tuning.
The vision tower and projector are frozen, so gradients flow only through the LoRA
adapters. On 80 GB H800s, a single node (8 GPUs) with `PER_DEVICE_TRAIN_BATCH_SIZE=1`
and `IMAGE_MAX_PIXELS=262144` has been used in practice.

Set:

```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

to avoid allocator fragmentation OOMs under ZeRO-2 gradient partitioning.
