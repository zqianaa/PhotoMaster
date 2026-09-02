# DiT Training — Qwen-Image-Edit-2511

Multi-node image-editing (I2I) fine-tuning of Qwen-Image-Edit-2511 on Slurm.

## Layout

```
scripts/dit_train/
├── train_qwen_edit_2511.slurm        # sbatch entry point — orchestrates the job
├── launch_qwen_edit_2511_train.sh    # per-rank worker — env setup + trainer invocation
└── trainer/                          # vendored training code
    ├── train.py                      # trainer entry point
    ├── utils/                        # training loop, checkpoint logger
    ├── pipelines/                    # QwenImagePipeline
    └── data/                         # dataset, resolution bucketing, samplers
        ├── tar_parquet_data_v2.py    # the live dataset (zip + parquet)
        ├── resolution_bucketing.py   # bucketers and batch samplers
        └── indexed_tarfile_reader.py
config/
├── data/*.yaml                       # dataset definitions
└── experiments/*.env                 # hyperparameters per experiment
```

`train_qwen_edit_2511.slurm` allocates the job, computes the rendezvous address, and
launches one task per GPU with `srun`. Each task runs
`launch_qwen_edit_2511_train.sh`, which activates the conda environment, locates the
base model, writes the DeepSpeed and Accelerate configs, and calls the trainer.

The worker also runs standalone without Slurm (single node, via `accelerate launch`):

```bash
OUTPUT_PATH=/path/to/output \
DATA_CONFIG_PATH=config/data/lingbot_dataAblation.yaml \
bash scripts/dit_train/launch_qwen_edit_2511_train.sh
```

## Prerequisites

| Requirement | Default location | Override |
|---|---|---|
| Conda environment | `<dataset>/wilson/env/photomaster_final` | `PHOTOMASTER_CONDA_ENV` |
| Base model `Qwen-Image-Edit-2511` | `<repo>/models/Qwen-Image-Edit-2511`, then `<repo>/models/base_models/Qwen-Image-Edit-2511`, then `<dataset>/dialog_edit_env/ckpts/Qwen/Qwen-Image-Edit-2511` | `QWEN_EDIT_BASE_MODEL_DIR` |
| Training code | `scripts/dit_train/trainer` (in this repo) | `TRAINER_DIR` |

The worker verifies that all 5 transformer shards, all 4 text-encoder shards, and the
VAE weights are present, and fails early with the missing path if not.

The trainer itself is vendored under `scripts/dit_train/trainer`, so no external
checkout is needed. It still depends on `diffsynth` (2.0.1), which is a pip package
installed in the conda environment — that is where the diffusion loss and the noise
and timestep sampling live.

## Running

Node and GPU counts must be passed on the command line — `#SBATCH` directives are
parsed before the script executes, so they cannot come from the experiment file.

```bash
# 4 nodes x 8 GPUs
sbatch --nodes=4 --ntasks-per-node=8 --gres=gpu:8 \
    scripts/dit_train/train_qwen_edit_2511.slurm \
    config/experiments/lingbot_ablation.env

# single node, using the built-in #SBATCH defaults
sbatch scripts/dit_train/train_qwen_edit_2511.slurm \
    config/experiments/lingbot_ablation.env
```

`--ntasks-per-node` must equal the GPU count per node: the trainer runs one process
per GPU. World size, `GPUS_PER_NODE`, and the master address are all derived from the
allocation, so they cannot drift out of sync with the flags you passed.

## Experiment configuration

An experiment file is plain shell assignments, sourced by the sbatch script. See
`config/experiments/lingbot_ablation.env`.

| Variable | Default | Meaning |
|---|---|---|
| `DATA_CONFIG_PATH` | `config/data/lingbotForward_captionV3.yaml` | Dataset config. Relative paths resolve against the repo root. |
| `OUTPUT_PATH` | — (required) | Checkpoints, logs, resolved configs. |
| `LEARNING_RATE` | `1e-5` | |
| `NUM_EPOCHS` | `2` | |
| `BATCH_SIZE_PER_GPU` | `1` | Per-GPU micro-batch. |
| `GRADIENT_ACCUMULATION_STEPS` | `1` | |
| `CHECKPOINTING_STEPS` | `1000` | Steps between checkpoints. Each is ~40.8 GB. |
| `QWEN_TRAIN_RESOLUTION_PRESET` | `qwen_edit_1024` | Resolution bucketing. Use `qwen_edit_2511_native` for 2511. |
| `MASTER_PORT` | `21638` | Change only on a port clash. |
| `POST_SUCCESS_GRACE_SECONDS` | `600` | Teardown grace after training completes. |

Global batch size is `BATCH_SIZE_PER_GPU x GRADIENT_ACCUMULATION_STEPS x total_gpus`.
Steps per epoch is `dataset_size / global_batch_size`, so **changing the node count
changes the step count and the loss trajectory**. Keep it fixed when comparing runs.

## Dataset configuration

`config/data/*.yaml` lists one entry per dataset:

```yaml
datset_configs:
  - dataset_name: lingbot_captionV3
    image_dir: /opt/huawei/.../training_data/lingbot/v2_captionV3/dit_qwen/zip
    annotation_dir: /opt/huawei/.../training_data/lingbot/v2_captionV3/dit_qwen/parquet
```

`image_dir` holds zipped images, `annotation_dir` the parquet annotations. Paths are
absolute and cluster-specific — repoint them for a different environment. (`datset_configs`
is spelled that way in the trainer's schema; it is not a typo to fix here.)

Available configs:

| Config | Datasets |
|---|---|
| `lingbot_dataAblation.yaml` | lingbot captionV3 only |
| `forward_dataAblation.yaml` | forward-pipeline ablation |
| `lingbotForward_captionV3.yaml` | lingbot + forward, captionV3 |
| `lingbotForward_dataAblation.yaml` | combined ablation |
| `allDataV3.yaml` | captionV3 + mti1 + mti2 |

## Output

```
$OUTPUT_PATH/
├── training_log.txt                       # per-step loss, timing, GPU memory
├── step-<N>.safetensors                   # every CHECKPOINTING_STEPS
├── epoch-<N>.safetensors                  # end of each epoch
├── images/                                # periodic validation samples
├── training_config/rank_<N>/              # resolved config, per rank
│   ├── data_config.yaml
│   ├── ds_config.json                     # generated DeepSpeed config
│   ├── accelerate_config.yaml
│   └── training_script.sh                 # copy of the worker that ran
└── .photomaster_training_complete.json    # written only on success
```

Per-step log lines look like:

```
[Rank 0][Epoch:1/2,GIter:100/2736,EIter:100/1368, 5.09s/iter,...],[Loss:0.0586,LR:0.000010],...
```

`GIter` is the global step, `EIter` the step within the epoch.

## Completion

Training is successful only if `.photomaster_training_complete.json` exists **and**
carries the current job's `success_token`; the sbatch script checks both and exits
non-zero otherwise. The token guards against a stale marker from an earlier run being
read as success.

Distributed teardown sometimes hangs after training finishes. Once the marker appears,
the job waits `POST_SUCCESS_GRACE_SECONDS` and then terminates `srun`, reporting
success — a hang after the marker is not a training failure.

## Notes

- Checkpoints are ~40.8 GB each. Size `CHECKPOINTING_STEPS` against available disk.
- `nvidia-smi` and the full resolved configuration are echoed at job start; read the
  header block in the job log first when diagnosing a failure.
- The worker symlinks several dataset directories into `/cache` on each node, matching
  the paths the trainer expects.

## Reproducibility

`trainer/train.py` seeds the global RNG at startup from `--seed` (default `SEED=0` in
the worker):

```python
set_seed(args.seed, device_specific=True)
```

This matters because the diffusion timestep and noise are drawn from the *global* torch
RNG inside `diffsynth` (`diffsynth/diffusion/loss.py`), not from a seeded generator. The
dataset shuffle and the resolution-bucket sampler are seeded off `--seed` separately and
were already deterministic.

`device_specific=True` offsets the seed by rank. This is not optional: seeding every rank
identically would make all ranks sample the same timestep, which changes what training
does rather than just making it repeatable.

Two runs of the same config at the same node count then agree closely, but **not
bit-exactly**. To check:

```bash
grep -oE 'Loss:[0-9.]+' run_1/training_log.txt | head -100 > a.txt
grep -oE 'Loss:[0-9.]+' run_2/training_log.txt | head -100 > b.txt
paste a.txt b.txt | awk '$1==$2 {m++} END {print m"/100 exact matches"}'
```

Measured on 3 nodes / 24 GPUs with `lingbot_dataAblation`, comparing the first 100 steps:

| | Exact matches | Mean abs diff | Max abs diff |
|---|---|---|---|
| Unseeded (before this change) | 0/100 | 1.5e-2 | 5.1e-2 |
| Seeded | 73/100 | 2.9e-5 | 2.0e-4 |

Every seeded mismatch is one or two units in the last logged decimal — the loss is printed
to 4 decimal places, so a 1e-4 disagreement is the smallest representable one. Treat "most
steps identical, the rest differing in the last digit" as a pass; the unseeded case is off
by two orders of magnitude more and is obvious by comparison.

Scope of the guarantee:

- Node count must match. World size changes steps per epoch and the per-rank seed offset.
- Not bit-exact, and checkpoints from two runs will not have identical hashes.
  `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG` are deliberately
  not set — they cost throughput. The residual 1e-4 noise is bf16 accumulation combined
  with NCCL all-reduce ordering, which makes gradient summation non-associative across
  ranks. Expect this floor early and gradual drift over long runs.
- TF32 needs no action: `torch.backends.cuda.matmul.allow_tf32` already defaults to
  `False` on torch 2.6.

## GPU memory

Memory per GPU depends on **how many GPUs you use**. DeepSpeed ZeRO stage 2 shards
optimizer state across data-parallel ranks, so shrinking the world size *increases*
per-GPU memory. A configuration that fits at 64 GPUs can OOM at 24.

Measured with `lingbot_dataAblation` on 80 GB H800s:

| GPUs | Preset | Result |
|---|---|---|
| 64 | `qwen_edit_2511_native` | fits, peak ~73.9 GB of 79.3 GB |
| 32 | `qwen_edit_2511_native` | marginal — OOM on 2 of 32 ranks |
| 24 | `qwen_edit_2511_native` | OOM, ~73.7 GB allocated and still short |
| 24 | `qwen_edit_1024` | fits comfortably |

So pick the preset to match your allocation: `qwen_edit_2511_native` at 8 nodes,
`qwen_edit_1024` when running smaller. If you must keep the native preset on a small
allocation, lower `BATCH_SIZE_PER_GPU` and raise `GRADIENT_ACCUMULATION_STEPS` to hold
the global batch size constant.

Also set (already in the shipped experiment config):

```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Without it a run can die from allocator *fragmentation* rather than a real shortfall —
in one case an 884 MB allocation failed while 1.77 GB sat reserved-but-unallocated.
With it, the same run wasted only ~103 MB. It does not change training numerics.

### Reading an OOM correctly

This failure does not present as an OOM. One rank OOMs inside the DeepSpeed gradient
all-reduce and stops participating; every other rank then blocks in `ALLREDUCE` until the
600 s NCCL watchdog fires and aborts the job with exit code 134. The log is dominated by
watchdog timeouts from the *healthy* ranks, which is misleading. Find the real culprit by
listing which ranks never reported a timeout:

```bash
grep -oE "Rank [0-9]+\] Watchdog caught" <job>.log | grep -oE "[0-9]+" | sort -nu
# any rank absent from that list is the one that actually failed
grep -B3 "torch.OutOfMemoryError" <job>.log
```

Which rank tips over depends on data ordering, so the same settings can pass at one node
count and fail at another.

### Check the GPUs are actually free first

Slurm reporting a node `idle` does not mean its GPUs are free. Orphaned `pt_data_worker`
processes from earlier aborted jobs can survive Slurm cleanup and hold tens of GB
indefinitely, which then OOMs your run for no visible reason. Before a large job:

```bash
srun --nodes=<N> --ntasks-per-node=1 --gres=gpu:8 bash -c \
  'echo "$(hostname) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr "\n" " ")"'
```

Anything above a few hundred MiB per GPU on an otherwise idle node is leftover state.
Do **not** `pkill` it: those processes may belong to another user's job, and on the
coordinator node the same command can take down `slurmctld` and the command proxy with
it. Exclude the affected nodes with `--exclude=<nodelist>`, or ask a cluster admin.


