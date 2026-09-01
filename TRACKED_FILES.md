# PhotoMaster — Files to Track

Scope: everything needed to reproduce the two training runs (I2I and VLM) and the
benchmark inference pipeline. Migration into this repository is happening one piece at a
time; this file tracks what has landed and what is still pending.

| Component | Status |
|---|---|
| I2I — Qwen-Image-Edit-2511 | **Migrated** — see `scripts/dit_train/` |
| VLM — Qwen3-VL LoRA SFT | Pending — source paths listed in §2 |
| Inference — VLM to DiT pipeline | **Migrated** — see `scripts/inference/` |

Source paths below are relative to the PhotoMaster project root
(`/home/ma-user/work/dataset/wilson/BOOGU/photomaster`).

---

## 1. I2I Training — Qwen-Image-Edit-2511 (migrated)

Entry point: `scripts/dit_train/train_qwen_edit_2511.slurm`. Usage, prerequisites, and the
full environment-variable reference are in `scripts/dit_train/README.md`.

| Tracked path | Source | Purpose |
|---|---|---|
| `scripts/dit_train/train_qwen_edit_2511.slurm` | new | sbatch entry point. Replaces a wrapper that was previously generated at launch time, so training is launchable from a clone. |
| `scripts/dit_train/launch_qwen_edit_2511_train.sh` | `i2i_training/launch_qwen_edit_2511_train.sh` | Per-rank worker. Conda setup, base-model resolution, DeepSpeed + Accelerate config generation, trainer invocation. |
| `config/data/*.yaml` (5 files) | `i2i_training/data_config/` | Dataset definitions, copied verbatim. |
| `config/experiments/lingbot_ablation.env` | new | Hyperparameters for the lingbot ablation run. |
| `scripts/dit_train/trainer/**` (11 files) | `DialogEdit_qwen/{train,utils,pipelines,data}` | Vendored training code, so a clone trains without an external checkout. |

### Vendored trainer

Copied from `DialogEdit_qwen` — the minimal import closure of `train/train.py`, nothing
more. Every file is byte-identical to source except two.

`train.py` has three changes:

1. `set_seed` added to the `accelerate.utils` import.
2. `sys.path.append` now appends the script's own directory rather than its grandparent,
   because `utils/`, `pipelines/` and `data/` are siblings here; upstream `train.py` sat
   one level deeper.
3. Global RNG seeding after `Accelerator()` — see the Reproducibility section of
   `scripts/dit_train/README.md`.

`data/resolution_bucketing.py` is upstream `data/data.py` reduced to the five classes that
`tar_parquet_data_v2.py` imports (`ResolutionBucket`, `ResolutionBucketer`,
`TaskResolutionBucketer`, `ResolutionBucketSampler`, `ResolutionBucketSampler_distributed`).
Those five are byte-identical to upstream; the file was renamed because nothing named
"data" survives in it. Dropped from it as unreachable: a second `UnifiedDataset` and
`create_unified_dataset` keyed on a different config schema (`type`/`path`/`*_column`,
where `config/data/*.yaml` uses `dataset_name`/`image_dir`/`annotation_dir`),
`get_dataset_configs` (hardcoded `/scratch/dyvm6xra/...` paths from another cluster),
`open_image`, `process_task_mapping_batch`, `generate_resolutions`, and the `__main__`
block. Removing them also removed the only importer of `data/dataloading.py`, so that
module is not vendored either — none of its nine loaders was reachable from any config.

Excluded as unreachable from `train.py`: the `train_vlm`/`train_vllm` entry points, nine
`pipelines/` variants (`*_bak`, `qwen_image_vlm`, `qwen_image_vllm*`, edit_plus/connector),
`utils/utils_bak.py`, `utils/model_loggger.py` (a typo-named duplicate), and seven unused
`data/` modules including `dataloading.py`.

`diffsynth` (2.0.1) is **not** vendored: it is a pip package in the conda environment.
Note the diffusion loss and the noise and timestep sampling live there, not in this tree.

Because the trainer is vendored, this repository is now the source of truth for trainer
behaviour and will not pick up upstream `DialogEdit_qwen` changes automatically.

The worker differs from its source in exactly five places, all path-related:

1. Repo root resolves two levels up (`scripts/dit_train/`) instead of one (`i2i_training/`).
2. `QWEN_EDIT_BASE_MODEL_DIR` added as the first base-model candidate — the base model
   lives outside this repository.
3. Default data config points at `config/data/`, and a missing file now fails with a clear error.
4. The trainer location is overridable via `DIALOGEDIT_DIR` and is existence-checked.
5. Default output path is `<repo>/outputs/` so a fresh clone writes inside itself.

Not migrated: `bash_scripts/train_QwenImageEdit2511.sh`, a superseded variant of the same
training. Keeping it would mean a second, untested entry point for one job.

---

## 2. VLM Training — Qwen3-VL LoRA SFT

Entry point: `bash_scripts/train_qwen3_vl_8b_photomaster_wilson.sh`

| Source path | Purpose |
|---|---|
| `bash_scripts/train_qwen3_vl_8b_photomaster_wilson.sh` | Launch script. Activates conda, sets NCCL/HCCL tuning, builds the LLaMA-Factory argument list, launches via `torchrun`. |
| `training_data/dataset_info.json` | LLaMA-Factory dataset registry. Maps dataset names (`lingbot_captionV3`, `forwardPipeline_captionV3`, …) to label JSON paths and ShareGPT column mappings. |

### Direct launch

```bash
export MODEL_PATH=/path/to/Qwen3-VL-32B-Instruct
export OUTPUT_PATH=/path/to/output
export DATASET_NAME=lingbot_captionV3,forwardPipeline_captionV3
export DATASET_DIR=/path/to/training_data
export CUTOFF_LEN=2048
export IMAGE_MAX_PIXELS=262144
export LEARNING_RATE=1e-5
export NUM_TRAIN_EPOCHS=3
export SAVE_STEPS=1000
export PER_DEVICE_TRAIN_BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=1
export MA_NUM_GPUS=8

bash bash_scripts/train_qwen3_vl_8b_photomaster_wilson.sh
```

As with I2I, all variables default, so the script runs bare. Training is LoRA
(`--finetuning_type lora --lora_rank 64 --lora_target all`) with the vision tower and
multimodal projector frozen. Multi-node follows the same `PHOTOMASTER_DIRECT_SLURM=1`
convention.

Dataset ratios: repeating a name in `DATASET_NAME` oversamples it. Expand a ratio map into
repeated names by hand when launching, e.g.
`DATASET_NAME=lingbot_captionV3,lingbot_captionV3,forwardPipeline_captionV3` for a 2:1 mix.

---

## 3. Inference — VLM to DiT benchmark pipeline (migrated)

Entry point: `scripts/inference/run_pipeline.sh`. Usage, prerequisites, and the full
environment-variable reference are in `scripts/inference/README.md`.

Three stages, each writing into one shared result directory
(`bench/<BENCH>/results/<RUN_NAME>/`): VLM critique to `vlm_output.json`, conversion to
`dit_input.jsonl`, then image editing into `<SAVE_MODEL_NAME>/`.

| Tracked path | Source | Purpose |
|---|---|---|
| `scripts/inference/run_pipeline.sh` | new | Chains all three stages for one `(MODEL, BENCH)` pair. |
| `scripts/inference/run_vlm_infer.sh` | `bash_scripts/run_infer.sh` | Stage 1 launcher. |
| `scripts/inference/run_vlm_to_dit.sh` | `bash_scripts/run_vlm_to_i2i.sh` (step 1 only) | Stage 2 launcher. |
| `scripts/inference/run_qwen_edit_infer.sh` | `bash_scripts/run_qwen_edit.sh` | Stage 3 launcher. |
| `scripts/inference/vlm_infer.py` | `inference/vlm_infer.py` | Stage 1 worker. |
| `scripts/inference/vlm_to_dit_bench_reformat.py` | `inference/vlm_to_dit_bench_reformat.py` | Stage 2 worker, byte-identical to source. |
| `scripts/inference/qwen_edit_infer.py` | `inference/qwen_edit_infer.py` | Stage 3 worker. |
| `scripts/inference/prompts/system_prompt_v{1,2,3}.txt` | `inference/system_prompt_v*.txt` | Analysis framework, copied verbatim. v3 is the default; v1 and v2 are kept because recorded runs reference them. |
| `scripts/inference/vendor/qwenimage/**` | `vendor/qwenimage/` | Vendored diffusers pipeline patch, copied verbatim. |
| `config/experiments/photomaster_bench_infer.env` | new | Settings for the recorded reference run. |
| `bench/photomaster_bench/metadata.json` | `bench/photomaster_bench/metadata.json` | 270-sample benchmark definition. |
| `bench/photomaster_bench/results/photomaster_reference/**` | `bench/photomaster_bench/results/qwen3vl32b_lora_lingbotForwardAblation/` | Recorded `vlm_output.json` and `dit_input.jsonl`, renamed to read as an example rather than a specific ablation. |

### Deviations from source

The two workers differ from their sources in four places:

1. `qwen_edit_infer.py` resolves the vendored pipeline next to itself rather than at an
   out-of-tree `vendor/` directory, since it is vendored into this tree. The
   `PHOTOMASTER_QWEN_EDIT_PIPELINE` override is unchanged.
2. `vlm_infer.py` defaults `--system-prompt-file` to `prompts/system_prompt_v3.txt`
   resolved from `__file__`. The source default was the bare relative name
   `system_prompt_v3.txt`, which only resolved when the process ran from `inference/`.
3. `vlm_infer.py` drops `--use-ref-img` and the reference-image branch of
   `build_messages`. No benchmark here supplies reference images, and the branch indexed
   `sample["ref_img_path"]` without a null check, so enabling it raised `TypeError` on
   every `photomaster_bench` sample. The `use_ref_img` key is correspondingly gone from
   the output JSON header.
4. All three launchers resolve the repository root two levels up (`scripts/inference/`)
   instead of one (`bash_scripts/`).

The launchers also gained behaviour their sources lacked:

- `run_vlm_infer.sh` takes `GPU_MODE` (`single` / `multi_gpu` / `multi_node`), matching the
  convention `run_qwen_edit.sh` already used. The source ran single-process only, while the
  recorded 32-rank reference run was launched by a generated script that existed outside
  version control. It also exports `PHOTOMASTER_GATHER_DIR` into the result
  directory: stage 1 gathers rank shards through the filesystem, and the upstream default
  of `/tmp/infer_gather` is node-local, so multi-node runs silently lost every non-master
  rank's results.
- `run_vlm_to_dit.sh` is the conversion step of `run_vlm_to_i2i.sh`, extracted so the three
  stages can be re-entered independently. Its sibling step, `run_banana_i2i.py`, is a
  Yunwu-API editing path unrelated to the DiT and is not migrated.

### Layout constraint

`qwen_edit_infer.py` resolves relative image paths in `dit_input.jsonl` against the JSONL's
grandparent directory. The result directory must therefore stay exactly two levels below
the bench root. `bench/` mirrors its original relative path for the same reason
`config/data/` does — it keeps recorded paths valid without rewriting them.

Existing result directories on the cluster contain `img` and `ref_img` symlinks back to the
bench root. These predate the path fallback in `load_images` and are not reproduced here.

### Not tracked

Benchmark images (`bench/photomaster_bench/img/`, 141 MB across 270 files) are pending a
decision on git-lfs; `metadata.json` lists every expected filename. Also excluded: the
other 18 result directories under `bench/photomaster_bench/results/`, all generated DiT
image directories, and the packed result archives (`temp0.7_results_dirs.tar` at 2.8 GB,
`ablation_result.tar` and `photomaster_bench_result.tar` at 1.3 GB and 1.1 GB).

Not migrated from `inference/`: `run_banana_i2i.py` and `hf_gradio_qwen_edit_infer.py`
(a Gradio demo, not part of the benchmark pipeline).

---

## 4. Data Preparation

These produce the label JSONs and packed datasets that the two trainings consume. Not
invoked by training itself, but required to rebuild the data from raw sources.

| Source path | Purpose |
|---|---|
| `training_data/scripts/convert_raw_to_dialogedit.py` | Raw capture to DialogEdit format. |
| `training_data/scripts/launch_convert_raw_to_dialogedit.sh` | Batch driver for the above. |
| `training_data/scripts/convert_boogu_to_dialogedit.py` | BOOGU format to DialogEdit format. |
| `training_data/scripts/create_clean_label.py` | Builds `clean_labels*.json` consumed via `dataset_info.json`. |
| `training_data/scripts/verify_clean_dialogedit.py` | Validates converted output. |
| `training_data/scripts/polish_instruction_en_yunwu.py` | Instruction rewriting via the Yunwu API. |
| `training_data/scripts/launch_instruction_polish.sh` | Batch driver for instruction polishing. |
| `training_data/scripts/merge_recaptioned.py` | Merges recaptioned shards. |
| `training_data/scripts/apply_recaptioned_json_versions.py` | Applies a recaption pass to a label version. |
| `training_data/scripts/create_tar.py` | Packs images into tars. |
| `training_data/scripts/extract_tars.sh` | Unpacks them. |
| `training_data/scripts/check_tars.py` | Integrity check on packed tars. |
| `training_data/scripts/pack_raw_data.sh` | Raw-data packing driver. |
| `training_data/scripts/build_forward_ablation_subset.py` | Builds the forward ablation subsets. |
| `training_data/scripts/check_forward_ablation_counts.py` | Verifies ablation subset sizes. |
| `training_data/scripts/count_vlm_token_lengths.py` | Token-length statistics for cutoff tuning. |
| `training_data/scripts/hard_code_dimension.py` | Dimension normalization pass. |
| `training_data/scripts/process_json.py` | Shared label JSON helper. |
| `training_data/scripts/question.json` | Prompt/question templates used by conversion. |
| `training_data/scripts/SAMPLE_FORMAT.md` | Documents the expected sample schema. |
| `training_data/polish_instruction_en_yunwu.py` | Top-level copy of the polish script (kept — the scripts/ copy and this one have diverged). |
| `process_json.py` | Root-level label JSON utility. |
| `get_data_subset.py` | Subset sampling utility. |

---

## 5. Excluded

| Path | Reason |
|---|---|
| `training_data/lingbot/**`, `training_data/forwardPipeline/**`, `training_data/banana_mirror/**` | Label JSONs are 76 MB–564 MB each; GitHub rejects blobs over 100 MB. |
| `training_data/to_process_data/` | 64,073 entries of raw staging data. |
| `training_data/**/dit_qwen/zip`, `training_data/**/dit_qwen/parquet`, `training_data/**/*.tar` | Packed image and annotation binaries consumed by I2I training. |
| `training_data/zian_data/`, `training_data/zian_data_v1/`, `training_data/dummy_data/`, `training_data/converter_benchmark_dummy/` | Sample and scratch datasets. |
| `models/`, `ft_ckpts/`, `cache/` | Base weights, fine-tuned checkpoints, and preprocessing cache. |
| `__pycache__/`, `.ruff_cache/`, `.ipynb_checkpoints/`, `.cache/` | Build and tooling artifacts. |

`training_data/` totals roughly 14 TB. Only `dataset_info.json` and `scripts/` are tracked
from that tree; everything else is data.

---

## 6. External Dependencies

Neither trainer lives in this repository. Both launch scripts call into code installed
elsewhere on the cluster; record versions separately. Inference, by contrast, is
self-contained apart from model weights — its pipeline patch is vendored.

| Dependency | Location | Used by |
|---|---|---|
| LLaMA-Factory | `/opt/huawei/schedule-train/algorithm/photomaster/LLaMA-Factory` — `src/train.py` and `examples/deepspeed/ds_z2_config.json` | VLM training |
| DialogEdit_qwen | `/opt/huawei/dataset/dialog_edit_env/DialogEdit_qwen` — `train/train.py` | I2I training |
| Conda environment `photomaster_final` | `/opt/huawei/dataset/wilson/env/photomaster_final` | Both trainings, and inference |
| Environment bootstrap | `/opt/huawei/dataset/xiaoyi_video_env/setup.sh` | VLM training |
| Qwen-Image-Edit-2511 base model | `models/base_models/Qwen-Image-Edit-2511`, falling back to `/opt/huawei/dataset/dialog_edit_env/ckpts/Qwen/Qwen-Image-Edit-2511` | I2I training, inference stage 3 |
| Qwen3-VL-32B-Instruct base model | `/opt/huawei/dataset/shared_models/Qwen3-VL-32B-Instruct` | VLM training, inference stage 1 |
| Fine-tuned VLM checkpoints | `ft_ckpts/` | Inference stage 1 |

Override the trainer locations with `ALGO_PATH` / `SCRIPT_PATH` (VLM) and the conda
environment with `PHOTOMASTER_CONDA_ENV` (both). The I2I script hardcodes its `cd` into
`DialogEdit_qwen`; change that line if the trainer lives elsewhere. Inference model paths
are set per run via `MODEL_ROOT` / `MODEL_PATH`; see `scripts/inference/README.md`.

---

## 7. Repository Layout

```
opensource/
├── TRACKED_FILES.md
├── scripts/
│   ├── dit_train/          # I2I training — slurm entry, worker, vendored trainer
│   └── inference/          # benchmark pipeline — 3 stages, prompts, vendored pipeline
├── config/
│   ├── data/*.yaml         # dataset definitions
│   └── experiments/*.env   # per-run settings, training and inference
└── bench/
    └── photomaster_bench/  # metadata, images, recorded reference result
```

Pending migration (§2 and §4) adds `training_data/` for `dataset_info.json` and the data
preparation scripts, plus the VLM training launcher.

Mirroring the original relative paths keeps recorded paths valid without rewriting the
contents of the files that carry them: the `config/data/` YAMLs and `dataset_info.json`
reference absolute cluster paths under
`/opt/huawei/dataset/wilson/BOOGU/photomaster/training_data/` and must be repointed for any
other environment, while `bench/` must keep its depth so `dit_input.jsonl`'s relative image
paths resolve (see §3).

