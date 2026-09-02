#!/bin/bash
set -euo pipefail

export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=4

nvidia-smi

cur_dir=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")"; pwd)
# Repo root: this script lives in <repo>/scripts/dit_train
PHOTOMASTER_DIR=$(cd "$cur_dir/../.." && pwd)

# ---------------------------------------------------------------------------
# Conda environment
# ---------------------------------------------------------------------------
PHOTOMASTER_CONDA_ENV="${PHOTOMASTER_CONDA_ENV:-$PHOTOMASTER_DIR/../../env/photomaster_final}"

# Conda compiler activation hooks read unset vars; guard against nounset mode
set +u
conda activate "$PHOTOMASTER_CONDA_ENV"
set -u

echo "which conda"; which conda
echo "which python"; which python

# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------
BASE_MODEL_DIR="${QWEN_EDIT_BASE_MODEL_DIR:-$PHOTOMASTER_DIR/models/Qwen-Image-Edit-2511}"

if [ ! -f "$BASE_MODEL_DIR/transformer/diffusion_pytorch_model-00001-of-00005.safetensors" ]; then
  echo "Qwen-Image-Edit-2511 not found at $BASE_MODEL_DIR" >&2
  echo "Override with QWEN_EDIT_BASE_MODEL_DIR." >&2
  exit 1
fi
echo "Using Qwen-Image-Edit-2511 model dir: $BASE_MODEL_DIR"

REQUIRED_MODEL_FILES=(
  "$BASE_MODEL_DIR/transformer/diffusion_pytorch_model-00001-of-00005.safetensors"
  "$BASE_MODEL_DIR/transformer/diffusion_pytorch_model-00002-of-00005.safetensors"
  "$BASE_MODEL_DIR/transformer/diffusion_pytorch_model-00003-of-00005.safetensors"
  "$BASE_MODEL_DIR/transformer/diffusion_pytorch_model-00004-of-00005.safetensors"
  "$BASE_MODEL_DIR/transformer/diffusion_pytorch_model-00005-of-00005.safetensors"
  "$BASE_MODEL_DIR/text_encoder/model-00001-of-00004.safetensors"
  "$BASE_MODEL_DIR/text_encoder/model-00002-of-00004.safetensors"
  "$BASE_MODEL_DIR/text_encoder/model-00003-of-00004.safetensors"
  "$BASE_MODEL_DIR/text_encoder/model-00004-of-00004.safetensors"
  "$BASE_MODEL_DIR/vae/diffusion_pytorch_model.safetensors"
)
for model_file in "${REQUIRED_MODEL_FILES[@]}"; do
  if [ ! -f "$model_file" ]; then
    echo "Missing checkpoint: $model_file" >&2
    exit 1
  fi
done

# ---------------------------------------------------------------------------
# Training hyperparameters (all overridable via environment)
# ---------------------------------------------------------------------------
OUTPUT_DIR="${OUTPUT_PATH:-$PHOTOMASTER_DIR/outputs/qwen_edit_2511_ft}"

dataset_config="${DATA_CONFIG_PATH:-$PHOTOMASTER_DIR/config/data/lingbotForward_captionV3.yaml}"
if [ ! -f "$dataset_config" ]; then
  echo "Data config not found: $dataset_config" >&2
  echo "Set DATA_CONFIG_PATH to a YAML under $PHOTOMASTER_DIR/config/data" >&2
  exit 1
fi

LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WEIGHT_DECAY=1e-2
NUM_EPOCHS="${NUM_EPOCHS:-2}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-1000}"
VALIDATION_STEPS="${VALIDATION_STEPS:-$CHECKPOINTING_STEPS}"
SEED=0
MASTER_PORT="${MASTER_PORT:-21638}"

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-${GPUS_PER_NODE:-8}}"
NPROCESSES=$((NGPUS_PER_NODE * NNODES))

GLOBAL_BATCH_SIZE=$((BATCH_SIZE_PER_GPU * GRADIENT_ACCUMULATION_STEPS * NPROCESSES))
echo "Global batch size: $GLOBAL_BATCH_SIZE (= $BATCH_SIZE_PER_GPU * $GRADIENT_ACCUMULATION_STEPS * $NPROCESSES)"

# ---------------------------------------------------------------------------
# Config files
# ---------------------------------------------------------------------------
TRAINING_CONFIG_DIR="$OUTPUT_DIR/training_config"
mkdir -p "$OUTPUT_DIR" "$TRAINING_CONFIG_DIR"

cp "$dataset_config" "$TRAINING_CONFIG_DIR/data_config.yaml"
cp "${BASH_SOURCE[0]}" "$TRAINING_CONFIG_DIR/training_script.sh"

DS_CONFIG_PATH="$TRAINING_CONFIG_DIR/ds_config.json"
cat > "$DS_CONFIG_PATH" << EOL
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": $BATCH_SIZE_PER_GPU,
  "gradient_accumulation_steps": $GRADIENT_ACCUMULATION_STEPS,
  "gradient_clipping": "auto",
  "zero3_init_flag": false,
  "zero_optimization": {
    "stage": 2,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "offload_optimizer": { "device": "none", "nvme_path": null },
    "offload_param":     { "device": "none", "nvme_path": null },
    "stage3_gather_16bit_weights_on_model_save": false
  },
  "bf16": { "enabled": true },
  "fp16": { "enabled": false }
}
EOL

ACCELERATE_CONFIG_PATH="$TRAINING_CONFIG_DIR/accelerate_config.yaml"
cat > "$ACCELERATE_CONFIG_PATH" << EOL
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: DEEPSPEED
downcast_bf16: 'no'
deepspeed_config:
  deepspeed_config_file: $DS_CONFIG_PATH
  deepspeed_multinode_launcher: standard
enable_cpu_affinity: false
main_training_function: main
num_processes: $NPROCESSES
num_machines: $NNODES
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOL

echo "DeepSpeed config: $DS_CONFIG_PATH"
echo "Accelerate config: $ACCELERATE_CONFIG_PATH"

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
TRAINER_DIR="${TRAINER_DIR:-$cur_dir/trainer}"
if [ ! -f "$TRAINER_DIR/train.py" ]; then
  echo "Trainer not found at $TRAINER_DIR/train.py" >&2
  echo "Set TRAINER_DIR to a directory containing train.py." >&2
  exit 1
fi
cd "$TRAINER_DIR"

QWEN_TRAIN_RESOLUTION_PRESET="${QWEN_TRAIN_RESOLUTION_PRESET:-qwen_edit_1024}"
echo "Resolution preset: $QWEN_TRAIN_RESOLUTION_PRESET"

accelerate launch \
  --config_file="$ACCELERATE_CONFIG_PATH" \
  --deepspeed_config_file="$DS_CONFIG_PATH" \
  --mixed_precision="bf16" \
  --num_machines="$NNODES" \
  --machine_rank="$NODE_RANK" \
  --num_processes="$NPROCESSES" \
  --main_process_ip="$MASTER_ADDR" \
  --main_process_port="$MASTER_PORT" \
  train.py \
  --config_path "$dataset_config" \
  --max_pixels 1048576 \
  --model_paths "[ \
    [ \
        \"$BASE_MODEL_DIR/transformer/diffusion_pytorch_model-00001-of-00005.safetensors\", \
        \"$BASE_MODEL_DIR/transformer/diffusion_pytorch_model-00002-of-00005.safetensors\", \
        \"$BASE_MODEL_DIR/transformer/diffusion_pytorch_model-00003-of-00005.safetensors\", \
        \"$BASE_MODEL_DIR/transformer/diffusion_pytorch_model-00004-of-00005.safetensors\", \
        \"$BASE_MODEL_DIR/transformer/diffusion_pytorch_model-00005-of-00005.safetensors\" \
    ], \
    [ \
        \"$BASE_MODEL_DIR/text_encoder/model-00001-of-00004.safetensors\", \
        \"$BASE_MODEL_DIR/text_encoder/model-00002-of-00004.safetensors\", \
        \"$BASE_MODEL_DIR/text_encoder/model-00003-of-00004.safetensors\", \
        \"$BASE_MODEL_DIR/text_encoder/model-00004-of-00004.safetensors\" \
    ], \
    \"$BASE_MODEL_DIR/vae/diffusion_pytorch_model.safetensors\" \
]" \
  --tokenizer_path "$BASE_MODEL_DIR/tokenizer" \
  --processor_path "$BASE_MODEL_DIR/processor" \
  --learning_rate "$LEARNING_RATE" \
  --batch_size "$BATCH_SIZE_PER_GPU" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --num_epochs "$NUM_EPOCHS" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$OUTPUT_DIR" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters \
  --save_steps="$CHECKPOINTING_STEPS" \
  --vis_steps=2000 \
  --seed="$SEED" \
  --global_batch_size="$GLOBAL_BATCH_SIZE" \
  --task_buckets=t2i,edit \
  --resolution_preset "$QWEN_TRAIN_RESOLUTION_PRESET" \
  --zero_cond_t
