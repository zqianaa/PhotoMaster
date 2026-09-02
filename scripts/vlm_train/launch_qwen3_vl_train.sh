#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root two levels up from this script's directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHOTOMASTER_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

nvidia-smi

# --- Conda environment -----------------------------------------------------------
PHOTOMASTER_CONDA_ENV="${PHOTOMASTER_CONDA_ENV:-$PHOTOMASTER_DIR/../../env/photomaster_final}"
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PHOTOMASTER_CONDA_ENV"
echo "which python: $(which python)"

# --- Paths -----------------------------------------------------------------------
SCRIPT_PATH="${SCRIPT_PATH:-$PHOTOMASTER_DIR/scripts/vlm_train/LLaMA-Factory/src/train.py}"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: LLaMA-Factory train.py not found at: $SCRIPT_PATH" >&2
    echo "  Set SCRIPT_PATH to the correct location." >&2
    exit 1
fi

MODEL_PATH="${MODEL_PATH:-$PHOTOMASTER_DIR/models/Qwen3-VL-32B-Instruct}"
OUTPUT_PATH="${OUTPUT_PATH:?ERROR: OUTPUT_PATH must be set (via env file or export)}"
dataset_dir="${DATASET_DIR:-$PHOTOMASTER_DIR/data}"
cache_dir="${CACHE_DIR:-$PHOTOMASTER_DIR/cache}"

# Resolve OUTPUT_PATH relative to repo root when it is not absolute.
if [[ "$OUTPUT_PATH" != /* ]]; then
    OUTPUT_PATH="$PHOTOMASTER_DIR/$OUTPUT_PATH"
fi

# --- Hyperparameters (sourced from experiment env file) --------------------------
dataset_name="${DATASET_NAME:-lingbot_captionV3,forwardPipeline_captionV3}"
cutoff_len="${CUTOFF_LEN:-2048}"
image_max_pixels="${IMAGE_MAX_PIXELS:-262144}"
learning_rate="${LEARNING_RATE:-1e-5}"
num_train_epochs="${NUM_TRAIN_EPOCHS:-3}"
save_steps="${SAVE_STEPS:-1000}"
logging_steps="${LOGGING_STEPS:-1}"
warmup_ratio="${WARMUP_RATIO:-0.02}"
weight_decay="${WEIGHT_DECAY:-0.01}"
per_device_train_batch_size="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-1}"
ddp_timeout="${DDP_TIMEOUT:-180000}"
lr_scheduler_type="${LR_SCHEDULER_TYPE:-cosine}"

# --- Rendezvous (single-node defaults) -------------------------------------------
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-39987}"

# --- DeepSpeed ZeRO-2 config (generated inline) ----------------------------------
mkdir -p "$OUTPUT_PATH"
DS_CONFIG_PATH="$OUTPUT_PATH/ds_z2_config.json"
cat > "$DS_CONFIG_PATH" <<'EOF'
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "zero_allow_untested_optimizer": true,
  "fp16": {
    "enabled": "auto",
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 16,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "bf16": {
    "enabled": "auto"
  },
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": false,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true,
    "round_robin_gradients": true
  }
}
EOF

# --- NCCL / HCCL tuning ----------------------------------------------------------
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=15
export NCCL_IB_TC=128
export NCCL_P2P_LEVEL=NVL
export NCCL_DEBUG=INFO
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA='^=mlx5_bond_0'
export NCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0
export GLOO_TIMEOUT_SECONDS=1800000
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=0
export HCCL_EXEC_TIMEOUT=10800
export HCCL_DETERMINISTIC=true

# --- Summary ---------------------------------------------------------------------
echo "model path:               ${MODEL_PATH}"
echo "script path:              ${SCRIPT_PATH}"
echo "dataset name:             ${dataset_name}"
echo "dataset dir:              ${dataset_dir}"
echo "cutoff len:               ${cutoff_len}"
echo "image max pixels:         ${image_max_pixels}"
echo "output path:              ${OUTPUT_PATH}"
echo "ds config:                ${DS_CONFIG_PATH}"
echo "learning rate:            ${learning_rate}"
echo "save steps:               ${save_steps}"
echo "epochs:                   ${num_train_epochs}"
echo "per device batch size:    ${per_device_train_batch_size}"
echo "gradient accum steps:     ${gradient_accumulation_steps}"
echo "nodes/gpus/rank/master:   ${NNODES}/${NGPUS_PER_NODE}/${NODE_RANK}/${MASTER_ADDR}:${MASTER_PORT}"

# --- Launch ----------------------------------------------------------------------
torchrun \
    --nproc_per_node="$NGPUS_PER_NODE" \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    "$SCRIPT_PATH" \
    --deepspeed "$DS_CONFIG_PATH" \
    --stage sft \
    --do_train \
    --model_name_or_path "$MODEL_PATH" \
    --cache_dir "$cache_dir" \
    --dataset "$dataset_name" \
    --dataset_dir "$dataset_dir" \
    --image_max_pixels "$image_max_pixels" \
    --template qwen3_vl_nothink \
    --finetuning_type lora \
    --lora_rank 64 \
    --lora_target all \
    --output_dir "$OUTPUT_PATH" \
    --overwrite_output_dir \
    --overwrite_cache \
    --warmup_ratio "$warmup_ratio" \
    --weight_decay "$weight_decay" \
    --per_device_train_batch_size "$per_device_train_batch_size" \
    --gradient_accumulation_steps "$gradient_accumulation_steps" \
    --ddp_timeout "$ddp_timeout" \
    --learning_rate "$learning_rate" \
    --lr_scheduler_type "$lr_scheduler_type" \
    --logging_steps "$logging_steps" \
    --cutoff_len "$cutoff_len" \
    --save_steps "$save_steps" \
    --num_train_epochs "$num_train_epochs" \
    --freeze_vision_tower True \
    --freeze_multi_modal_projector True \
    --packing True \
    --plot_loss True \
    --report_to tensorboard \
    --bf16 \
    --preprocessing_num_workers 128 \
    --preprocessing_batch_size 100
