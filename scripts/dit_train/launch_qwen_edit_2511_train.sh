#!/bin/bash

# ======= BASIC ENV SETUP =======
nvidia-smi
[ -z "$BASH_VERSION" ] && exec bash "$0" "$@"
echo "Running in: ${BASH_VERSION:-unknown shell}"
echo "Shell: $0"
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=4

# export NCCL_IB_GID_INDEX=3
# export NCCL_IB_HCA='^=mlx5_bond_0'
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_TC=128
# export NCCL_IB_TIMEOUT=22
# export NCCL_IB_RETRY_CNT=15
# export NCCL_IB_QPS_PER_CONNECTION=32

# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

cur_dir=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")"; pwd)
# Repo root: this script lives in <repo>/scripts/dit_train
PHOTOMASTER_DIR=$(cd "$cur_dir/../.." && pwd)

if [ -d "/home/ma-user/work" ]; then
    DEFAULT_DIR="/home/ma-user/work"
else
    DEFAULT_DIR="/opt/huawei"
fi

OPT_DIR=$DEFAULT_DIR
# setup conda env
# bash $OPT_DIR/dataset/xiaoyi_video_env/setup.sh
# . $OPT_DIR/dataset/xiaoyi_video_env/setup.sh

if [ -d "/cache/ckpts" ]; then
    echo "skip ln ckpts"
else
    # ln -s $DEFAULT_DIR/dataset/dialogue_edit/ckpts /cache/
    ln -s $DEFAULT_DIR/dataset/dialog_edit_env/ckpts /cache/
fi
if [ -d "/cache/editing" ]; then
    echo "skip ln editing"
else
    ln -s $DEFAULT_DIR/dataset/dialogue_edit/editing /cache/
fi

# if [ -d "/cache/qwen_outputs" ]; then
#     echo "skip ln qwen_outputs"
# else
#     mkdir -p $DEFAULT_DIR/dataset/wulan2_ckpt_sfs/qwen_outputs
#     ln -s $DEFAULT_DIR/dataset/wulan2_ckpt_sfs/qwen_outputs /cache/
# fi

if [ -d "/cache/outputs" ]; then
    echo "skip ln outputs"
else
    mkdir -p $DEFAULT_DIR/dataset/Editing_Understanding/qwen_outputs
    ln -s $DEFAULT_DIR/dataset/Editing_Understanding/qwen_outputs /cache/outputs
fi

if [ -d "/cache/AIGC_Editing" ]; then
    echo "skip ln AIGC_Editing"
else
    ln -s $DEFAULT_DIR/dataset/Editing_Understanding/AIGC_Editing /cache/
fi


if [ -d "/cache/Editing_Understanding" ]; then
    echo "skip ln Editing_Understanding"
else
    ln -s $DEFAULT_DIR/dataset/Editing_Understanding /cache/
fi

if [ -d "/cache/dataset" ]; then
    echo "skip ln dataset"
else
    ln -s $DEFAULT_DIR/dataset /cache/
fi

# echo $cur_dir
# PROJECT_DIR=$cur_dir/../../
# cd $PROJECT_DIR
#

sourceConda(){
   # >>> conda initialize >>>
   # !! Contents within this block are managed by 'conda init' !!
   __conda_setup="$("$OPT_DIR/dataset/dialog_edit_env/miniconda3/bin/conda" 'shell.bash' 'hook' 2> /dev/null)"
   if [ $? -eq 0 ]; then
       eval "$__conda_setup"
   else
       if [ -f "$OPT_DIR/dataset/dialog_edit_env/miniconda3/etc/profile.d/conda.sh" ]; then
           . "$OPT_DIR/dataset/dialog_edit_env/miniconda3/etc/profile.d/conda.sh"
       else
           export PATH="$OPT_DIR/dataset/dialog_edit_env/miniconda3/bin:$PATH"
       fi
   fi
   unset __conda_setup
   # <<< conda initialize <<<
}

echo "start install miniconda3"
export OPT_DIR=$DEFAULT_DIR
export CONDA_PATH=$OPT_DIR/dataset/dialog_edit_env/miniconda3
export PATH="$OPT_DIR/dataset/dialog_edit_env/miniconda3/bin:$PATH"

# bash $OPT_DIR/dataset/dialog_edit_env/Miniconda3-latest-Linux-x86_64.sh -p $CONDA_PATH -b -u
# ls $CONDA_PATH/bin/ -l
$CONDA_PATH/bin/conda init bash
# conda init
# source ~/.bashrc
sourceConda
echo "done install miniconda3"

# conda activate $DEFAULT_DIR/dataset/wilson/env/qwen-image
# legacy: conda activate $DEFAULT_DIR/dataset/wilson/env/qwen-image-fa3
PHOTOMASTER_CONDA_ENV="${PHOTOMASTER_CONDA_ENV:-$DEFAULT_DIR/dataset/wilson/env/photomaster_final}"
conda activate "$PHOTOMASTER_CONDA_ENV"

# python -c "import flash_attn_interface;print(flash_attn_interface.flash_attn_func());"

# source $DEFAULT_DIR/dataset/wilson/env/qwen-image/bin/activate

echo "which conda"
which conda
echo "which python"
which python
pip list
# echo "###########################show conda list#####################################"
# conda list
echo "#####################Current path of work directory############################"
pwd

# TODO
OUTPUT_DIR="${OUTPUT_PATH:-$PHOTOMASTER_DIR/outputs/qwen_edit_2511_ft}"
MODEL_DIR_CANDIDATES=(
  "${QWEN_EDIT_BASE_MODEL_DIR:-}"
  "$PHOTOMASTER_DIR/models/base_models/Qwen-Image-Edit-2511"
  "/home/ma-user/work/dataset/wilson/BOOGU/photomaster/models/base_models/Qwen-Image-Edit-2511"
  "$DEFAULT_DIR/dataset/wilson/BOOGU/photomaster/models/base_models/Qwen-Image-Edit-2511"
  "/cache/ckpts/Qwen-Image-Edit-2511"
)
BASE_MODEL_DIR="$DEFAULT_DIR/dataset/dialog_edit_env/ckpts/Qwen/Qwen-Image-Edit-2511/"
for candidate_model_dir in "${MODEL_DIR_CANDIDATES[@]}"; do
  if [ -f "$candidate_model_dir/transformer/diffusion_pytorch_model-00001-of-00005.safetensors" ]; then
    BASE_MODEL_DIR="$candidate_model_dir"
    break
  fi
done

if [ -z "$BASE_MODEL_DIR" ]; then
  echo "Unable to locate Qwen-Image-Edit-2511 checkpoints. Checked:" >&2
  printf '  %s\n' "${MODEL_DIR_CANDIDATES[@]}" >&2
  exit 1
fi
echo "Using Qwen-Image-Edit-2511 model dir: $BASE_MODEL_DIR"

# OUTPUT_DIR="$DEFAULT_DIR/dataset/dialog_edit_env/qwen_outputs/debug"

if [ "${PHOTOMASTER_DIRECT_SLURM:-0}" = "1" ] && [ -n "${SLURM_PROCID:-}" ]; then
    TRAINING_CONFIG_DIR="$OUTPUT_DIR/training_config/rank_${SLURM_PROCID}"
else
    TRAINING_CONFIG_DIR="$OUTPUT_DIR/training_config"
fi
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TRAINING_CONFIG_DIR"
# dataset_config="${DATA_CONFIG_PATH:-$DEFAULT_DIR/dataset/dialog_edit_env/DialogEdit_qwen/configs/data/data_config_v260117.yaml}"
dataset_config="${DATA_CONFIG_PATH:-$PHOTOMASTER_DIR/config/data/lingbotForward_captionV3.yaml}"

if [ ! -f "$dataset_config" ]; then
  echo "Data config not found: $dataset_config" >&2
  echo "Set DATA_CONFIG_PATH to a YAML under $PHOTOMASTER_DIR/config/data" >&2
  exit 1
fi

# dataset_config='configs/data/edit_apple_60w_t2i_120w.yaml'

cat "$dataset_config" > "$TRAINING_CONFIG_DIR/data_config.yaml"

# SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_PATH="$cur_dir/$(basename "${BASH_SOURCE[0]}")"
cp "$SCRIPT_PATH" "$TRAINING_CONFIG_DIR/training_script.sh"

MASTER_HOST="${VC_WORKER_HOSTS:-}"
MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
NNODES="${MA_NUM_HOSTS:-}"
NODE_RANK="${VC_TASK_INDEX:-}"

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"

NGPUS_PER_NODE="${NGPUS_PER_NODE:-${GPUS_PER_NODE:-8}}"
NPROCESSES=$((NGPUS_PER_NODE * NNODES))
echo "$MASTER_ADDR, $NNODES, $NODE_RANK, $NPROCESSES"
# /home/ma-user/work/dataset/dialog_edit_env/part1
MASTER_PORT="${MASTER_PORT:-21638}"

BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
GLOBAL_BATCH_SIZE=$((BATCH_SIZE_PER_GPU * GRADIENT_ACCUMULATION_STEPS * NPROCESSES))
echo -e "\nUsing global batch size: $GLOBAL_BATCH_SIZE (= $BATCH_SIZE_PER_GPU * $GRADIENT_ACCUMULATION_STEPS * $NPROCESSES)\n"


# Other training settings
LEARNING_RATE="${LEARNING_RATE:-1e-5}"

WEIGHT_DECAY=1e-2
NUM_EPOCHS="${NUM_EPOCHS:-2}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-1000}"
VALIDATION_STEPS="${VALIDATION_STEPS:-$CHECKPOINTING_STEPS}"
SEED=0

# Path to validation JSON file
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
    echo "Missing Qwen-Image-Edit-2511 checkpoint: $model_file" >&2
    exit 1
  fi
done

# ======= SETUP CONFIG FILES DYNAMICALLY =======
# Create DeepSpeed config file with current parameters
DS_CONFIG_PATH="$TRAINING_CONFIG_DIR/ds_config.json"

# "allgather_bucket_size": 2e8,
# "reduce_bucket_size": 2e8,

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
    "offload_optimizer": {
      "device": "none",
      "nvme_path": null
    },
    "offload_param": {
      "device": "none",
      "nvme_path": null
    },
    "stage3_gather_16bit_weights_on_model_save": false
  },
  "bf16": {
    "enabled": true
  },
  "fp16": {
    "enabled": false
  }
}
EOL
echo "Generated DeepSpeed config at $DS_CONFIG_PATH"

# Create or update Accelerate config file
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
echo "Generated Accelerate config at $ACCELERATE_CONFIG_PATH"

# The trainer is vendored in this repository (scripts/dit_train/trainer), so a
# clone can train without an external DialogEdit_qwen checkout. Point
# TRAINER_DIR elsewhere to run against a different copy.
TRAINER_DIR="${TRAINER_DIR:-$cur_dir/trainer}"
if [ ! -f "$TRAINER_DIR/train.py" ]; then
  echo "Trainer not found at $TRAINER_DIR/train.py" >&2
  echo "Set TRAINER_DIR to a directory containing train.py." >&2
  exit 1
fi
cd "$TRAINER_DIR"
QWEN_TRAIN_RESOLUTION_PRESET="${QWEN_TRAIN_RESOLUTION_PRESET:-qwen_edit_1024}"
echo "Using Qwen train resolution preset: $QWEN_TRAIN_RESOLUTION_PRESET"
train_args=(
  "train.py"
  --config_path "$dataset_config"
  --max_pixels 1048576
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
]"
  --tokenizer_path "$BASE_MODEL_DIR/tokenizer"
  --processor_path "$BASE_MODEL_DIR/processor"
  --learning_rate "$LEARNING_RATE"
  --batch_size "$BATCH_SIZE_PER_GPU"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --num_epochs "$NUM_EPOCHS"
  --remove_prefix_in_ckpt "pipe.dit."
  --output_path "$OUTPUT_DIR"
  --trainable_models "dit"
  --use_gradient_checkpointing
  --dataset_num_workers 8
  --find_unused_parameters
  --save_steps="$CHECKPOINTING_STEPS"
  --vis_steps=2000
  --seed="$SEED"
  --global_batch_size="$GLOBAL_BATCH_SIZE"
  --task_buckets=t2i,edit
  --resolution_preset "$QWEN_TRAIN_RESOLUTION_PRESET"
  --zero_cond_t
)

if [ "${PHOTOMASTER_DIRECT_SLURM:-0}" = "1" ]; then
  export RANK="${SLURM_PROCID:?SLURM_PROCID is required for PHOTOMASTER_DIRECT_SLURM=1}"
  export LOCAL_RANK="${SLURM_LOCALID:?SLURM_LOCALID is required for PHOTOMASTER_DIRECT_SLURM=1}"
  export WORLD_SIZE="${SLURM_NTASKS:?SLURM_NTASKS is required for PHOTOMASTER_DIRECT_SLURM=1}"
  export NODE_RANK="${SLURM_NODEID:-$NODE_RANK}"
  export MASTER_ADDR
  export MASTER_PORT
  export ACCELERATE_USE_DEEPSPEED=true
  export ACCELERATE_DEEPSPEED_CONFIG_FILE="$DS_CONFIG_PATH"
  export ACCELERATE_MIXED_PRECISION=bf16
  echo "Direct Slurm launch rank: $RANK/$WORLD_SIZE local_rank=$LOCAL_RANK node_rank=$NODE_RANK master=${MASTER_ADDR}:${MASTER_PORT}"
  python "${train_args[@]}"
  exit $?
else
  accelerate launch \
    --config_file="$ACCELERATE_CONFIG_PATH" \
    --deepspeed_config_file="$DS_CONFIG_PATH" \
    --mixed_precision="bf16" \
    --num_machines="$NNODES" \
    --machine_rank="$NODE_RANK" \
    --num_processes="$NPROCESSES" \
    --main_process_ip="$MASTER_ADDR" \
    --main_process_port="$MASTER_PORT" \
    "${train_args[@]}"
fi
