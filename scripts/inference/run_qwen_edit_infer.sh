#!/usr/bin/env bash
# Stage 3 of the inference pipeline — run Qwen-Image-Edit-2511 on DiT input.
#
# Reads bench/<BENCH>/results/<RUN_NAME>/dit_input.jsonl and writes edited images to
# bench/<BENCH>/results/<RUN_NAME>/<SAVE_MODEL_NAME>/.
#
# Already-generated images are skipped, so re-running the same command resumes an
# interrupted run. The script exits non-zero if any sample failed or if the output
# image count is short of the input sample count.
#
# Usage:
#   bash scripts/inference/run_qwen_edit_infer.sh [RUN_NAME] [BENCH]
#
# Arguments (both optional; environment variables of the same name are the defaults):
#   RUN_NAME  Result directory name under results/ (default: vlm)
#   BENCH     Bench directory name under bench/    (default: photomaster_bench)
#
# Environment variables:
#   MODEL_PATH       Path to Qwen-Image-Edit-2511 weights
#                    (default: <repo>/models/Qwen-Image-Edit-2511)
#   FT_CKPT_PATH     Optional fine-tuned DiT .safetensors checkpoint. Unset by default,
#                    so the base model runs; the I2I training output lands under
#                    <repo>/models/ft_ckpts/i2i/ (e.g. epoch-2.safetensors).
#   FT_CKPT_IS_FULL  If 1, require a complete 2511 transformer checkpoint and skip
#                    base transformer weights; transformer/config.json is still required (default: 0).
#   SAVE_MODEL_NAME  Output subfolder name (default: qwen_edit_images)
#   TRUE_CFG_SCALE   True CFG scale (default: 4.0)
#   INFERENCE_STEPS  Number of inference steps (default: 50)
#   SEED             Random seed (default: 0)
#   QWEN_OUTPUT_RESOLUTION      Square-equivalent output resolution: 1024 or 1328 (default: 1328)
#   QWEN_VAE_INPUT_RESOLUTION   Square-equivalent VAE input resolution: 1024 or 1328 (default: 1328)
#   GPU_MODE         single / multi_gpu / multi_node (default: multi_gpu)
#   GPUS_PER_NODE    GPUs per node (default: 8)
#   NUM_NODES        Number of nodes, multi_node only (default: 1)
#   MASTER_ADDR      Rendezvous host for multi_node (default: localhost)

set -euo pipefail

START_TIME=$(date +%s)

RUN_NAME="${RUN_NAME:-vlm}"
BENCH="${BENCH:-photomaster_bench}"

RUN_NAME="${1:-$RUN_NAME}"
BENCH="${2:-$BENCH}"

# -----------------------------------------------------------------------
# Resolve all paths relative to the repository root (two levels up).
# -----------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/models/Qwen-Image-Edit-2511}"
FT_CKPT_PATH="${FT_CKPT_PATH:-}"
FT_CKPT_IS_FULL="${FT_CKPT_IS_FULL:-0}"
SAVE_MODEL_NAME="${SAVE_MODEL_NAME:-qwen_edit_images}"
TRUE_CFG_SCALE="${TRUE_CFG_SCALE:-4.0}"
INFERENCE_STEPS="${INFERENCE_STEPS:-50}"
SEED="${SEED:-0}"
QWEN_OUTPUT_RESOLUTION="${QWEN_OUTPUT_RESOLUTION:-1328}"
QWEN_VAE_INPUT_RESOLUTION="${QWEN_VAE_INPUT_RESOLUTION:-1328}"
GPU_MODE="${GPU_MODE:-multi_gpu}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NUM_NODES="${NUM_NODES:-1}"

RESULT_DIR="${REPO_ROOT}/bench/${BENCH}/results/${RUN_NAME}"
JSONL_PATH="${RESULT_DIR}/dit_input.jsonl"
OUTPUT_DIR="${RESULT_DIR}/${SAVE_MODEL_NAME}"
INFER_SCRIPT="${SCRIPT_DIR}/qwen_edit_infer.py"

echo "============================================"
echo "  Stage 3 — Qwen Image Edit 2511 Inference"
echo "============================================"
echo "  Repo root:       $REPO_ROOT"
echo "  Model path:      $MODEL_PATH"
echo "  FT checkpoint:   ${FT_CKPT_PATH:-<none>}"
echo "  Full FT weights: $FT_CKPT_IS_FULL"
echo "  Bench:           $BENCH"
echo "  Run name:        $RUN_NAME"
echo "  JSONL:           $JSONL_PATH"
echo "  Output dir:      $OUTPUT_DIR"
echo "  True CFG scale:  $TRUE_CFG_SCALE"
echo "  Inference steps: $INFERENCE_STEPS"
echo "  Seed:            $SEED"
echo "  Output res:      $QWEN_OUTPUT_RESOLUTION"
echo "  VAE input res:   $QWEN_VAE_INPUT_RESOLUTION"
echo "  GPU mode:        $GPU_MODE"
echo "  GPUs per node:   $GPUS_PER_NODE"
echo "  Num nodes:       $NUM_NODES"
echo "============================================"

# -----------------------------------------------------------------------
# Validate
# -----------------------------------------------------------------------
if [[ "$FT_CKPT_IS_FULL" != "0" && "$FT_CKPT_IS_FULL" != "1" ]]; then
    echo "[ERROR] FT_CKPT_IS_FULL must be 0 or 1."
    exit 1
fi
if [[ "$FT_CKPT_IS_FULL" == "1" && "$FT_CKPT_PATH" != *.safetensors ]]; then
    echo "[ERROR] FT_CKPT_IS_FULL=1 requires FT_CKPT_PATH to a complete .safetensors file."
    exit 1
fi
if [ ! -d "$MODEL_PATH" ]; then
    echo "[ERROR] Qwen-Image-Edit-2511 weights not found: $MODEL_PATH"
    echo "        Place them under <repo>/models/ or set MODEL_PATH."
    exit 1
fi
if [ -n "$FT_CKPT_PATH" ] && [ ! -f "$FT_CKPT_PATH" ]; then
    echo "[ERROR] FT checkpoint not found: $FT_CKPT_PATH"
    exit 1
fi
if [ ! -f "$JSONL_PATH" ]; then
    echo "[ERROR] JSONL not found: $JSONL_PATH"
    echo "        Run scripts/inference/run_vlm_to_dit.sh first."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# -----------------------------------------------------------------------
# Build and run command
# -----------------------------------------------------------------------
COMMON_ARGS=(
    --model-path "$MODEL_PATH"
    --jsonl-path "$JSONL_PATH"
    --output-dir "$OUTPUT_DIR"
    --true-cfg-scale "$TRUE_CFG_SCALE"
    --inference-steps "$INFERENCE_STEPS"
    --seed "$SEED"
    --output-resolution "$QWEN_OUTPUT_RESOLUTION"
    --vae-input-resolution "$QWEN_VAE_INPUT_RESOLUTION"
)

if [ -n "$FT_CKPT_PATH" ]; then
    COMMON_ARGS+=(--ckpt-path "$FT_CKPT_PATH")
fi
if [ "$FT_CKPT_IS_FULL" = "1" ]; then
    COMMON_ARGS+=(--full-transformer-checkpoint)
fi

if [ "$GPU_MODE" = "single" ]; then
    echo "[MODE] Single GPU"
    python "$INFER_SCRIPT" "${COMMON_ARGS[@]}"

elif [ "$GPU_MODE" = "multi_gpu" ]; then
    echo "[MODE] Multi-GPU ($GPUS_PER_NODE GPUs)"
    torchrun --nproc_per_node="$GPUS_PER_NODE" \
        "$INFER_SCRIPT" "${COMMON_ARGS[@]}"

elif [ "$GPU_MODE" = "multi_node" ]; then
    echo "[MODE] Multi-Node ($NUM_NODES nodes x $GPUS_PER_NODE GPUs)"
    torchrun \
        --nnodes="$NUM_NODES" \
        --nproc_per_node="$GPUS_PER_NODE" \
        --rdzv_id="${SLURM_JOB_ID:-0}" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${MASTER_ADDR:-localhost}:29500" \
        "$INFER_SCRIPT" "${COMMON_ARGS[@]}"
else
    echo "[ERROR] Unknown GPU_MODE: $GPU_MODE"
    exit 1
fi

echo ""
echo "============================================"
echo "[RESULT] Output saved to: $OUTPUT_DIR"
ELAPSED=$(( $(date +%s) - START_TIME ))
printf "[TIME] Total: %dm %ds\n" $((ELAPSED / 60)) $((ELAPSED % 60))
echo "============================================"
