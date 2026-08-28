#!/usr/bin/env bash
# Stage 1 of the inference pipeline — run VLM inference on a benchmark directory.
#
# Loads a Qwen3-VL checkpoint (full-finetuned or LoRA) and runs vlm_infer.py against
# one bench. Output is written to bench/<BENCH>/results/<RUN_NAME>/vlm_output.json.
#
# Usage:
#   bash scripts/inference/run_vlm_infer.sh [MODEL] [BENCH]
#
# Arguments (both optional; environment variables of the same name are the defaults):
#   MODEL   Model directory name under MODEL_ROOT (default: Qwen3-VL-32B-Instruct)
#   BENCH   Bench directory name under bench/     (default: photomaster_bench)
#
# Environment variables:
#   MODEL_ROOT      Directory containing model checkpoints
#   MODEL_PATH      Full path to the checkpoint; overrides MODEL_ROOT/MODEL
#   BASE_MODEL      Base model path for LoRA checkpoints. If unset, the path recorded
#                   in the adapter's adapter_config.json is used.
#   RUN_NAME        Result directory name under results/ (default: $MODEL)
#   SYSTEM_PROMPT   Prompt file; relative paths resolve against prompts/ (default: system_prompt.txt)
#   MAX_NEW_TOKENS, TEMPERATURE, TOP_P, TOP_K   Generation settings
#   NUM_SAMPLES     If set, process only the first N samples (smoke tests)
#   GPU_MODE        single / multi_gpu / multi_node (default: single)
#   GPUS_PER_NODE   GPUs per node (default: 8)
#   NUM_NODES       Number of nodes, multi_node only (default: 1)
#   MASTER_ADDR, MASTER_PORT   Rendezvous endpoint for multi_node
#
# Examples:
#   bash scripts/inference/run_vlm_infer.sh
#   MODEL_ROOT=/path/to/ft_ckpts GPU_MODE=multi_gpu \
#       bash scripts/inference/run_vlm_infer.sh my_lora_ckpt photomaster_bench

set -euo pipefail

START_TIME=$(date +%s)

# -----------------------------------------------------------------------
# Defaults — all overridable from the environment.
# -----------------------------------------------------------------------
MODEL="${MODEL:-Qwen3-VL-32B-Instruct}"
BENCH="${BENCH:-photomaster_bench}"

# Args override defaults if provided.
MODEL="${1:-$MODEL}"
BENCH="${2:-$BENCH}"

# -----------------------------------------------------------------------
# Resolve all paths relative to the repository root (two levels up).
# -----------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/models}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/${MODEL}}"
DATA_DIR="${REPO_ROOT}/bench/${BENCH}"
INFER_SCRIPT="${SCRIPT_DIR}/vlm_infer.py"

SYSTEM_PROMPT="${SYSTEM_PROMPT:-system_prompt.txt}"
if [[ "$SYSTEM_PROMPT" != /* ]]; then
    SYSTEM_PROMPT="${SCRIPT_DIR}/prompts/${SYSTEM_PROMPT}"
fi

RUN_NAME="${RUN_NAME:-$MODEL}"
OUTPUT_DIR="${DATA_DIR}/results/${RUN_NAME}"
OUTPUT_JSON="${OUTPUT_DIR}/vlm_output.json"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.8}"
TOP_K="${TOP_K:-20}"
GPU_MODE="${GPU_MODE:-single}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NUM_NODES="${NUM_NODES:-1}"

echo "============================================"
echo "  Stage 1 — VLM Inference"
echo "============================================"
echo "  Repo root:       $REPO_ROOT"
echo "  Model:           $MODEL"
echo "  Model path:      $MODEL_PATH"
if [ -n "${BASE_MODEL:-}" ]; then
    echo "  Base model:      $BASE_MODEL"
fi
echo "  Bench:           $BENCH"
echo "  Data dir:        $DATA_DIR"
echo "  Run name:        $RUN_NAME"
echo "  Output:          $OUTPUT_JSON"
echo "  System prompt:   $SYSTEM_PROMPT"
echo "  Max new tokens:  $MAX_NEW_TOKENS"
echo "  Temperature:     $TEMPERATURE"
echo "  Top-p / Top-k:   $TOP_P / $TOP_K"
echo "  GPU mode:        $GPU_MODE"
echo "============================================"

# -----------------------------------------------------------------------
# Validate
# -----------------------------------------------------------------------
if [ ! -d "$MODEL_PATH" ]; then
    echo "[ERROR] Model directory not found: $MODEL_PATH"
    exit 1
fi
if [ ! -d "$DATA_DIR" ]; then
    echo "[ERROR] Bench directory not found: $DATA_DIR"
    exit 1
fi
if [ ! -f "$SYSTEM_PROMPT" ]; then
    echo "[ERROR] System prompt file not found: $SYSTEM_PROMPT"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Ranks gather their shards through the filesystem, so this must be storage
# every node can see — not a node-local /tmp.
export PHOTOMASTER_GATHER_DIR="${PHOTOMASTER_GATHER_DIR:-${OUTPUT_DIR}/.gather}"

# -----------------------------------------------------------------------
# Build and run command
# -----------------------------------------------------------------------
COMMON_ARGS=(
    --model-path "$MODEL_PATH"
    --data-dir "$DATA_DIR"
    --output-json "$OUTPUT_JSON"
    --system-prompt-file "$SYSTEM_PROMPT"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --top-k "$TOP_K"
)
if [ -n "${BASE_MODEL:-}" ]; then
    COMMON_ARGS+=(--base-model-path "$BASE_MODEL")
fi
if [ -n "${NUM_SAMPLES:-}" ]; then
    COMMON_ARGS+=(--num_samples "$NUM_SAMPLES")
fi

echo ""
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
        --rdzv_endpoint="${MASTER_ADDR:-localhost}:${MASTER_PORT:-29500}" \
        "$INFER_SCRIPT" "${COMMON_ARGS[@]}"
else
    echo "[ERROR] Unknown GPU_MODE: $GPU_MODE"
    exit 1
fi

rm -rf "$PHOTOMASTER_GATHER_DIR" 2>/dev/null || true

echo ""
echo "============================================"
echo "[RESULT] Output saved to: $OUTPUT_JSON"
ELAPSED=$(( $(date +%s) - START_TIME ))
printf "[TIME] Total: %dm %ds\n" $((ELAPSED / 60)) $((ELAPSED % 60))
echo "============================================"
