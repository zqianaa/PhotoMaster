#!/usr/bin/env bash
# Run all three inference stages for one (MODEL, BENCH) pair.
#
#   1. run_vlm_infer.sh        vlm_output.json
#   2. run_vlm_to_dit.sh       dit_input.jsonl
#   3. run_qwen_edit_infer.sh  <SAVE_MODEL_NAME>/*.png
#
# Every stage uses bench/<BENCH>/results/<RUN_NAME>/. This entry point requires
# a new result directory on every invocation to prevent stale JSONL/images from
# being combined with new VLM output. For an interrupted run, resume a stage
# directly only if its inputs, checkpoints and settings have not changed.
#
# Usage:
#   bash scripts/inference/run_pipeline.sh [MODEL] [BENCH]
#
# Every path defaults into <repo>/models/, so a populated models/ directory needs no
# variables at all. All other variables are documented in the individual stage scripts
# and are passed through unchanged. Source an experiment env file to reproduce a
# recorded run:
#
#   set -a; . config/experiments/photomaster_bench_infer.env; set +a
#   bash scripts/inference/run_pipeline.sh vlm photomaster_bench
#
# Note that MODEL_PATH means different things to the two GPU stages — the VLM
# checkpoint in stage 1, the DiT weights in stage 3 — so set it on a stage script
# directly rather than here, where it would reach both.

set -euo pipefail

MODEL="${MODEL:-vlm}"
BENCH="${BENCH:-photomaster_bench}"

MODEL="${1:-$MODEL}"
BENCH="${2:-$BENCH}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Stage 1 derives RUN_NAME from MODEL; stages 2 and 3 take it as their first
# argument, so resolve it once here and keep all three pointed at one directory.
RUN_NAME="${RUN_NAME:-$MODEL}"
export RUN_NAME

if [[ -n "${MODEL_PATH:-}" ]]; then
    echo "[ERROR] MODEL_PATH means different models in stages 1 and 3."
    echo "        Unset it here; use MODEL_ROOT/BASE_MODEL for VLM paths, or run stages separately."
    exit 1
fi
for name in "$BENCH" "$RUN_NAME"; do
    if [[ ! "$name" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]]; then
        echo "[ERROR] BENCH and RUN_NAME must be directory names containing letters, digits, _, - or ."
        exit 1
    fi
done
export GPU_MODE="${GPU_MODE:-single}"
case "$GPU_MODE" in
    single|multi_gpu) ;;
    multi_node)
        echo "[ERROR] Run multi-node jobs using the individual stage launchers on each node."
        echo "        This entry point reserves one fresh run directory for a single host."
        exit 1 ;;
    *) echo "[ERROR] Unknown GPU_MODE: $GPU_MODE"; exit 1 ;;
esac
if [[ "${ALLOW_BASE_DIT:-0}" != "0" && "${ALLOW_BASE_DIT:-0}" != "1" ]] ||
   [[ -n "${FT_CKPT_IS_FULL:-}" && "$FT_CKPT_IS_FULL" != "0" && "$FT_CKPT_IS_FULL" != "1" ]]; then
    echo "[ERROR] ALLOW_BASE_DIT and FT_CKPT_IS_FULL must be 0 or 1."
    exit 1
fi
if [[ "${ALLOW_BASE_DIT:-0}" == "1" ]] &&
   [[ -n "${FT_CKPT_PATH:-}" || "${FT_CKPT_IS_FULL:-0}" == "1" ]]; then
    echo "[ERROR] Base-renderer mode requires FT_CKPT_PATH unset and FT_CKPT_IS_FULL=0."
    exit 1
fi

RESULT_DIR="${REPO_ROOT}/bench/${BENCH}/results/${RUN_NAME}"
if [[ -e "$RESULT_DIR" || -L "$RESULT_DIR" ]]; then
    echo "[ERROR] Result directory already exists: $RESULT_DIR"
    echo "        Choose a new RUN_NAME. Whole-pipeline runs never reuse prior results."
    exit 1
fi

# Check both GPU stages before spending time on VLM generation. This imports
# dependencies and validates local paths, but does not load model weights.
PREFLIGHT_ARGS=(
    --data-dir "${REPO_ROOT}/bench/${BENCH}"
    --vlm-path "${MODEL_ROOT:-${REPO_ROOT}/models/ft_ckpts}/${MODEL}"
    --base-vlm "${BASE_MODEL:-${REPO_ROOT}/models/Qwen3-VL-32B-Instruct}"
    --dit-path "${REPO_ROOT}/models/Qwen-Image-Edit-2511"
)
if [[ -n "${FT_CKPT_PATH:-}" ]]; then
    PREFLIGHT_ARGS+=(--ft-ckpt "$FT_CKPT_PATH")
elif [[ "${ALLOW_BASE_DIT:-0}" == "1" ]]; then
    PREFLIGHT_ARGS+=(--allow-base-dit)
else
    PREFLIGHT_ARGS+=(--ft-ckpt "${REPO_ROOT}/models/ft_ckpts/i2i/photomaster_i2i.safetensors")
fi
python "${REPO_ROOT}/scripts/check_setup.py" "${PREFLIGHT_ARGS[@]}"

# Atomic reservation also prevents two invocations from sharing a run directory.
mkdir -p "$(dirname "$RESULT_DIR")"
if ! mkdir "$RESULT_DIR"; then
    echo "[ERROR] Could not reserve a fresh result directory; choose a new RUN_NAME."
    exit 1
fi

bash "${SCRIPT_DIR}/run_vlm_infer.sh" "$MODEL" "$BENCH"
bash "${SCRIPT_DIR}/run_vlm_to_dit.sh" "$RUN_NAME" "$BENCH"
bash "${SCRIPT_DIR}/run_qwen_edit_infer.sh" "$RUN_NAME" "$BENCH"
