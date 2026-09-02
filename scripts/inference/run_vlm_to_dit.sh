#!/usr/bin/env bash
# Stage 2 of the inference pipeline — convert VLM output into DiT input.
#
# Reads bench/<BENCH>/results/<RUN_NAME>/vlm_output.json and writes dit_input.jsonl
# beside it. The output must stay at that path: qwen_edit_infer.py resolves relative
# image paths against the JSONL's grandparent directory, which is the bench root.
#
# Usage:
#   bash scripts/inference/run_vlm_to_dit.sh [RUN_NAME] [BENCH]
#
# Arguments (both optional; environment variables of the same name are the defaults):
#   RUN_NAME  Result directory name under results/ (default: vlm)
#   BENCH     Bench directory name under bench/    (default: photomaster_bench)
#
# Environment variables:
#   FORCE   If 1, overwrite an existing dit_input.jsonl (default: skip and keep it)

set -euo pipefail

RUN_NAME="${RUN_NAME:-vlm}"
BENCH="${BENCH:-photomaster_bench}"

RUN_NAME="${1:-$RUN_NAME}"
BENCH="${2:-$BENCH}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RESULT_DIR="${REPO_ROOT}/bench/${BENCH}/results/${RUN_NAME}"
VLM_JSON="${RESULT_DIR}/vlm_output.json"
DIT_JSONL="${RESULT_DIR}/dit_input.jsonl"
CONVERT_SCRIPT="${SCRIPT_DIR}/vlm_to_dit_bench_reformat.py"
FORCE="${FORCE:-0}"

echo "============================================"
echo "  Stage 2 — VLM output to DiT input"
echo "============================================"
echo "  Repo root:       $REPO_ROOT"
echo "  Bench:           $BENCH"
echo "  Run name:        $RUN_NAME"
echo "  Input:           $VLM_JSON"
echo "  Output:          $DIT_JSONL"
echo "============================================"

if [ ! -f "$VLM_JSON" ]; then
    echo "[ERROR] vlm_output.json not found: $VLM_JSON"
    echo "        Run scripts/inference/run_vlm_infer.sh first."
    exit 1
fi

if [ -f "$DIT_JSONL" ] && [ "$FORCE" != "1" ]; then
    echo "[SKIP] dit_input.jsonl already exists ($(wc -l < "$DIT_JSONL") samples)."
    echo "       Set FORCE=1 to regenerate it."
    exit 0
fi

python "$CONVERT_SCRIPT" "$VLM_JSON" -o "$DIT_JSONL"

echo ""
echo "============================================"
echo "[RESULT] Wrote $(wc -l < "$DIT_JSONL") samples to: $DIT_JSONL"
echo "============================================"
