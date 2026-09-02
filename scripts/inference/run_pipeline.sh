#!/usr/bin/env bash
# Run all three inference stages for one (MODEL, BENCH) pair.
#
#   1. run_vlm_infer.sh        vlm_output.json
#   2. run_vlm_to_dit.sh       dit_input.jsonl
#   3. run_qwen_edit_infer.sh  <SAVE_MODEL_NAME>/*.png
#
# Every stage keys off the same result directory,
# bench/<BENCH>/results/<RUN_NAME>/, so the chain can be re-entered at any stage by
# calling that stage's script directly. Stages 2 and 3 skip work that is already
# done, so re-running this script resumes rather than restarts.
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

# Stage 1 derives RUN_NAME from MODEL; stages 2 and 3 take it as their first
# argument, so resolve it once here and keep all three pointed at one directory.
RUN_NAME="${RUN_NAME:-$MODEL}"
export RUN_NAME

bash "${SCRIPT_DIR}/run_vlm_infer.sh" "$MODEL" "$BENCH"
bash "${SCRIPT_DIR}/run_vlm_to_dit.sh" "$RUN_NAME" "$BENCH"
bash "${SCRIPT_DIR}/run_qwen_edit_infer.sh" "$RUN_NAME" "$BENCH"
