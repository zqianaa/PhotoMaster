#!/bin/bash
#SBATCH --job-name=qwen_edit_train
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --output=qwen_edit_train_%j.log
#SBATCH --error=qwen_edit_train_%j.log

# Multi-node Qwen-Image-Edit-2511 training entry point.
#
# Submit with (node/GPU counts must be given on the command line, because
# #SBATCH directives are parsed before this script runs):
#
#   sbatch --nodes=4 --ntasks-per-node=8 --gres=gpu:8 \
#       scripts/dit_train/train_qwen_edit_2511.slurm \
#       config/experiments/lingbot_ablation.env
#
# One task per GPU. Each task runs launch_qwen_edit_2511_train.sh for its rank.

set -uo pipefail

# Locating the repo is not as simple as dirname "$0": sbatch copies the batch script
# into a Slurm spool directory, so under sbatch ${BASH_SOURCE[0]} points at
# /tmp/slurm/spool/... and not at the checkout. Search for a marker file instead,
# starting from the submission directory, and fall back to the script's own location
# for a plain `bash scripts/dit_train/train_qwen_edit_2511.slurm` run.
MARKER="scripts/dit_train/launch_qwen_edit_2511_train.sh"

find_repo_root() {
    local d="$1"
    [ -n "$d" ] || return 1
    d=$(cd "$d" 2>/dev/null && pwd) || return 1
    while :; do
        if [ -f "$d/$MARKER" ]; then
            echo "$d"
            return 0
        fi
        [ "$d" = "/" ] && return 1
        d=$(dirname "$d")
    done
}

REPO_ROOT="${REPO_ROOT:-}"
if [ -z "$REPO_ROOT" ]; then
    REPO_ROOT=$(find_repo_root "${SLURM_SUBMIT_DIR:-}") \
        || REPO_ROOT=$(find_repo_root "$(dirname "${BASH_SOURCE[0]:-$0}")") \
        || REPO_ROOT=$(find_repo_root "$PWD") \
        || {
            echo "Cannot locate the repository root (looking for $MARKER)." >&2
            echo "Submit from inside the checkout, or set REPO_ROOT explicitly." >&2
            exit 1
        }
fi

SCRIPT_DIR="$REPO_ROOT/scripts/dit_train"
WORKER_SCRIPT="$SCRIPT_DIR/launch_qwen_edit_2511_train.sh"

if [ ! -f "$WORKER_SCRIPT" ]; then
    echo "Worker script not found: $WORKER_SCRIPT" >&2
    exit 1
fi

# ======= EXPERIMENT CONFIG =======
# $1 is an experiment env file. Relative paths resolve against the repo root.
EXPERIMENT_CONFIG="${1:-config/experiments/lingbot_ablation.env}"
case "$EXPERIMENT_CONFIG" in
    /*) ;;
    *) EXPERIMENT_CONFIG="$REPO_ROOT/$EXPERIMENT_CONFIG" ;;
esac

if [ ! -f "$EXPERIMENT_CONFIG" ]; then
    echo "Experiment config not found: $EXPERIMENT_CONFIG" >&2
    exit 1
fi

echo "Loading experiment config: $EXPERIMENT_CONFIG"
set -a
# shellcheck disable=SC1090
. "$EXPERIMENT_CONFIG"
set +a

# Resolve DATA_CONFIG_PATH relative to the repo root when it is not absolute.
if [ -n "${DATA_CONFIG_PATH:-}" ]; then
    case "$DATA_CONFIG_PATH" in
        /*) ;;
        *) DATA_CONFIG_PATH="$REPO_ROOT/$DATA_CONFIG_PATH" ;;
    esac
    export DATA_CONFIG_PATH
fi

if [ -z "${OUTPUT_PATH:-}" ]; then
    echo "OUTPUT_PATH must be set in $EXPERIMENT_CONFIG" >&2
    exit 1
fi
case "$OUTPUT_PATH" in
    /*) ;;
    *) OUTPUT_PATH="$REPO_ROOT/$OUTPUT_PATH" ;;
esac
export OUTPUT_PATH

# ======= TOPOLOGY =======
# Derive from the actual allocation so these can never drift from the -N/-n flags.
NUM_NODES="${SLURM_NNODES:-1}"
GPUS_PER_NODE="${SLURM_NTASKS_PER_NODE:-${SLURM_GPUS_ON_NODE:-8}}"
TOTAL_TASKS="${SLURM_NTASKS:-$((NUM_NODES * GPUS_PER_NODE))}"

export NGPUS_PER_NODE="$GPUS_PER_NODE"
export GPUS_PER_NODE

if [ -n "${SLURM_NODELIST:-}" ]; then
    MASTER_ADDR=$(python3 -c "
import re
nl = '${SLURM_NODELIST}'
m = re.match(r'(.+)\[([^\]]+)\]', nl)
if m:
    prefix, suffixes = m.group(1), m.group(2)
    first = suffixes.split(',')[0].split('-')[0]
    print(prefix + first)
else:
    print(nl.split(',')[0])
")
else
    MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
fi
export MASTER_ADDR
export MASTER_PORT="${MASTER_PORT:-21638}"

# The worker derives its own RANK/LOCAL_RANK/WORLD_SIZE from SLURM_* per task.
# Only these two values have to be supplied from outside, and both are constant
# across tasks, so exporting them here is safe.
export PHOTOMASTER_DIRECT_SLURM=1
export VC_WORKER_HOSTS="$MASTER_ADDR"
export MA_NUM_HOSTS="$NUM_NODES"

# ======= COMPLETION MARKER =======
# The trainer writes this on success. Use it to distinguish a real failure from a
# non-zero exit during distributed teardown, which is common and harmless.
SUCCESS_MARKER="$OUTPUT_PATH/.photomaster_training_complete.json"
SUCCESS_TOKEN="${SLURM_JOB_ID:-manual}:$$"
POST_SUCCESS_GRACE_SECONDS="${POST_SUCCESS_GRACE_SECONDS:-600}"
export PHOTOMASTER_SUCCESS_TOKEN="$SUCCESS_TOKEN"

mkdir -p "$OUTPUT_PATH"
if [ -e "$SUCCESS_MARKER" ] && ! rm -f "$SUCCESS_MARKER" 2>/dev/null; then
    echo "Unable to remove stale success marker: $SUCCESS_MARKER" >&2
    exit 1
fi

success_marker_matches() {
    [ -f "$SUCCESS_MARKER" ] || return 1
    python3 - "$SUCCESS_MARKER" "$SUCCESS_TOKEN" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as f:
        marker = json.load(f)
except Exception:
    sys.exit(1)

sys.exit(0 if marker.get("success_token") == sys.argv[2] else 1)
PY
}

cleanup() {
    echo "Received termination signal; stopping training"
    kill "${SRUN_PID:-}" 2>/dev/null || true
    exit 1
}
trap cleanup SIGTERM SIGINT

nvidia-smi

cat <<EOF
========== Qwen-Image-Edit-2511 training ==========
REPO_ROOT                  = $REPO_ROOT
EXPERIMENT_CONFIG          = $EXPERIMENT_CONFIG
NUM_NODES                  = $NUM_NODES
GPUS_PER_NODE              = $GPUS_PER_NODE
TOTAL_TASKS                = $TOTAL_TASKS
MASTER_ADDR                = $MASTER_ADDR:$MASTER_PORT
OUTPUT_PATH                = $OUTPUT_PATH
DATA_CONFIG_PATH           = ${DATA_CONFIG_PATH:-<worker default>}
LEARNING_RATE              = ${LEARNING_RATE:-<worker default>}
NUM_EPOCHS                 = ${NUM_EPOCHS:-<worker default>}
CHECKPOINTING_STEPS        = ${CHECKPOINTING_STEPS:-<worker default>}
BATCH_SIZE_PER_GPU         = ${BATCH_SIZE_PER_GPU:-<worker default>}
GRADIENT_ACCUMULATION_STEPS= ${GRADIENT_ACCUMULATION_STEPS:-<worker default>}
RESOLUTION_PRESET          = ${QWEN_TRAIN_RESOLUTION_PRESET:-<worker default>}
SUCCESS_MARKER             = $SUCCESS_MARKER
===================================================
EOF

srun --export=ALL --kill-on-bad-exit=1 \
    --ntasks="$TOTAL_TASKS" --ntasks-per-node="$GPUS_PER_NODE" \
    bash "$WORKER_SCRIPT" &
SRUN_PID=$!

# Watch for the success marker. If the trainer has finished but srun is still
# alive past the grace period, it is hung in teardown -- terminate it.
SUCCESS_SEEN=0
SUCCESS_DEADLINE=0
POST_SUCCESS_TERMINATED=0

while kill -0 "$SRUN_PID" 2>/dev/null; do
    if success_marker_matches; then
        if [ "$SUCCESS_SEEN" -eq 0 ]; then
            SUCCESS_SEEN=1
            SUCCESS_DEADLINE=$(($(date +%s) + POST_SUCCESS_GRACE_SECONDS))
            echo "Training success marker detected; allowing ${POST_SUCCESS_GRACE_SECONDS}s for distributed shutdown"
        elif [ "$(date +%s)" -ge "$SUCCESS_DEADLINE" ]; then
            echo "Training complete but srun still active after grace period; terminating"
            POST_SUCCESS_TERMINATED=1
            kill "$SRUN_PID" 2>/dev/null || true
            sleep 20
            kill -0 "$SRUN_PID" 2>/dev/null && kill -KILL "$SRUN_PID" 2>/dev/null || true
            break
        fi
    fi
    sleep 10
done

wait "$SRUN_PID"
TRAIN_EXIT_CODE=$?

if success_marker_matches; then
    if [ "$TRAIN_EXIT_CODE" -ne 0 ] || [ "$POST_SUCCESS_TERMINATED" -eq 1 ]; then
        echo "Training completed (marker written); srun exited $TRAIN_EXIT_CODE during shutdown"
    else
        echo "Training completed successfully"
    fi
    echo "Checkpoints in: $OUTPUT_PATH"
    exit 0
fi

if [ "$TRAIN_EXIT_CODE" -ne 0 ]; then
    echo "Training exited with code $TRAIN_EXIT_CODE" >&2
else
    echo "Training exited without writing the success marker" >&2
fi
exit 1
