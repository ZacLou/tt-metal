#!/usr/bin/env bash
# c-perf-recon single-run driver. One pytest process, never piped.
# usage: run_recon.sh <label> <envfile> <nodeid> <timeout_sec> [extra pytest args...]
#   <envfile> is a shell fragment sourced before the run; it must export the
#   model-selecting variables (LLAMA_DIR/HF_MODEL/TT_CACHE_PATH).
set -u
REPO=/proj_sw/user_dev/ctr-apbernal/tt-metal
cd "$REPO" || exit 91
LABEL="$1"; ENVFILE="$2"; NODEID="$3"; TMO="$4"; shift 4
OUT="$REPO/tttv2_milestone_c_evidence/perf/recon/$LABEL"
if [ -e "$OUT" ]; then echo "REFUSING: $OUT exists (never overwrite a log)"; exit 92; fi
mkdir -p "$OUT"

export TT_METAL_HOME="$REPO"
export PYTHONPATH="$REPO"
export HF_HOME=/localdev/ctr-apbernal/hf_data
unset LLAMA_DIR HF_MODEL TT_CACHE_PATH MESH_DEVICE TTTV2_GALAXY_CCL_TRACE LINE_RS LINE_AG
# shellcheck disable=SC1090
. "$ENVFILE"

{
  echo "label=$LABEL"
  echo "utc_start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "epoch_start=$(date +%s)"
  echo "commit=$(git rev-parse HEAD)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "tracked_dirty=$(git status --porcelain --untracked-files=no | wc -l)"
  echo "host=$(hostname)"
  echo "boards=$(ls /sys/class/tenstorrent | wc -l)"
} > "$OUT/meta.txt"

cp "$ENVFILE" "$OUT/envfile.sh"
{
  echo "cd $REPO"
  echo "export TT_METAL_HOME=$TT_METAL_HOME"
  echo "export PYTHONPATH=$PYTHONPATH"
  echo "export HF_HOME=$HF_HOME"
  echo "# --- from $ENVFILE ---"
  cat "$ENVFILE"
  echo "# --- command ---"
  echo "timeout --signal=TERM --kill-after=180 $TMO python -m pytest -v -rA --color=no -p no:cacheprovider $* '$NODEID'"
} > "$OUT/cmd.txt"

env | sort > "$OUT/env.txt"
tt-smi -ls > "$OUT/tt-smi-ls.pre.txt" 2>&1

LOG="$OUT/run.log"
S=$(date +%s)
timeout --signal=TERM --kill-after=180 "$TMO" \
  python -m pytest -v -rA --color=no -p no:cacheprovider "$@" "$NODEID" > "$LOG" 2>&1
RC=$?
E=$(date +%s)
echo "exit=$RC" >> "$LOG"

tt-smi -ls > "$OUT/tt-smi-ls.post.txt" 2>&1
{
  echo "utc_end=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "epoch_end=$E"
  echo "wall_clock_s=$((E-S))"
  echo "exit=$RC"
} >> "$OUT/meta.txt"
echo "$LABEL rc=$RC wall_s=$((E-S))" >> "$REPO/tttv2_milestone_c_evidence/perf/recon/QUEUE_RESULTS.txt"
exit $RC
