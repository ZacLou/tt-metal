#!/bin/bash

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Run tt-train galaxy models on an exabox multihost leg and hand the results back to CI.
#
# Rank 0 lands on a worker whose tt-metal tree the CI container cannot see, so run_models.py
# writes everything to the shared NFS scratch ($PIPELINE_DIR) instead. This script runs in the
# container, so once mpirun exits it can stage that output where the workflow's generic steps
# look for it:
#
#   $GITHUB_WORKSPACE/generated/tt-train-metrics/  metrics JSON for the SFTP upload
#   $PIPELINE_DIR/plots/                           PNGs for the generic plot-artifact step
#   $GITHUB_STEP_SUMMARY                           the run summary table and loss plots
#
# Usage: run_models_multihost.sh <filter-filename>

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "ERROR: expected exactly one --filter-filenames value, e.g. tinyllama_1_1b_ddp32_shakespeare" >&2
    exit 1
fi

FILTER="$1"

: "${TT_METAL_HOME:?TT_METAL_HOME is not set}"
: "${PIPELINE_DIR:?PIPELINE_DIR is not set (shared NFS scratch for this leg)}"

METRICS_DIR="$PIPELINE_DIR/tt-train-metrics"

# The rank's script is double-quoted so FILTER and METRICS_DIR are substituted here — the rank
# inherits neither. Anything added to it that must be evaluated on the rank needs escaping.
rc=0
mpirun -np 1 --bind-to none --tag-output --wdir "$TT_METAL_HOME" bash -lc "
  ./tt-train/scripts/setup_pth.sh
  python tt-train/scripts/run_models.py \
    --model_config tt-train/scripts/run_models_configs/galaxy.yaml \
    --filter-filenames '$FILTER' \
    --output-dir '$METRICS_DIR'
" || rc=$?

# hashFiles() in the workflow only resolves paths under $GITHUB_WORKSPACE, which on these
# runners is a separate copy of the tree from $TT_METAL_HOME.
if [ -n "${GITHUB_WORKSPACE:-}" ] && compgen -G "$METRICS_DIR/*.json" > /dev/null; then
    mkdir -p "$GITHUB_WORKSPACE/generated/tt-train-metrics"
    cp -f "$METRICS_DIR"/*.json "$GITHUB_WORKSPACE/generated/tt-train-metrics/"
fi

if compgen -G "$METRICS_DIR/plots/*.png" > /dev/null; then
    mkdir -p "$PIPELINE_DIR/plots"
    cp -f "$METRICS_DIR"/plots/*.png "$PIPELINE_DIR/plots/"
fi

# Mermaid loss plots render inline in the job summary; the workflow appends a download link
# for the full-resolution PNGs after this step.
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    shopt -s nullglob
    for md in "$METRICS_DIR/summary.md" "$METRICS_DIR"/plots/*.md; do
        if [ -s "$md" ]; then
            cat "$md" >> "$GITHUB_STEP_SUMMARY"
        fi
    done
    shopt -u nullglob
fi

exit "$rc"
