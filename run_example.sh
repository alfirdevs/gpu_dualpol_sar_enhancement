#!/usr/bin/env bash
set -e

# Edit these paths before running.
VV_PATH="/path/to/VV.tif"
VH_PATH="/path/to/VH.tif"
OUTDIR="results"

python3 gpu_dualpol_sar_enhancement.py \
  --vv "$VV_PATH" \
  --vh "$VH_PATH" \
  --outdir "$OUTDIR" \
  --benchmark-cpu
