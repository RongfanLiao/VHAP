#!/bin/bash
# Run video_to_flame_param.py on all *right*.mp4 files found recursively in a given directory.
# Local version (no SLURM).
#
# Usage:
#   bash scripts/run_right_videos.sh <DATA_DIR>
#
# Arguments:
#   DATA_DIR  Directory to search recursively for *right*.mp4 files (required).
#
# Example:
#   bash scripts/run_right_videos.sh LookingFace/documentary/Marine_reacts_to_Army_Basic_Training_by_CombatArmsChannel
#   bash scripts/run_right_videos.sh LookingFace/documentary
#

export PYTHONPATH=$PWD:$PYTHONPATH
PY="/gpfs/home/r/rongfan/micromamba/envs/react/bin/python"

DATA_DIR="${1:?Usage: bash scripts/run_right_videos.sh <DATA_DIR>}"

# Recursively find all *right*.mp4 files
FILES=($(find "${DATA_DIR}" -name '*right*.mp4' | sort))
TOTAL=${#FILES[@]}

echo "Found ${TOTAL} right videos"

for i in $(seq 0 $((TOTAL - 1))); do
  INPUT=${FILES[$i]}
  echo "=== [$(date)] Processing $((i+1))/${TOTAL} | ${INPUT} ==="

  ${PY} tools/video_to_flame_param.py --input "$INPUT"

  echo "=== [$(date)] Finished $((i+1))/${TOTAL} ==="
  echo
done

echo "All done."
