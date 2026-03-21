#!/bin/bash
# Run video_to_flame_param.py on all *right*.mp4 files found recursively in a given directory.
# Local version (no SLURM).
#
# Usage:
#   bash scripts/run_right_videos.sh <DATA_DIR> [DATA_DIR2 ...]
#
# Arguments:
#   DATA_DIR  One or more directories to search recursively for *right*.mp4 files (at least one required).
#
# Example:
#   bash scripts/run_right_videos.sh LookingFace/documentary/Marine_reacts_to_Army_Basic_Training_by_CombatArmsChannel
#   bash scripts/run_right_videos.sh LookingFace/documentary LookingFace/music
#

export PYTHONPATH=$PWD:$PYTHONPATH
PY="/gpfs/home/r/rongfan/micromamba/envs/react/bin/python"

if [ $# -eq 0 ]; then
  echo "Usage: bash scripts/run_right_videos.sh <DATA_DIR> [DATA_DIR2 ...]"
  exit 1
fi

# Recursively find all *right*.mp4 files across all given directories
FILES=()
for DIR in "$@"; do
  while IFS= read -r -d '' f; do
    FILES+=("$f")
  done < <(find "$DIR" -name '*right*.mp4' -print0 | sort -z)
done
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
