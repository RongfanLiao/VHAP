#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --partition=gpulowpri
#SBATCH --time=24:00:00
#SBATCH --account=su004-gpulowpri

# Run video_to_flame_param.py on all *right*.mp4 files found recursively in a given directory.
#
# Usage:
#   sbatch scripts/run_right_videos.slurm <DATA_DIR> [DATA_DIR2 ...]
#
# Arguments:
#   DATA_DIR  One or more directories to search recursively for *right*.mp4 files (at least one required).
#
# Example:
#   sbatch scripts/run_right_videos.slurm LookingFace/documentary/Marine_reacts_to_Army_Basic_Training_by_CombatArmsChannel
#   sbatch scripts/run_right_videos.slurm LookingFace/documentary LookingFace/music
#

export PYTHONPATH=$PWD:$PYTHONPATH
PY="/gpfs/home/r/rongfan/micromamba/envs/react/bin/python"

if [ $# -eq 0 ]; then
  echo "Usage: sbatch scripts/run_right_videos.slurm <DATA_DIR> [DATA_DIR2 ...]"
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
echo "JOBID=${SLURM_JOB_ID}"

for i in $(seq 0 $((TOTAL - 1))); do
  INPUT=${FILES[$i]}
  echo "=== [$(date)] Processing $((i+1))/${TOTAL} | ${INPUT} ==="

  ${PY} tools/video_to_flame_param.py "$INPUT"

  echo "=== [$(date)] Finished $((i+1))/${TOTAL} ==="
  echo
done

echo "All done."
