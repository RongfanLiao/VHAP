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
#   sbatch scripts/run_right_videos.slurm <DATA_DIR>
#
# Arguments:
#   DATA_DIR  Directory to search recursively for *right*.mp4 files (required).
#
# Example:
#   sbatch scripts/run_right_videos.slurm LookingFace/documentary/Marine_reacts_to_Army_Basic_Training_by_CombatArmsChannel
#   sbatch scripts/run_right_videos.slurm LookingFace/documentary
#

export PYTHONPATH=$PWD:$PYTHONPATH
PY="/gpfs/home/r/rongfan/micromamba/envs/react/bin/python"

DATA_DIR="${1:?Usage: sbatch scripts/run_right_videos.slurm <DATA_DIR>}"

# Recursively find all *right*.mp4 files
FILES=($(find "${DATA_DIR}" -name '*right*.mp4' | sort))
TOTAL=${#FILES[@]}

echo "Found ${TOTAL} right videos"
echo "JOBID=${SLURM_JOB_ID}"

for i in $(seq 0 $((TOTAL - 1))); do
  INPUT=${FILES[$i]}
  echo "=== [$(date)] Processing $((i+1))/${TOTAL} | ${INPUT} ==="

  ${PY} tools/video_to_flame_param.py --input "$INPUT"

  echo "=== [$(date)] Finished $((i+1))/${TOTAL} ==="
  echo
done

echo "All done."
