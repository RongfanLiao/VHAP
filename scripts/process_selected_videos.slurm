#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --partition=gpulowpri
#SBATCH --time=24:00:00
#SBATCH --account=su004-gpulowpri

export PYTHONPATH=$PWD:$PYTHONPATH
PY="/gpfs/home/r/rongfan/micromamba/envs/react/bin/python"

FILES=(
  LookingFace/music/Avenged_Sevenfold_by_Rayactions/0755_right.mp4
  LookingFace/music/Avenged_Sevenfold_by_Rayactions/0756_right.mp4
  LookingFace/music/Avenged_Sevenfold_by_Rayactions/0757_right.mp4
  LookingFace/music/Avenged_Sevenfold_by_Rayactions/0758_right.mp4
  LookingFace/music/Avenged_Sevenfold_by_Rayactions/0759_right.mp4
  LookingFace/music/Avenged_Sevenfold_by_Rayactions/0760_right.mp4
)
TOTAL=${#FILES[@]}

echo "Processing ${TOTAL} videos"
echo "JOBID=${SLURM_JOB_ID}"

for i in $(seq 0 $((TOTAL - 1))); do
  INPUT=${FILES[$i]}
  echo "=== [$(date)] Processing $((i+1))/${TOTAL} | ${INPUT} ==="

  ${PY} tools/video_to_flame_param.py "$INPUT"

  echo "=== [$(date)] Finished $((i+1))/${TOTAL} ==="
  echo
done

echo "All done."
