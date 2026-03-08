#!/bin/bash

DATA_FOLDER="data"

# Find all .mp4 videos with "right" in the name
SEQUENCES=()
for f in ${DATA_FOLDER}/*right*.mp4; do
    if [ -f "$f" ]; then
        seq=$(basename "$f" .mp4)
        SEQUENCES+=("$seq")
    fi
done

echo "Found ${#SEQUENCES[@]} right videos: ${SEQUENCES[@]}"

for SEQUENCE in "${SEQUENCES[@]}"; do
    echo ""
    echo "========================================="
    echo "Processing: ${SEQUENCE}"
    echo "========================================="

    #======= Preprocess =======#
    RAW_VIDEO_PATH="${DATA_FOLDER}/${SEQUENCE}.mp4"
    PREPROCESSED_DIR="${DATA_FOLDER}/${SEQUENCE}"

    if [ ! -d "${PREPROCESSED_DIR}/images" ]; then
        echo "[Preprocess] ${SEQUENCE}"
        python vhap/preprocess_video.py \
            --input "${RAW_VIDEO_PATH}" \
            --matting_method robust_video_matting
    else
        echo "[Preprocess] Skipping ${SEQUENCE} (already preprocessed)"
    fi

    #======= Track (landmark-only, no photometric) =======#
    TRACK_OUTPUT_FOLDER="output/monocular/${SEQUENCE}_lmkOnly"

    # Check if tracking already completed (last epoch npz exists)
    last_folder=$(find "$TRACK_OUTPUT_FOLDER" -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1)
    if [ -n "$last_folder" ] && [ -e "$last_folder/tracked_flame_params_30.npz" ]; then
        echo "[Track] Skipping ${SEQUENCE} (already tracked)"
    else
        echo "[Track] ${SEQUENCE}"
        python vhap/track.py \
            --data.root_folder "${DATA_FOLDER}" \
            --data.sequence "${SEQUENCE}" \
            --exp.output_folder "${TRACK_OUTPUT_FOLDER}" \
            --exp.no_photometric
    fi
done

echo ""
echo "All done."
