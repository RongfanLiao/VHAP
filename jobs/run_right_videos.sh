#!/bin/bash

DATA_FOLDER="data/video_pairs"
START_TIME=$(date +%s)

format_duration() {
    local total_seconds=$1
    local hours=$((total_seconds / 3600))
    local minutes=$(((total_seconds % 3600) / 60))
    local seconds=$((total_seconds % 60))
    printf "%02d:%02d:%02d" "$hours" "$minutes" "$seconds"
}

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
    # Run the Python script for each sequence
    python preprocess_track_export.py -i "${DATA_FOLDER}/${SEQUENCE}.mp4"
done

echo ""
echo "All done."

END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))

echo "Total time consumed: $(format_duration "$ELAPSED_TIME") (${ELAPSED_TIME}s)"
