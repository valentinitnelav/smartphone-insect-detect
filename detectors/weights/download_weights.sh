#!/bin/bash

# Shell script to download the YOLOv5 and v7 weights from our previous study.
# https://github.com/stark-t/PAI/tree/main/detectors/trained_weights

# USAGE:
# Directly run the script in its current directory to download weights there:
# cd /scratch/$USER/smartphone-insect-detect/detectors/weights/ # path to this script
# bash download_weights.sh

# URLs of the files you want to download
urls=(
    "https://github.com/stark-t/PAI/raw/main/detectors/trained_weights/yolov5_n_best.pt"
    "https://github.com/stark-t/PAI/raw/main/detectors/trained_weights/yolov5_s_best.pt"
    "https://github.com/stark-t/PAI/raw/main/detectors/trained_weights/yolov7_tiny_best.pt"
)
# Loop through the URLs and download each file
for url in "${urls[@]}"; do
    # Extract the filename from the URL
    filename=$(basename "$url")
    # Download the file
    wget "$url" -O "$filename"
    # Check if the download was successful
    if [ $? -eq 0 ]; then
        echo "File '$filename' downloaded successfully."
    else
        echo "Failed to download file '$filename'."
    fi
done
