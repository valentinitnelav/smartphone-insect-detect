#!/bin/bash

# This script activates a Python environment and runs the YOLOv5 detection model
# on background images so that we can estimate FPs on such images without
# insects.

# Usage:
# prj_path=/scratch/$USER/smartphone-insect-detect/ # path to this repository
# cd "${prj_path}"/code/
# bash detect_on_backgrounds.sh "${prj_path}"


PRJ_PATH=$1
OUTPUT_FOLDER="${PRJ_PATH}/detectors/predictions/yolov5s/backgrounds/"

# Redirect standard output to txt files:
LOG_PATH="${PRJ_PATH}"/detectors/logs/log_detect_on_backgrounds_$(date +"%Y-%m-%d_%H-%M-%S")
echo "Log file: ${LOG_PATH}.log"
exec 1>"${LOG_PATH}".log

echo "Project path: ${PRJ_PATH}"
echo "Detections will be saved at: ${OUTPUT_FOLDER}"

START_TIME=$(date +%s)


# Activate the needed environment
source "${PRJ_PATH}/envs/yolov5/bin/activate"

# Call the helper script session_info.sh which will print in the *.log file info
# about the used environment and hardware.
source ${PRJ_PATH}/code/session_info.sh

# Run detections using the optimal NMS values
python3 "${PRJ_PATH}"/detectors/yolov5/detect.py \
--weights "${PRJ_PATH}"/detectors/weights/yolov5_s_best.pt \
--source "${PRJ_PATH}"/data/images/backgrounds/cropped_bg/ \
--img-size 640 \
--conf-thres 0.201918 \
--iou-thres 0.3 \
--save-txt \
--save-csv \
--save-conf \
--nosave \
--project "${OUTPUT_FOLDER}" \
--name bg_detections

deactivate # deactivate the environment


END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
echo "Time taken: ${TOTAL_TIME} seconds ($((TOTAL_TIME / 60)) minutes)."