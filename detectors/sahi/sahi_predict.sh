#!/bin/bash

# Bash script to run SAHI predict

# USAGE:
# bash sahi_predict.sh <prj_path> <dataset_json_path> <slice> <conf> <overlap> <nms_iou>

# Example:
# prj_path=/scratch/$USER/smartphone-insect-detect/ # path to this repository
# cd "${prj_path}"/detectors/sahi/
# json_path="${prj_path}"/detectors/sahi/sahi_predictions/ground_truth_without_tp_for_sahi.json
# bash sahi_predict.sh "${prj_path}" "${json_path}" 640 0.001 0.2 0


prj_path=$1
dataset_json_path=$2
slice=$3   # passed to slice_height & slice_width
conf=$4    # passed to model_confidence_threshold
overlap=$5 # passed to overlap_height_ratio & overlap_width_ratio
nms_iou=$6 # passed to postprocess_match_threshold


img_dir="${prj_path}"/data/images/cropped/
yolo_weights="${prj_path}"/weights/yolov5_s_best.pt
config_path="${prj_path}"/detectors/sahi/class_agnostic.yaml
output_dir="${prj_path}"/detectors/sahi/sahi_predictions/
output_dir_json="pred_slice_${slice}_conf_${conf}_overlap_${overlap}_iou_${nms_iou}"

# Redirect standard output to txt files:
log_path="${prj_path}"/detectors/sahi/logs/log_sahi_$(date +"%Y-%m-%d_%H-%M-%S")
echo "Log file: ${log_path}.log"
exec 1>"${log_path}".log

echo "prj_path: ${prj_path}"
echo "dataset_json_path: ${dataset_json_path}"
echo "img_dir: ${img_dir}"
echo "yolo_weights: ${yolo_weights}"
echo "config_path: ${config_path}"
echo "Root project dir: ${output_dir}"
echo "JSON results folder: ${output_dir_json}"
echo "slice: ${slice}"
echo "conf: ${conf}"
echo "overlap: ${overlap}"
echo "nms_iou: ${nms_iou}"

# Activate virtual environment
source "${prj_path}"/envs/sahi/bin/activate

cd "${prj_path}"/detectors/sahi/

start_time=$(date +%s)

sahi predict \
--source "${img_dir}" \
--dataset_json_path "${dataset_json_path}" \
--model_type 'yolov5' \
--model_path "${yolo_weights}" \
--model_config_path "${config_path}" \
--model_device 'cuda' \
--model_confidence_threshold ${conf} \
--slice_height ${slice} \
--slice_width ${slice} \
--overlap_height_ratio ${overlap} \
--overlap_width_ratio ${overlap} \
--postprocess_match_threshold ${nms_iou} \
--postprocess_class_agnostic \
--postprocess_type 'GREEDYNMM' \
--postprocess_match_metric 'IOU' \
--project "${output_dir}" \
--name "${output_dir_json}" \
--novisual

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Time taken: ${duration} seconds ($((duration / 60)) minutes)."

# Deactivate virtual environment
deactivate