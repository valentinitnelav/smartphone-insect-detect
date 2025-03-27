#!/bin/bash

# Shell script for running detection on the given YOLO model (v5 or v7) with 
# the specified IoU and Conf values. 
#
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Usage:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# sbatch [OPTIONS] detect.sh --model <model> --iou <iou(s)> --conf <conf(s)> --output <path>
#
# Options:
#
# --partition=clara    : Specifies the partition. Maximum of 2 days for 'clara' and 'paula'.
#                          Use 'clara-long' for jobs up to 10 days.
#                          See also https://www.sc.uni-leipzig.de/05_Instructions/Slurm/#slurm-partitions
# --gres=gpu:<model>:1 : GPU model and count. Available models (with dedicated RAM / VRAM): 
#                          a30 (24G VRAM), v100 (32G VRAM), rtx2080ti (11G VRAM).
# --mem=4G             : Node's RAM allocated for the job, shared between CPU and GPU.
# --time=<time>        : Job time limit in 'd-hh:mm:ss' (2-00:00:00 = 2 days) or 
#                          'hh:mm:ss' format (10:00:00 = 10 hours, 00:30:00 = 30 min).
#                          For time limits see comments at --partition above.
# --model <model>      : YOLO model. Expected values: yolov5n, yolov5s, yolov7-tiny.
# --conf <conf(s)>     : NMS conf threshold(s). Can be a single numeric value or a range (start:step:end).
# --iou <iou(s)>       : NMS IoU threshold(s). Can be a single numeric value or a range (start:step:end).
# --proj <path>        : Path to the project folder.
# --output <path>      : Path to the output folder. 

# Examples:
#
# proj=~/smartphone-insect-detect/
# output="${proj}/detectors/runs/detections_conf_0.001/"
# cd "${proj}/code/"
# sbatch --partition=clara --gres=gpu:rtx2080ti:1 --mem=4G --time=0-10:00:00 detect.sh --model yolov5s --conf 0.001 --iou 0.1:0.1:0.9 --proj "${proj}" --output "${output}"
# sbatch --partition=paula --gres=gpu:a30:1 --mem=4G --time=0-10:00:00 detect.sh --model yolov5s --conf 0.001 --iou 0.1:0.1:0.9 --proj "${proj}" --output "${output}"
# sbatch --partition=paula --gres=gpu:a30:1 --mem=4G --time=0-10:00:00 detect.sh --model yolov5s --conf 0.2 --iou 0.5 --proj "${proj}" --output "${output}"
#
# For testing purposes, comment out the detect.py call below and run the script with the following commands:
# bash detect.sh --model yolov5s --conf --iou 0.1:0.1:0.9 0.1:0.1:0.9 --proj "${proj}" --output "${output}"
# bash detect.sh --model yolov5s --conf 0.001 --iou 0.1:0.1:0.9 --proj "${proj}" --output "${output}"
# bash detect.sh --model yolov5s --conf 0.001 --iou 0.5 --proj "${proj}" --output "${output}"

# Notes:
#
# For more information on SLURM options, visit: https://www.sc.uni-leipzig.de/05_Instructions/Slurm/
# Check also `man sbatch` for more information.
#
# Note that if you instead of --mem option you use --mem-per-gpu, that it is not the GPU memory 
# to be requested (GPU RAM / VRAM is fixed/dedicated as mentioned above). 
# So it's the RAM you allocate per GPU request. The internal/dedicated GPU-RAM (VRAM) cannot be allocated.
# So, if you set --mem-per-gpu=11G and request 1 GPU, you request 11 GB of node's RAM. 
# If you would request two GPUs, you would allocate 22 GB of node's RAM.
# Whenever you allocate a GPU, you automatically allocate it's full dedicated memory:
# gpu:a30 (24G VRAM), gpu:v100 (32G VRAM), gpu:rtx2080ti (11G VRAM)
#
# To get available GPU info:
# sinfo -t idle,mixed --format="%30N %20G %10D %10m" | grep -v 'null'

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Fixed SLURM job options
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# The requested GPU resources with hardware constraints and requested time 
# will be passed at execution time to the sbatch command shown above.
# Below are the fixed options irrespective of the requested resources.
# The lines below will be evaluated by the job scheduler. These are not comments:

#SBATCH --job-name=test_ood
#SBATCH --cpus-per-task=2 # request number of CPUs; 2 CPUs are enough for detection (possibly even 1 would be enough)
#SBATCH --output=/home/sc.uni-leipzig.de/%u/smartphone-insect-detect/detectors/logs/log_detect_%j.log # path for job-id.log file;
#SBATCH --error=/home/sc.uni-leipzig.de/%u/smartphone-insect-detect/detectors/logs/log_detect_%j.err # path for job-id.err file;
#SBATCH --mail-type=BEGIN,TIME_LIMIT,END # email options;


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Process arguments & helper functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Measure total execution time
start_time=$(date +%s)

# Initialize variables
model=""
conf_str=""
iou_str=""
confs=()
ious=()
proj=""
output=""

# Manual parsing of arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) model="$2"; shift ;;
        --conf) conf_str="$2"; shift ;;
        --iou) iou_str="$2"; shift ;;
        --proj) proj_path="$2"; shift ;;
        --output) output_path="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check if required variables are set ----------------------------------------

# Declare an associative array with variable names and their human-readable descriptions
declare -A required_vars=(
    [model]="YOLO model"
    [conf_str]="NMS Conf value(s)"
    [iou_str]="NMS IoU value(s)"
    [proj_path]="Project folder path"
    [output_path]="Output folder path"
)

# Loop through the associative array to check each variable
for var_name in "${!required_vars[@]}"; do
    if [[ -z "${!var_name}" ]]; then  # Indirect expansion to get the value of the variable named by var_name
        echo "${required_vars[$var_name]} is required."
        exit 1
    fi
done

# Parse ranges into array ----------------------------------------------------

# Function to parse range into array
# Usage: parse_range "0.1:0.1:0.9" (start:step:end)
# The LC_ALL=C is to go around the fact that in some locales, the decimal 
# separator is a comma "," instead of a period "."
# So, the C locale ensures that numbers are formatted using a period (.) 
# as the decimal separator. 
parse_range() {
    IFS=':' read -r start step end <<< "$1"
    echo $(LC_ALL=C seq "$start" "$step" "$end")
}

# Process Conf and IoU parameters.
if [[ "$conf_str" == *":"* ]]; then
    # If conf_str is a range, generate the array
    readarray -t confs <<< "$(parse_range "$conf_str")"
else
    # If not a range, treat as a single value array
    confs=("$conf_str")
fi

if [[ "$iou_str" == *":"* ]]; then
    # If iou_str is a range, generate the array
    readarray -t ious <<< "$(parse_range "$iou_str")"
else
    # If not a range, treat as a single value array
    ious=("$iou_str")
fi

# Select detector -----------------------------------------------------------

# Select the detector and weights file name based on the model name
# The env names are assumed to be as the detector names and must be the ones in 
# ~/smartphone-insect-detect/envs/
case $model in
    yolov5n)
        weights_file="yolov5_n_best.pt"
        detector="yolov5"
        ;;
    yolov5s)
        weights_file="yolov5_s_best.pt"
        detector="yolov5"
        ;;
    yolov7-tiny)
        weights_file="yolov7_tiny_best.pt"
        detector="yolov7"
        ;;
    *)
        echo "Invalid model name specified"
        exit 1
        ;;
esac


# The YOLO output project folder might need to be created depending on the model.
# Check if directory exists and create if not
if [ ! -d "$output_path" ]; then
    mkdir -p "$output_path"
    echo "Output folder created at: $output_path"
else
    echo "Output folder already exists at: $output_path"
fi

echo "Model: $model"
echo "Confs: ${confs[*]}"
echo "IoUs: ${ious[*]}"
echo "Output folder: $output_path"


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Environment setup & activation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Load the module used for creating the environment.
# Start with a clean environment, then load only the needed modules
module purge 
module load Python/3.10.8-GCCcore-12.2.0

# Activate virtual environment
source ${proj_path}/envs/${detector}/bin/activate

# Ensure that the virtual environment packages are preferred over the packages 
# that are installed in the module loaded above. Make sure that 
# env/.../lib/python3.10/ exists (that is, the correct python version).
printf '\n'
export PYTHONPATH=${proj_path}/envs/${detector}/lib/python3.10/site-packages:$PYTHONPATH
echo "PYTHONPATH: $PYTHONPATH"
printf '\n'

# Call the helper script session_info.sh which will print in the *.log file info 
# about the used environment and hardware.
source ${proj_path}/code/session_info.sh

cd ${proj_path}


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Detection
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# In the output folder, create `--project`` subfolder for each given model and then `--name`
# sub-folders for each combination of IoU and Conf containing the YOLO txt files with predictions.

for iou in ${ious[@]}; do
    for conf in ${confs[@]}; do
        echo "Running detection for Model: $model, IoU: $iou, Conf: $conf"

        # For testing purposes, use the following data path with few images
        # --source "${proj_path}"/data/images/cropped_test_sample/
        
        # Start loop time
        loop_start_time=$(date +%s)
        
        python3 "${proj_path}"/detectors/${detector}/detect.py \
        --weights "${proj_path}"/detectors/weights/${weights_file} \
        --source "${proj_path}"/data/images/cropped/ \
        --img-size 640 \
        --conf-thres ${conf} \
        --iou-thres ${iou} \
        --save-txt \
        --save-conf \
        --nosave \
        --project "${output_path}"/${model} \
        --name conf_${conf}_iou_${iou}

        # Measure loop time
        loop_end_time=$(date +%s)
        loop_duration=$((loop_end_time - loop_start_time))
        echo "Loop time elapsed for conf ${conf} and IoU ${iou}: ${loop_duration} seconds."
        printf '\n'
    done
done

# Note that `--max-det 1000` is the current default in detect.py for yolov5. 
# At the moment, only YOLOv5 has this argument, but not in YOLOv7
# I assume that since we have at most few insects per image, having a big max-det is not necessary.
# See https://github.com/ultralytics/yolov5/blob/master/detect.py
# and https://github.com/WongKinYiu/yolov7/blob/main/detect.py

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Time taken: ${duration} seconds ($((duration / 60)) minutes)."

# Deactivate virtual environment for the given model
deactivate