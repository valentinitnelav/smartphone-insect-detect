#!/bin/bash

# This shell script binds the YOLO detection txt files with predictions into a single txt file. 
#
# The model directory contains subdirectories with the naming convention 'conf_..._iou...'. 
# Each subdirectory contains a 'labels' folder with YOLO detection txt files. 
# The output txt file is named after the subdirectory ('conf_..._iou...txt') and 
# it is placed within the subdirectory.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Usage:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# cd ~/smartphone-insect-detect/code/
# sbatch [OPTIONS] bind_yolo_txt_files.sh <dir>
#
# Options:
# --partition=clara : Specifies the partition.
#                     See also https://www.sc.uni-leipzig.de/05_Instructions/Slurm/#slurm-partitions
# --mem=1G          : Node's RAM allocated for the job
# --time=<time>     : Job time limit in 'd-hh:mm:ss' (2-00:00:00 = 2 days) or 
#                     'hh:mm:ss' format (10:00:00 = 10 hours, 00:30:00 = 30 min).
# <dir>             : Main model directory containing subdirectories with YOLO detection txt files.

# Examples:
# cd ~/smartphone-insect-detect/code/
# dir=~/smartphone-insect-detect/detectors/runs/loop_conf_iou/yolov5s
# sbatch --partition=clara --mem=1G --time=0-01:00:00 bind_yolo_txt_files.sh $dir


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Fixed SLURM job options
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# The requested compute resources with hardware constraints and requested time 
# will be passed at execution time to the sbatch command shown above.
# Below are the fixed options irrespective of the requested resources.

#SBATCH --job-name=bind_yolo_txt_files
#SBATCH --cpus-per-task=1 # request number of CPUs;
#SBATCH --output=/home/sc.uni-leipzig.de/%u/smartphone-insect-detect/detectors/logs/bind_yolo_txt_files_%j.log # path for job-id.log file;
#SBATCH --error=/home/sc.uni-leipzig.de/%u/smartphone-insect-detect/detectors/logs/bind_yolo_txt_files_%j.err # path for job-id.err file;
#SBATCH --mail-type=BEGIN,TIME_LIMIT,END # email options;


# Main directory input.
# e.g. smartphone-insect-detect/detectors/runs/loop_conf_iou/yolov5s
MAIN_DIR=$1 # positional argument 1

# Measure total execution time of the shell script
start_time=$(date +%s)

# Loop through each 'conf_._iou_.' subdirectory in the main directory
for subdir in "$MAIN_DIR"/*/; do

    echo "Processing subdirectory: $subdir"
    # Measure subdirectory time
    subdir_start_time=$(date +%s)

    if [[ -d "$subdir" ]]; then
        # Directory containing YOLO detection files (inside 'labels' folder)
        DETECTION_DIR="${subdir}labels"

        # Output file within the subdirectory
        OUTPUT_FILE="${subdir}$(basename "$subdir").txt"

        # Clearing the output file
        > "$OUTPUT_FILE"

        # Open file descriptor for output file
        exec 3>"$OUTPUT_FILE"

        # Processing each YOLO detection file
        find "$DETECTION_DIR" -name '*.txt' | while read -r file; do
            # Check if the file is empty
            if [ ! -s "$file" ]; then
                echo "Empty file detected: $file"
                continue
            fi

            # Extract filename without path and extension
            filename=$(basename "$file" .txt)

            # Read each line in the file and append to the output file
            while read -r line; do
                echo "${filename} ${line}" >&3
            done < "$file"
        done

        # Close file descriptor
        exec 3>&-
    fi
    # Compute subdirectory processing time
    subdir_end_time=$(date +%s)
    subdir_duration=$((subdir_end_time - subdir_start_time))
    echo "Subdirectory time elapsed: ${subdir_duration} seconds."
    printf '\n'
done

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Total time elapsed: ${duration} seconds."