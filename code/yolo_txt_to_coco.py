"""
Converts concatenated YOLO detection text files to COCO JSON format.

This script processes one or multiple concatenated YOLO detection text files,
converting them to the COCO JSON format. The script reads each text file,
calculates absolute bounding box coordinates from relative ones, and formats the data to match the COCO
JSON structure. The output JSON files are saved in the same directories as the
source concatenated YOLO text files.

Usage:
1. Activate the corresponding environment:
   $ source ./envs/general/bin/activate
2. Run the script from the root folder of the project:   
   $ python3 ./code/yolo_txt_to_coco.py --dir <directory_path> --cores <number_of_cores>

Parameters:
    --dir:     Path to the root directory containing the concatenated YOLO prediction text files. 
               These files include all predictions for all images in a single text file, 
               rather than separate files for each image.
    --gt_path: Path to the ground truth JSON file.
    --cores:   Number of CPU cores to use (default: 4).
"""


import os
import sys
import pandas as pd
import json
import time
import argparse
import traceback
import logging
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from pycocotools.coco import COCO
import git
import logging

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get the path to the project folder
prj_path = git.Repo('.', search_parent_directories=True).working_tree_dir
print(f"Root path of the project:\n'{prj_path}'")


start_time = time.time()
print("Processing...")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Parse arguments
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Initialize parser
parser = argparse.ArgumentParser(description='Process input.')

# Adding first argument for nms_range
parser.add_argument('--dir', type=str,
                    help='path to main directory with YOLO txt files for each model')

# Adding second argument for ground truth path
parser.add_argument('--gt_path', type=str,
                    help='Path to the ground truth JSON file')

# Adding third argument for number of CPU cores
parser.add_argument('--cores', type=int, default=4,
                    help='Number of CPU cores to use (default: 4)')

# Parsing arguments
args = parser.parse_args()

# Validate the directory path
if not os.path.exists(args.dir):
    logging.error(f"Directory path does not exist: {args.dir}")
    sys.exit(1)

# Validate the ground truth path
if not os.path.exists(args.gt_path):
    logging.error(f"JSON ground truth file path does not exist: {args.gt_path}")
    sys.exit(1)
    
# Check if the ground truth file is a JSON file
_, file_extension = os.path.splitext(args.gt_path)
if file_extension != '.json':
    logging.error(f"Not a JSON file: {args.gt_path}")
    sys.exit(1)

# Check the number of CPU cores
available_cpus = os.cpu_count()
requested_cpus = args.cores

if requested_cpus > available_cpus:
    logging.error(f"Requested {requested_cpus} CPUs, but only {available_cpus} are available.")
    sys.exit(1)
n_cpus = requested_cpus


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Check if all subdirectories contain .txt files
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print(f"Checking if all subdirectories contain .txt files in given root dir {args.dir}")

yolo_file_paths = []
subdirs_without_txt = []

for root, dirs, files in os.walk(args.dir):
    # Check if the directory is a leaf directory (no subdirectories)
    if not dirs:
        txt_files_found = any(f.endswith('.txt') for f in files)
        if txt_files_found:
            # Add all .txt files in this leaf directory to the list
            yolo_file_paths.extend(os.path.join(root, f) for f in files if f.endswith('.txt'))
        # If no .txt files were found in the current directory, add it to the list
        else:
            # Leaf directory with no .txt files
            subdirs_without_txt.append(root)
# `root` refers to the current directory being examined
# If dirs is empty (not dirs is True), it means that root is a leaf directory, 
# as there are no further subdirectories within it. If dirs is not empty, 
# then root is a parent directory, having one or more subdirectories.


print("Total .txt files found:", len(yolo_file_paths))
if subdirs_without_txt:
    logging.error("Error: Subdirectories without .txt files:")
    for subdir in subdirs_without_txt:
        logging.info(subdir)
    sys.exit(1)  # Exit the script with an error status
else:
    logging.info("All subdirectories have .txt files.")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Read ground truth data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Read ground truth COCO JSON file to extract and pass the exact image ids to 
# the prediction COCO json file that we will create. This assures that the 
# image ids are the same in both files and that pycocotools does the correct 
# matching between predictions and ground truth for each image.
# We also need the image dimensions of the cropped images so that we can convert 
# the YOLO coordinates relative format into COCO format (absolute pixel values).

# Read the ground truth json file to extract the image details.
gt_json_file = args.gt_path

with open(gt_json_file, 'r') as f:
    df_true_coco = json.load(f)

df_img = pd.DataFrame(df_true_coco['images'])

# Rename 'width', 'height' to 'width_crop', 'height_crop' for clarity (they are the cropped dimensions)
# Rename 'id' to 'image_id' for clarity
df_img.rename(columns={'id': 'image_id',
                       'width': 'width_crop',
                       'height': 'height_crop'}, 
              inplace=True)
# Drop the 'license' & 'date_captured' columns
df_img.drop(columns=['license', 'date_captured'], inplace=True)

# Create a new column 'file_name_no_ext' (as key) for merging purposes with the predictions data frame
df_img['file_name_no_ext'] = df_img['file_name'].apply(lambda x: os.path.splitext(x)[0])

print(f"Number of unique images in the ground truth: {df_img.shape[0]}")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Helper functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def coords_relative_to_normal(row):
    """
    Function to convert box coordinates from relative to normal/absolute.
    """
    # Returns NaN for x, y, width, and height if any value is NaN
    if pd.isnull(row[['x_center_rel', 'y_center_rel', 'width_rel', 'height_rel', 
                      'width_crop', 'height_crop']]).any():
        return pd.Series([float('nan')] * 4)  
    
    img_width = int(row['width_crop'])
    img_height = int(row['height_crop'])
    
    x = int(row['x_center_rel'] * img_width - row['width_rel'] * img_width / 2)
    y = int(row['y_center_rel'] * img_height - row['height_rel'] * img_height / 2)
    width = int(row['width_rel'] * img_width)
    height = int(row['height_rel'] * img_height)
    
    return pd.Series([x, y, width, height])


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create COCO JSON file from YOLO predictions (txt files)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Function to parallelize the creation of the COCO JSON file for predictions
def process_file(file_path, df_img):
    """
    Processes a single bound/concatenated YOLO detection text file, converts the detection 
    coordinates from relative to absolute, and saves the processed data in COCO JSON format.

    This function reads the YOLO detection data from a given text file, 
    merges it with a DataFrame containing image dimensions to convert the coordinates, 
    and then formats the data into the structure expected by COCO. 
    The final JSON is saved to the same directory as the source text file but 
    with a .json extension.

    Parameters:
    - file_path (str): Path to the bound YOLO detection text file.
    - df_img (pd.DataFrame): DataFrame containing the image dimensions
                             and the filenames of the images for matching with predictions.

    The function does not return any value but writes the output directly to a file. 
    """
    
    print(f"Processing bound YOLO txt file: {file_path}")
    
    try:
        if os.path.getsize(file_path) == 0:
            print(f"Empty concatenated YOLO txt file encountered and skipped: {file_path}")
            return  # Skip further processing for this file
        
        # Read bound txt files with YOLO detections
        # Define the header - column names
        column_names = ['file_name_no_ext', 'yolo_label', 'x_center_rel', 'y_center_rel', 
                        'width_rel', 'height_rel', 'yolo_conf']
        df_pred = pd.read_csv(file_path, sep=' ', header=None, names=column_names)

        # Add the image dimensions to df_pred.
        df_pred = pd.merge(df_pred, df_img, on='file_name_no_ext', how='inner', sort=True)

        # Convert box coordinates from relative to normal as these are expected by pycocotools
        cols = ['yolo_label', 'x_center_rel', 'y_center_rel', 'width_rel', 'height_rel']
        df_pred[cols] = df_pred[cols].apply(pd.to_numeric)
        df_pred[['x', 'y', 'width', 'height']] = df_pred.apply(coords_relative_to_normal, axis=1)

        # Bring predictions in the needed form expected by pycocotools
        pred_annotations_list = df_pred.reset_index().apply(lambda row: {
            "id": int(row['index']),  # Use the resetted index as the id
            "image_id": int(row['image_id']),
            "file_name": row['file_name'], # Usually not a field in the annotation section in COCO
            "category_id": int(row['yolo_label']),
            # "category_id": 1, # or evaluate as an insect detector (single class); 
            # otherwise coco evaluation will ignore class 7, which doesn't exist in ground truth.
            # WARNING - if you do single class, do not use 0 for category_id because COCO eval
            # starts indexing from 1 for category_id
            "bbox": [row['x'], row['y'], row['width'], row['height']],
            "area": float(row['width'] * row['height']),
            "iscrowd": 0,
            "score": float(row['yolo_conf'])
        }, axis=1).tolist()

        # write json file with predictions to disk
        pred_json_file = os.path.splitext(file_path)[0] + ".json" # replace .txt with .json
        with open(pred_json_file, 'w') as f:
            json.dump(pred_annotations_list, f, indent=4)
        print(f"File saved with COCO format at: {pred_json_file}")
    
    # Catch and log any exception
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        traceback.print_exc() # print the detailed traceback


# Execute the process_file function in parallel -------------------------------

n_cpus = requested_cpus

with ThreadPoolExecutor(max_workers=n_cpus) as executor:
    # Using a list comprehension to create futures
    futures = [executor.submit(process_file, path, df_img) for path in yolo_file_paths]
    
    # Ensure all futures complete
    for future in concurrent.futures.as_completed(futures):
        try:
            # We call future.result() to catch any exceptions raised during task execution
            future.result()
        except Exception as e:
            # Log the exception
            logging.error(f"Future execution resulted in an error: {e}")
            traceback.print_exc() # print the detailed traceback

end_time = time.time()
total_time = end_time - start_time
print(f"Time taken: {total_time:.2f} seconds ({total_time/60:.2f} minutes) on {n_cpus} CPUs.")