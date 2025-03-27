"""
Overview:
Script to crop the full frame images to the cropping dimensions defined by the union of the ROI
and the visiting insect bounding box.

Usage:
1. Activate the corresponding environment:
   $ source ./envs/general/bin/activate

2. Run the script from the root folder of the project:
   $ python3 ./data/code/crop_full_frames.py

Inputs:
- Feather file containing data for cropping
  './data/processed/df_crops_interim.feather'
  
Outputs:
- Cropped images stored in ./data/images/cropped/
"""

import os
import sys
import time
import multiprocessing as mp
import pandas as pd
import git

# Get the root path of the project using Git and print it for verification
prj_path = git.Repo('.', search_parent_directories=True).working_tree_dir
print(f"Root path of the project:\n'{prj_path}'")

# Add the path to the directory containing utils.py to sys.path
sys.path.append(f"{prj_path}/code")

# Import needed function from utils.py
from utils import crop_image

# Start the timer
start_time = time.time()
print("Processing...")


# Read the data frame with cropping information. It contains the paths to the full frame images
# and also the cropping dimensions of the cropping box (union of ROI and insect bounding box)
# for each image.
file_path = os.path.join(prj_path, 'data', 'processed', 'df_crops_interim.feather')
df_crops = pd.read_feather(file_path)
# df_crops = df_crops.head(5) # for testing purposes only

new_path = os.path.join(prj_path, 'data', 'images', 'cropped')
if not os.path.exists(new_path):
    os.makedirs(new_path)


# Multiprocessing

def crop_image_parallel(row):
    try:
        crop_image(row, dir_path=new_path)
    except Exception as e:
        # Handle any exceptions and store the error paths for later analysis
        error_paths.append(row['path'])
        print(f"Error processing {row['path']}: {e}")

n_workers = max(1, mp.cpu_count() - 1)  # Ensure at least one CPU is used
print(f"Using {n_workers} workers")

# Initialize an empty list in the global scope.
# See crop_image() function in ultils.py
error_paths = []

with mp.Pool(n_workers) as pool:
    pool.map(crop_image_parallel, [row for _, row in df_crops.iterrows()])     


print(f"Number of errors: {len(error_paths)}") # expect 0 errors

# Count the number of jpg files in the new_path directory.
# This must match the number of rows in df_crops
n_files = len([name for name in os.listdir(new_path) if name.endswith('.jpg')])
n_rows = df_crops.shape[0]
print(f"Number of files in new_path: {n_files} must match the number of rows in df_crops: {n_rows}")
# Expect: Number of files in new_path: 23899 must match the number of rows in df_crops: 23899

end_time = time.time()
total_time = end_time - start_time
print(f"Time taken: {total_time:.2f} seconds ({total_time/60:.2f} minutes) on {n_workers} CPUs.")
# Time taken: 50.79 seconds (0.85 minutes) on 15 CPUs.