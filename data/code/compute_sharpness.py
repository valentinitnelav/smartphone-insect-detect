"""
Overview:

Compute sharpness/blurriness of pixels within insect boxes in cropped images.

Metric: Tenengrad (Sobel's operator).

Quantifies edge presence, which is reduced in blurred images. Higher values
indicate sharper images, lower values indicate blurrier images. This metrics
measure high-frequency content, not direct focus.

References: - Pertuz, S., Puig, D., & Garcia, M. A. (2013). 
    Analysis of focus measure operators for shape-from-focus. Pattern
    Recognition, 46(5), 1415-1432.
- Blur detection with OpenCV by Adrian Rosebrock, 2015
    https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
- How to identify blurry images at rbaron.net, 2020  
    https://rbaron.net/blog/2020/02/16/How-to-identify-blurry-images.html
- Calculating sharpness of an image at Stack Overflow 
    https://stackoverflow.com/questions/28717054/calculating-sharpness-of-an-image

Usage: 1. Activate the corresponding environment:
   $ source ./envs/general/bin/activate

2. Run the script from the root folder of the project: 
   $ python3 ./data/code/compute_sharpness.py

Inputs: - Feather file containing the OOD annotation data frame:
  './data/processed/df_roi.feather'

Outputs: - Updated OOD annotation data frame with sharpness metrics:
  './data/processed/df_roi.feather'
"""

import os
import time
import pandas as pd
import numpy as np
import cv2
import git
import multiprocessing as mp

# Get the root path of the project using Git and print it for verification
prj_path = git.Repo('.', search_parent_directories=True).working_tree_dir
print(f"Root path of the project:\n'{prj_path}'")


# Start the timer
start_time = time.time()
print("Processing...")

# Read the data frame with the annotation information for each insect box
# prepared for analysis.
# This is the ground truth data = the OOD annotation dataset.
file_path = os.path.join(prj_path, 'data', 'processed', 'df_roi.feather')
df = pd.read_feather(file_path)

# Check if column 'score_sobel' already exist.
# If it exists, then give a warning and exit. This is to avoid overwriting the
# data in case this script is run multiple times.
if 'score_sobel' in df.columns:
    print("Warning: Column 'score_sobel' already exist in the data frame.")
    print("Exiting...")
    exit()
# If they do not exists, proceed with the computation.
print(f"Computing sharpness metrics for: {df.shape[0]} insect boxes...")

# Generate the paths to the cropped images
cropped_dir = os.path.join(prj_path, 'data', 'images', 'cropped')
df['path'] = df['new_filename'].apply(lambda x: os.path.join(cropped_dir, x))

# Helper function:
def calculate_sharpness_score(row):
    id_raw = row['id_raw']
    try:
        # Load the image
        image = cv2.imread(row['path'])
        # Convert the coordinates and dimensions to integers
        x = round(row['x'])
        y = round(row['y'])
        width = round(row['width'])
        height = round(row['height'])
        # Extract the region of the image inside the bounding box.
        # Note that OpenCV treats image coordinates in (y, x) order and not in
        # (x, y) order.
        region = image[y:y+height, x:x+width]
        # Convert the region to grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        # Compute the Sobel gradient of the region
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        score_sobel = np.sum(sobelx**2.0 + sobely**2.0)
    except Exception as e:
        # Log the error if needed
        print(f"Error processing image {id_raw}: {e}")
        # Return NaN if there is an error
        score_sobel = float('nan')
    
    # Returns a tuple with the id_raw and the scores
    return id_raw, score_sobel


# Apply the function to the data frame in parallel:

# Prepare a list of rows
list_temp = [row for _, row in df.iterrows()]

# Create a pool of processes. 
# Determine the number of workers to use (all available minus one)
n_workers = max(1, mp.cpu_count() - 1)  # Ensure at least one CPU is used
with mp.Pool(n_workers) as pool:
    scores = pool.map(func=calculate_sharpness_score, 
                      iterable=list_temp)

# Convert scores to a DataFrame and assign column names
scores_df = pd.DataFrame(scores, columns=['id_raw', 'score_sobel'])

# Merge with df
df = pd.merge(df, scores_df, on='id_raw', how='inner')

# Calculate also the bounding box relative area
df['box_area_rel'] = (df['width'] * df['height']) / (df['width_crop'] * df['height_crop'])


# Save the data frame with the sharpness metrics
# - as feather file
file_path = os.path.join(prj_path, 'data', 'processed', 'df_roi.feather')
df.to_feather(file_path)
print(f"Updated OOD annotation data frame saved as Feather file at:\n'{file_path}'")
# - as txt file
file_path_txt = os.path.join(prj_path, 'data', 'processed', 'df_roi.txt')
df.to_csv(file_path_txt, sep='\t', index=False, quoting=3)
print(f"Updated OOD annotation data frame saved as tab separated txt file at:\n'{file_path}'")


end_time = time.time()
total_time = end_time - start_time
print(f"Time taken: {total_time:.2f} seconds ({total_time/60:.2f} minutes) on {n_workers} CPUs.")
