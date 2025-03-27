"""
Overview:
Script to generate a VIA (VGG Image Annotator) compatible JSON file for insect
boxes within the cropped images.

Usage:
1. Activate the corresponding environment:
   $ source ./envs/general/bin/activate

2. Run the script from the root folder of the project:
   $ python3 ./data/code/make_via_json_cropped_frames.py

Inputs:
- Feather file containing raw annotation data:
  './data/processed/df_roi.feather'

Outputs:
- VIA-compatible JSON file:
  './data/processed/annotations_cropped_frames_for_vggvia.json'

Notes:
- The generated JSON file can be visualized using the VIA tool:
  https://www.robots.ox.ac.uk/~vgg/software/via/
"""

import os
import sys
import time
import pandas as pd
import simplejson
import git

# Get the root path of the project using Git and print it for verification
prj_path = git.Repo('.', search_parent_directories=True).working_tree_dir
print(f"Root path of the project:\n'{prj_path}'")

# Add the path to the directory containing utils.py to sys.path
sys.path.append(f"{prj_path}/code")

# Import needed function from utils.py
from utils import df_to_via_json


# Start the timer
start_time = time.time()
print("Processing...")


# Read the data frame with the annotation information for each insect box prepared for analysis.
# This is the ground truth data = the OOD annotation dataset.
file_path = os.path.join(prj_path, 'data', 'processed', 'df_roi.feather')
df = pd.read_feather(file_path)

# Generate the paths to the cropped images
cropped_dir = os.path.join(prj_path, 'data', 'images', 'cropped')
df['path'] = df['new_filename'].apply(lambda x: os.path.join(cropped_dir, x))

# Sort values by path so that they appear sorted in the final JSON file
df = df.sort_values(by=['path'])

# Prepare the data for VIA
via_json = df_to_via_json(df,
                          project_name='OOD_cropped_frames',
                          attribute_cols=['p1_labels', 'id_raw', 'seq_id']) 
# in attribute_cols include the metadata columns of your desire that exist in df

# Save the JSON file
file_path = os.path.join(prj_path, 'data', 'annotations', 'annotations_ood_cropped_frames_for_vggvia.json')
with open(file_path, 'w') as f:
    simplejson.dump(via_json, f, ignore_nan=True, indent=4)

print(f"VIA JSON file saved to:\n'{file_path}'")
# This JSON file now can be visualized in the VIA tool 
# See https://www.robots.ox.ac.uk/~vgg/software/via/

# Print time taken in seconds and minutes
end_time = time.time()
total_time = end_time - start_time
print(f"Time taken: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")