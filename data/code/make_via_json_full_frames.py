"""
Overview:

Script to generate a VIA (VGG Image Annotator) compatible JSON file for insect 
annotations within the full frame dataset, together with the regions of interest (ROIs).

This script reads in the raw annotation data (for the full frame images), 
constructs image paths, organizes metadata, and generates a JSON file 
that can be visualized in the VIA tool. The output JSON file includes both insect 
bounding boxes and ROI boxes.

Usage:
1. Activate the corresponding environment:
   $ source ./envs/general/bin/activate

2. Run the script from the root folder of the project:
   $ python3 ./data/code/make_via_json_full_frames.py

Inputs:
- Feather file containing raw annotation data:
  './data/annotations/annotations_full_frames.feather'

Outputs:
- VIA-compatible JSON file:
  './data/annotations/annotations_full_frames_for_vggvia.json'

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


# Read the raw annotation data and construct full image paths
file_path = os.path.join(prj_path, 'data', 'annotations', 'annotations_full_frames.feather')
df_raw = pd.read_feather(file_path)

# Construct image paths and add them to the dataframe
full_frame_dir_path = os.path.join(prj_path, 'data', 'images', 'raw')
df_raw['path'] = df_raw['filename_full_frame'].apply(lambda x: os.path.join(full_frame_dir_path, x))

# Prepare the data for VIA. This requires a row with coordinates for each bounding 
# box in the json file.
# Therefore, we need to split the data frame into two data frames,
# one for the insects and one for the ROIs.
# Then concatenate the two data frames which can be further used for the VIA JSON file.
cols_roi = ['path', 'x_roi', 'y_roi', 'width_roi', 'height_roi']
# Not all fields below are needed for the VIA JSON file, but the non-coordinate fields 
# are useful metadata.
cols_insect = ['path', 'id_raw', 'id_box', 'seq_id', 'keep_seq', 
               'order', 'family', 'clustergenera', 'genus', 'morphospecies', 'species',
               'x', 'y', 'width', 'height']

# Create a data frame for the ROIs. Make sure to have only unique path.
df_roi = df_raw[cols_roi].copy()
# Remove duplicates based on 'path' to ensure uniqueness
df_roi = df_roi.drop_duplicates(subset='path')
# Add a new column 'roi' and set all its values to True
df_roi['roi'] = True

# Create a data frame for the insects. No need of unique path here.
df_insect = df_raw[cols_insect].copy()
# Add a new column 'roi' and set all its values to False
df_insect['roi'] = False

# Rename the columns so that they are the same in both data frames
df_roi.rename(columns={'x_roi': 'x', 
                       'y_roi': 'y', 
                       'width_roi': 'width', 
                       'height_roi': 'height'}, 
              inplace=True)

# Concatenate the two data frames. Concatenating along rows (axis=0).
# Columns that are not common between the two data frames will result in missing 
# values (NaN) in those places where a data frame doesn't have the corresponding column.
df_via = pd.concat([df_roi, df_insect], ignore_index=True, axis=0)

# Sort values so that the annotation row for an insect box 
# appears before the one for the ROI box in the VIA GUI.
# Then sort by any other metadata field of desire.
df_via = df_via.sort_values(by=['path', 'order', 'family', 'seq_id', 'id_box'])

# These are the columns that will be used as attributes in the VIA JSON file
cols_via_attributes = ['roi', 'seq_id', 'keep_seq', 'id_raw', 'id_box',
                       'order', 'family', 'clustergenera', 'genus', 'morphospecies', 'species']

# Generate the VIA-compatible JSON file and save it to the specified path
via_json = df_to_via_json(df_via,
                          project_name='annotations_with_roi_all_raw', # stored within the JSON file
                          attribute_cols=cols_via_attributes)

file_path = os.path.join(prj_path, 'data', 'annotations', 'annotations_full_frames_for_vggvia.json')
with open(file_path, 'w') as f:
    simplejson.dump(via_json, f, ignore_nan=True, indent=4)

print(f"VIA JSON file saved to:\n'{file_path}'")
# This JSON file now can be visualized in the VIA tool 
# See https://www.robots.ox.ac.uk/~vgg/software/via/

# Print time taken in seconds and minutes
end_time = time.time()
total_time = end_time - start_time
print(f"Time taken: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")