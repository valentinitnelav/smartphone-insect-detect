"""
Convert ground truth data to COCO json file format. 
This is needed for using the pycocotools COCO evaluation API for computing the mAP.

Note that the extra classes like 'orthoptera' come from the pretrained weights (previous study).
This label would be ignored by pycocotools, so we will have to run the evaluation for 
insect detection by ignoring the classes in the prediction files (class-agnostic).

Usage:
1. Activate the corresponding environment:
   $ source ./envs/general/bin/activate

2. Run the script from the root folder of the project:
   $ python3 ./data/code/ground_truths_to_coco.py
   
Inputs:
- Feather file containing the OOD annotation data frame:
  './data/processed/df_roi.feather'

Outputs:
- OOD annotations as JSON files in COCO format:
  './data/processed/ground_truth_coco.json' # all classes kept
  './data/processed/ground_truth_coco_single_cls.json' # single class (class-agnostic)
"""

import os
import sys
import json
import pandas as pd
import git

# Get the root path of the project using Git and print it for verification
prj_path = git.Repo('.', search_parent_directories=True).working_tree_dir
print(f"Root path of the project:\n'{prj_path}'")

# Add the path to the directory containing utils.py to sys.path
sys.path.append(f"{prj_path}/code")

# Import needed function from utils.py
from utils import df_to_coco


# Ground truth data - OOD annotations data frame
file_path = os.path.join(prj_path, 'data', 'processed', 'df_roi.feather')
df_true = pd.read_feather(file_path)

# Sort df_true by path and then by id_box
df_true.sort_values(by=['new_filename', 'id_box'], inplace=True)
# Reset index
df_true.reset_index(drop=True, inplace=True)

# Map the unique labels in 'p1_labels' (labels of the pretrained weights) to integers
# Use a mapping dictionary.
# Note also that for COCO category_id = 0, "it is usually considered as having
# no label information
# However, categories can have any integer id value, including 0"
# https://github.com/openvinotoolkit/datumaro/issues/156
# For single class I used id 1; but note that YOLO label IDs start from 0 
# and I passed them as such to the JSON file.
label_mapping = {
    'araneae':       0,
    'coleoptera':    1,
    'diptera':       2,
    'hemiptera':     3,
    'hymenoptera_f': 4,
    'hymenoptera':   5,
    'lepidoptera':   6,
    'orthoptera':    7
}

# Create the new column by mapping the 'p1_labels' column
df_true['yolo_lbl_id'] = df_true['p1_labels'].map(label_mapping)

# for diagnostic purposes:
# print( df_true[["yolo_lbl_id", "p1_labels"]].value_counts() )


# Save to JSON:

# Path to the cropped images folder
cropped_dir = os.path.join(prj_path, 'data', 'images', 'cropped')

# a) - treat all insects as separate classes (their true labels)
gt_json_file = os.path.join(prj_path, 'data', 'processed', 'ground_truth_coco.json')
df_true_coco = df_to_coco(df_true, cropped_dir, label_mapping, 
                          single_class=False, include_metadata=False)
with open(gt_json_file, 'w') as f:
    json.dump(df_true_coco, f, indent=4)
print(f"Multi-class COCO JSON file saved to:\n'{gt_json_file}'")
    
# b) - treat all insects as a single class
gt_json_file = os.path.join(prj_path, 'data', 'processed', 'ground_truth_coco_single_cls.json')
df_true_coco = df_to_coco(df_true, cropped_dir, label_mapping, 
                          single_class=True, include_metadata=False)
with open(gt_json_file, 'w') as f:
    json.dump(df_true_coco, f, indent=4)
print(f"Class-agnostic COCO JSON file saved to:\n'{gt_json_file}'")