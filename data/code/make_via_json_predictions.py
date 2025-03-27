"""
Overview:

Script to generate a VIA (VGG Image Annotator) compatible JSON file for insect
annotations within the cropped OOD images, together with the predicted boxes of
the "best" model.


Usage:
1. Activate the corresponding environment:
   $ source ./envs/general/bin/activate

2. Run the script from the root folder of the project:
   $ python3 ./data/code/make_via_json_predictions.py

Inputs:
- COCO JSON file with ground truth boxes:
  './data/processed/ground_truth_coco.json'
- COCO JSON file with predictions:
  './detectors/predictions/yolov5s/conf_0.201918_iou_0.3.json'

Outputs:
- VIA-compatible JSON file:
  './detectors/predictions/yolov5s/gt_and_pred_for_via.json'

Notes:
- The generated JSON file can be visualized using the VIA tool:
  https://www.robots.ox.ac.uk/~vgg/software/via/
"""


import os
import sys
import time
import pandas as pd
import json
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


# Read the COCO JSON file with ground truth boxes
path_gt_json = os.path.join(prj_path, 'data', 'processed', 'ground_truth_coco.json')
with open(path_gt_json, 'r') as file:
    data = json.load(file)

df_gt_img = pd.DataFrame([{
    "image_id": item["id"],
    "img_width": item["width"],   # optional
    "img_height": item["height"], # optional
    "file_name": item["file_name"]
} for item in data["images"]])

df_gt_boxes = pd.DataFrame([{
    "image_id": item["image_id"],
    "category_id": item["category_id"],
    "x": item["bbox"][0],
    "y": item["bbox"][1],
    "width": item["bbox"][2],
    "height": item["bbox"][3]
} for item in data["annotations"]])

df_gt_categ = pd.DataFrame([{
    "category_id": item["id"],
    "category_name": item["name"]
} for item in data["categories"]])

# Merge:
df_gt_boxes = pd.merge(df_gt_boxes, df_gt_categ, on='category_id')
df_gt = pd.merge(df_gt_boxes, df_gt_img, on='image_id')


# Read the COCO JSON file with predictions
path_pred_json = os.path.join(prj_path, 'detectors', 'predictions', 'yolov5s', 
                              'conf_0.201918_iou_0.3.json')

with open(path_pred_json, 'r') as file:
    data = json.load(file)
    
# Extract data and create DataFrame
df_pred = pd.DataFrame([{
    "image_id": item["image_id"],
    "file_name": item["file_name"],
    "x": item["bbox"][0],
    "y": item["bbox"][1],
    "width": item["bbox"][2],
    "height": item["bbox"][3],
    "score": item["score"],
    "category_id": item["category_id"]
} for item in data])


# Add the category name to predictions based on a mapping dictionary
label_dict = {
    0: 'araneae',
    1: 'coleoptera',
    2: 'diptera',
    3: 'hemiptera',
    4: 'hymenoptera_f',
    5: 'hymenoptera',
    6: 'lepidoptera',
    7: 'orthoptera'
}

# Create the new column 'yolo_label_name'
df_pred['category_name'] = df_pred['category_id'].map(label_dict)
print( df_pred[["category_name", "category_id"]].value_counts() ) # for verification


# Construct the file paths (use "path" name exactly for this column):
df_gt['path'] = df_gt['file_name'].apply(lambda x: os.path.join(prj_path, 'data', 'images', 'cropped', x))
df_pred['path'] = df_pred['file_name'].apply(lambda x: os.path.join(prj_path, 'data', 'images', 'cropped', x))


# Add a "object_type" column to both data frames:
# For the ground truth: use "ground truth" and for the predictions: use "prediction"
df_gt['object_type'] = "ground truth"
df_pred['object_type'] = "prediction"


# Concatenate the two data frames (pred & ground truths)
df_for_via = pd.concat([df_gt, df_pred], ignore_index=True)
# Sort by path and then by object_type and category_id
df_for_via = df_for_via.sort_values(by=['path', 'object_type', 'category_id'], ascending=True)


# Convert to VIA JSON

cols_atr = ['object_type', 'category_id', 'category_name', 'score', 'img_width', 'img_height']

via_json = df_to_via_json(df_for_via,
                          project_name='ground_truth_and_pred',
                          attribute_cols=cols_atr)

file_path = os.path.join(prj_path, 'detectors', 'predictions', 'yolov5s',
                         'gt_and_pred_for_via.json')
with open(file_path, 'w') as f:
    simplejson.dump(via_json, f, ignore_nan=True, indent=4)

print(f"VIA JSON file saved to:\n'{file_path}'")
# This JSON file now can be visualized in the VIA tool 
# See https://www.robots.ox.ac.uk/~vgg/software/via/

# Print time taken in seconds and minutes
end_time = time.time()
total_time = end_time - start_time
print(f"Time taken: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")