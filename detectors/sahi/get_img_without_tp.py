"""
Overview:
SAHI was executed only on the images where the NMS optimized model (YOLOv5s)
failed to detect insect instances. This scripts filters for the images without
any TPs (at IoU 0.5) and store the corresponding annotations for further
processing with SAHI.

Usage:
1. Activate the corresponding environment:
   $ source ./envs/pycocotools/bin/activate # for numpy-pycocotools compatibility

2. Run the script from the root folder of the project:
   $ python3 ./detectors/sahi/get_img_without_tp.py

Inputs:
- COCO JSON file with ground truth boxes (class agnostic):
  './data/processed/ground_truth_coco_single_cls.json'
- COCO JSON file with predictions

Outputs:
- COCO JSON file with ground truth boxes for images where the NMS optimized
model failed to detect insect instances.
"""


import os
import time
import json
import numpy as np
import pandas as pd
import git
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Get the path to the project folder
prj_path = git.Repo('.', search_parent_directories=True).working_tree_dir
print(f"Root path of the project:\n'{prj_path}'")

# Start the timer
start_time = time.time()
print("Processing...")


# ==============================================================================
# Read json files - gt & dt
# ==============================================================================
# - gt = ground truths
# - dt = detections

# Read the COCO ground truth data - single class
path_gt_json = os.path.join(prj_path, 'data', 'processed', 'ground_truth_coco_single_cls.json')
coco_gt = COCO(path_gt_json)

# Load predictions (detections)
path_pr_json = os.path.join(prj_path, 'detectors', 'predictions', 
                              'yolov5s', 'conf_0.201918_iou_0.3.json')
coco_pr_data = json.load(open(path_pr_json))


# Reduce all categories to category_id=1 = evaluate as insect detector
for item in coco_pr_data:
    item['category_id'] = 1
coco_pr = coco_gt.loadRes(coco_pr_data)


# ==============================================================================
# Run COCO evaluation so that we defined TP, FP, FN, etc. for each image.
# ==============================================================================

# Set only needed parameters to gain speed

# Create COCO Eval object
coco_eval = COCOeval(cocoGt=coco_gt, 
                     cocoDt=coco_pr, 
                     iouType='bbox')

# Notes:
# - if useCats=0 category labels are ignored as in proposal scoring.
# - multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
# https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py#L32C1-L33C72

coco_eval.params.useCats = 0 
# not really needed here, because we have only one class already in the json
# files, but keep it for defensive reasons at the moment.


# Set box area ranges to "all" only, to avoid computing separately for each case
# (small, medium, large)
# You will get -1 for AP and AR metrics for the small, medium and large
# "Note: precision and recall==-1 for settings with no gt objects." from:
# https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/cocoeval.py#L52C7-L52C70
coco_eval.params.areaRng = [[0 ** 2, 1e5 ** 2]]
coco_eval.params.areaRngLbl = ['all']

# Set IoU threshold to only 0.5
coco_eval.params.iouThrs = [0.5]

# Set Max detections per image.
# These are how many detections are allowed for evaluation.
# Cal only be given as a list of integers, for example:
coco_eval.params.maxDets = [0, 0, 100] # perhaps to speed up a bit as well
# Will get values of 0 for AR for maxDets=1 and 10
# coco_eval.params.maxDets = [0, 0, 1000] # this fails to compute 1st row AP
# coco_eval.params.maxDets = [10, 100, 1000] # this computes 1st row AP, but only for M=100...

# Evaluation
coco_eval.evaluate()   # Calculates the metrics
coco_eval.accumulate() # Stores the values in the coco_eval's 'eval' object
coco_eval.summarize()  # Compute and display summary metrics


# ==============================================================================
# Get TPs and images without TPs
# ==============================================================================

# Initialize counters
gt_ignore_count = 0
dt_ignore_count = 0

# Iterating through all evaluation images
for eval_img in coco_eval.evalImgs:
    # Check if there's any gtIgnore marked
    if np.any(eval_img['gtIgnore']):
        gt_ignore_count += 1

    # Check if there's any dtIgnore marked
    if np.any(eval_img['dtIgnore']):
        dt_ignore_count += 1

# Example of element contained in the 'evalImgs' list; for debugging purposes.
# coco_eval.evalImgs[0]

# Assuming coco_eval.evalImgs is already defined and populated
for eval_img in coco_eval.evalImgs:
    # Initialize counts
    tp_count = 0
    fp_count = 0
    fn_count = 0
    
    # Extract matches
    dt_matches = eval_img['dtMatches']
    gt_matches = eval_img['gtMatches']
    
    # Compute TP, FP, FN
    tp = (dt_matches != 0).sum()
    fp = (dt_matches == 0).sum()
    fn = (gt_matches == 0).sum()
    
    # Update counts
    tp_count += tp
    fp_count += fp
    fn_count += fn
    
    # Update the dictionary with new fields
    eval_img['TP'] = tp_count
    eval_img['FP'] = fp_count
    eval_img['FN'] = fn_count

# At this point, coco_eval.evalImgs has been updated with 'TP', 'FP', 'FN' for
# each image.

# coco_eval.evalImgs[0] # for debugging purposes.
# coco_gt.imgs[1]

# Find all image_ids where TP = 0
image_ids_with_tp_zero = [eval_img['image_id'] for eval_img in coco_eval.evalImgs if eval_img['TP'] == 0]

# Fetch filenames for these image IDs from the coco_gt object
filenames_with_tp_zero = [coco_gt.imgs[image_id]['file_name'] for image_id in image_ids_with_tp_zero]
print(f"Found {len(filenames_with_tp_zero)} images with TP = 0")
# Now use these images to create a JSON file for running SAHI on them.


# ==============================================================================
# Make JSON file for SAHI
# ==============================================================================

# Read the ground truth and filter only for the images with TP 0 and get their
# annotations.

with open(path_gt_json, 'r') as file:
    json_gt = json.load(file)

# Remove the 'dir_path' key from the JSON, if it exists
json_gt.pop('dir_path', None)

# Filter images to keep only those with filenames in filenames_with_tp_zero
filtered_images = [img for img in json_gt["images"] if img['file_name'] in filenames_with_tp_zero]

# Collect the IDs of the filtered images
filtered_image_ids = set(img['id'] for img in filtered_images)

# Filter annotations to keep only those related to the filtered images
filtered_annotations = [anno for anno in json_gt["annotations"] if anno['image_id'] in filtered_image_ids]
    
# Replace the original images and annotations with the filtered ones
json_gt["images"] = filtered_images
json_gt["annotations"] = filtered_annotations


# Save the filtered COCO JSON to a new file
filtered_path_gt_json = os.path.join(prj_path, 'detectors', 'sahi', 'sahi_predictions' 
                                    'ground_truth_without_tp_for_sahi.json')

with open(filtered_path_gt_json, 'w') as file:
    json.dump(json_gt, file, indent=4)

print(f'Output saved to:\n{filtered_path_gt_json}')

end_time = time.time()
total_time = end_time - start_time
print(f"Time taken: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")    