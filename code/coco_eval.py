"""
This script is used to evaluate the predictions of the YOLO models using the
pycoctools library. The evaluation is done for a single class only
(class-agnostic, insect detector evaluation).

Usage:
1. Activate the corresponding environment:
   $ source ./envs/general/bin/activate
2. Run the script from the root folder of the project:   
   $ python3 ./code/coco_eval.py --path_pred --path_gt --cores

Parameters:
--path_pred: Path to the directory containing the prediction JSON files. It can have nested folders.
--path_gt:   Path to the the ground truth JSON file. Must contain a single class.
--path_res:  Path and name with the feather file where the results will be saved
--cores:     Number of CPU cores to use (default: 4)

Outputs: two feather files with evaluation metrics.

Example:

prj_path="~/smartphone-insect-detect/"
cd "${prj_path}/code"
path_pred="${prj_path}/detectors/runs/detections_conf_0.001/"
path_gt="${prj_path}/data/processed/ground_truth_coco_single_cls.json"
path_res="${prj_path}/data/processed/df_coco_eval_conf_0.001_ious_0.1_0.9.feather"

python3 coco_eval.py \
    --path_pred "${path_pred}" \
    --path_gt "${path_gt}" \
    --path_res "${path_res}" \
    --cores 14

# It will output:
# - ./data/processed/df_coco_eval_conf_0.001_ious_0.1_0.9.feather
# - ./data/processed/df_coco_eval_conf_0.001_ious_0.1_0.9_arrays.feather

NOTE:
numpy-pycocotools compatibility.

Note that due to updates in `numpy`, you could get an error like this:

ValueError: Calling nonzero on 0d arrays is not allowed. Use
np.atleast_1d(scalar).nonzero() instead. If the context of this error is of the
form `arr[nonzero(cond)]`, just use `arr[cond]`. 

You might need then:
pip install pycocotools==2.0.7
pip install numpy==1.25.2
See ./envs/README.md, section about pycocotools environment.
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set ups for the script
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import gc
import os
import sys
import re
import time
import json
import argparse
import numpy as np
import pandas as pd
from scipy.integrate import trapz
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import dask
from dask.distributed import Client, as_completed
# Additional import for printing / logging and error handling
import traceback
import logging

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Helper functions to run evaluation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def run_COCOeval(path_dt):
    """
    Run COCOeval for a single JSON file with detections.
    
    Args:
    path_gt: Path to the ground truth JSON file
    path_dt: Path to the JSON file with detections
    
    Returns:
    coco_eval: The COCOeval object
    """
    try:
        # Load the ground truth data using COCO
        coco_gt = COCO(args.path_gt)
        # You can read the multi class as well and then convert all the classes
        # to a single class.

        # Load predictions (detections) - generated with detection/yolo_txt_to_coco.py
        # Convert them to single class (if not already done) - evaluate as insect detector
        coco_pr_data = json.load(open(path_dt))
        for item in coco_pr_data:
            item['category_id'] = 1 # this categ id should match the one in the ground truth file
        coco_pr = coco_gt.loadRes(coco_pr_data)
        # Remove coco_pr_data with the attempt to release memory back to OS
        # (useful for big files)
        del coco_pr_data
        gc.collect() # force garbage collection

        # Set only needed parameters for evaluation

        # Create COCO Eval object
        coco_eval = COCOeval(cocoGt=coco_gt, cocoDt=coco_pr, iouType='bbox')
        
        # Note: if useCats=0 category labels are ignored as in proposal scoring.
        # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
        # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py#L32C1-L33C72
        coco_eval.params.useCats = 0 
        # set evaluation for single class (safety net in case caetg ids do nto match)
        # print(f"catIds: {coco_eval.params.catIds}") # for debugging purposes

        # Set box area ranges to "all" only, to avoid computing separately for
        # each case (small, medium, large).        
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


        # Evaluation
        coco_eval.evaluate()   # Calculates the metrics
        coco_eval.accumulate() # Stores the values in the coco_eval's 'eval' object
        coco_eval.summarize()  # Compute and display summary metrics
        
        return coco_eval

    except Exception as e:
        # Log the exception
        logging.error(f"Error in {path_dt}: {e}")
        # Return None
        return None
    

def extract_path_parts(path_dt):
    """
    Extracts model, nms_conf, and nms_iou from a given json file path.
    Assumes the format like <model>/conf_<nms_conf>_iou_<nms_iou>/...
    """
    # Regular expression to identify the model name before '/conf_'
    match_model = re.search(r'/([^/]+)/conf_.*', path_dt)
    # Regular expression to identify 'conf_<nms_conf>_iou_<nms_iou>' before '.json'
    match_conf_iou = re.search(r'conf_([0-9\.]+)_iou_([0-9\.]+)\.json', path_dt)

    if match_model and match_conf_iou:
        model = match_model.group(1)
        nms_conf = match_conf_iou.group(1)
        nms_iou = match_conf_iou.group(2)
        
        return {'model': model, 'nms_conf': nms_conf, 'nms_iou': nms_iou}
    else:
        # Log an error if the pattern is not found
        logging.error(f"Error: Could not extract model, nms_conf, and nms_iou from {path_dt}")
        return None


def get_metrics(coco_eval, path_dt):
    """
    Calculate and extract detection metrics from a COCOeval object. This
    function computes detection counts (true positives, false positives, false
    negatives), precision, recall, F1 score, and also extracts array data for
    precision, recall, and confidence scores. Furthermore, it computes the area
    under the curve (AUC) for the precision-recall curve and identifies the
    maximum F1 score along with its corresponding confidence score.

    Args:
        coco_eval (COCOeval): The COCO evaluation object used to gather
            precision, recall, scores, and to compute overall detection metrics.
        path_dt (str): The json file path with detections used to extract
            additional metadata such as model, nms_conf, and nms_iou.

    Returns:
        dict: A nested dictionary with two keys corresponding to two data frames:
              - 'df_scalars': Contains all scalar values such as true positives
                (TP), false positives (FP), false negatives (FN), precision (P), 
                recall (R), F1 score (F1), mean average precision (AP_mean), 
                AUC of the precision-recall curve (AP_auc), maximum F1 score
                (maxF1), and its corresponding confidence score (maxF1conf).
              - 'df_arrays': Contains arrays of precision ('P_array'), recall
                ('R_array'), confidence scores ('conf_array'), and calculated 
                F1 scores ('F1_array'). It also includes the model, nms_conf, 
                and nms_iou scalar values from the file path analysis, broadcast
                across all rows.

    Detailed Steps:
        A) Computes true positives, false positives, and false negatives from 
           the COCOeval object for each evaluated image.
        B) Extracts precision, recall, and scores from the COCOeval object based 
           on fixed recall thresholds.
        C) Calculates the F1 score for each recall threshold and identifies 
           the maximum F1 score.
        D) Computes the AUC for the precision-recall curve.
        E) Returns the results organized in dictionaries, also extracting 
           additional data from `path_dt`.
    """
    
    """
    A) Compute TP, FP, FN and then overall precision, recall, and F1 score based on them.
    """
    
    # For debugging purposes
    # print(f"len(coco_eval.evalImgs): {len(coco_eval.evalImgs)}")
    # print(f"coco_eval.evalImgs[0]: {coco_eval.evalImgs[0]}")
    # print(f"coco_eval.evalImgs: {coco_eval.evalImgs}")
    
    tp_count = 0
    fp_count = 0
    fn_count = 0
    
    for eval_img in coco_eval.evalImgs:          
        dt_matches = eval_img['dtMatches']
        gt_matches = eval_img['gtMatches']
        
        # Get Nr. of TP; the operation is vectorized (element-wise);
        # A TP is any detection ("dt") box that was matched to a ground truth box ("gt")
        tp_count += (dt_matches != 0).sum()

        # Get Nr. of FP. A FP is any detection box that was not matched to a ground truth box 
        fp_count += (dt_matches == 0).sum()

        # Get nr of FN. A FN is any ground truth box that was not matched to a detection box
        fn_count += (gt_matches == 0).sum()
    
    # Compute overall precision, recall, and F1 score
    P = tp_count / (tp_count + fp_count) # precision
    R = tp_count / (tp_count + fn_count) # recall
    # Recall value same as coco_eval.stats[8] # for single class
    F1 = 2 * (P * R) / (P + R) # F1 score  
    
    
    """
    B) Extract precision, recall, and scores arrays from the COCOeval object.
    These are arrays needed to build the P-R and F1 curves (for the overall 
    scalar metrics see section A above).
    They correspond to 101 recall thresholds as per pycocotools documentation:
    https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
    
    Slicing the precision and scores attributes of the COCOeval object.
    [TxRxKxAxM] dimensions of the precision and scores arrays, where: 
        0 = IoU 0.5, 
        : = all 101 recall thresholds (coco_eval.params.recThrs)
        0 = all categories (here only one anyways, so can use 0 instead of :)
        0 = all area ranges
        2 = all max detections per image (default 100)
    """
    
    try:
        # Precision at different recall thresholds
        precision_array = coco_eval.eval['precision'][0, :, 0, 0, 2]

        # These are fixed recall levels (ranging from 0 to 1 at intervals, e.g., 0, 0.01, 0.02, ..., 1.00) 
        # used to evaluate the precision. They do not represent the actual recall values.
        # For F1, we are using the recall thresholds as a proxy for actual recall values.
        recThrs_array = coco_eval.params.recThrs  # note, same as np.arange(0, 1.01, 0.01)

        scores_array = coco_eval.eval['scores'][0, :, 0, 0, 2]
    except Exception as e:
        logging.error(f"Error in extracting the arrays for precision, recall, and scores: {e}")
        # Assign None to the arrays when extraction fails
        precision_array, recThrs_array, scores_array = (None,) * 3   

    """
    C) Compute F1 array (for F1 curve) and get maxF1 and its corresponding confidence score
    """
    
    try:
        f1_scores = 2 * (precision_array * recThrs_array) / (precision_array + recThrs_array)

        # Find the index of the maximum F1 score
        max_f1_index = np.argmax(f1_scores)

        # Find the maximum F1 score
        max_f1 = f1_scores[max_f1_index]

        # Find the corresponding confidence score
        corresponding_score = scores_array[max_f1_index]
    except Exception as e:
        # Log the error
        logging.error(f"Error computing F1 array and maxF1: {e}")
        # Assign None to max_f1 and corresponding_score when computation fails
        max_f1 = None
        corresponding_score = None

    """
    D) Compute area under the curve (AUC) for precision-recall curve.
    This should be similar to the average precision (AP) for the single class case.
    """
    
    try:
        auc = trapz(precision_array, recThrs_array)
    except Exception as e:
        # Log the error
        logging.error(f"Error computing AUC: {e}")
        # Assign None to auc when computation fails
        auc = None

    AP = coco_eval.stats[0]
    # Or, for single class, same as:
    # AP = np.mean(coco_eval.eval['precision'][0, :, 0, 0, 2])  

    """
    E) Prepare the results as a dictionary of two data frames for scalars and arrays.
    """
    
    # Extract model, nms_conf, and nms_iou from the path
    path_parts = extract_path_parts(path_dt)
    
     # Create DataFrame for scalars
    df_scalars = pd.DataFrame([{
        **path_parts, # Unpack the dictionary from extract_path_parts
        'TP': tp_count, 
        'FP': fp_count, 
        'FN': fn_count,
        'P': P,
        'R': R,
        'F1': F1,
        'AP_mean': AP,
        'AP_auc': auc,
        'maxF1': max_f1, 
        'maxF1conf': corresponding_score
        }])

    # Create DataFrame for arrays
    df_arrays = pd.DataFrame({
        'P_array': precision_array,
        'R_array': recThrs_array,
        'conf_array': scores_array,
        'F1_array': f1_scores
    })
    
    # Broadcast path_parts scalars across all rows in df_arrays. This is important when
    # concatenating the data frames from multiple json detection files later.
    for key, value in path_parts.items():
        df_arrays[key] = value
        
    return {
        'df_scalars': df_scalars,
        'df_arrays': df_arrays
    }


def eval_json(file_path):
    """    
    Evaluate a single json file with detections. 
    This function will be called in parallel below.
    """
    print(f"Evaluating {file_path}")
    
    coco_eval = run_COCOeval(file_path)
    results = get_metrics(coco_eval, file_path)
    
    return results


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Note: To my understanding, dask requires the parallel code to be inside a
# __main__ block / function to prevent unintended behavior when the
# multiprocessing module tries to fork or spawn new processes.
if __name__ == '__main__':
    
    start_time = time.time()

    # Parse arguments ----------------------------------------------------------

    # Initialize parser
    parser = argparse.ArgumentParser(description='Process input.')

    # Adding first argument for --path_pred
    parser.add_argument('--path_pred', type=str, required=True,
                        help='Path to the directory containing the prediction JSON files')

    # Adding 2nd argument for --path_gt
    parser.add_argument('--path_gt', type=str, required=True,
                        help='Path to the the ground truth JSON file')

    # Adding 3rd argument for --path_res
    parser.add_argument('--path_res', type=str, required=True,
                        help='Path and name with the feather file where the results will be saved')

    # Adding 4th argument for number of CPU cores
    parser.add_argument('--cores', type=int, default=4,
                        help='Number of CPU cores to use (default: 4)')

    # Parsing arguments
    args = parser.parse_args()


    # Check the number of CPU cores
    available_cpus = os.cpu_count()
    requested_cpus = args.cores

    if requested_cpus > available_cpus:
        sys.exit(f"Error: Requested {requested_cpus} CPUs, but only {available_cpus} CPUs are available. Please adjust your request.")
    else:
        n_cpus = requested_cpus


    # Check if args.path_gt exists
    if not os.path.exists(args.path_gt):
        sys.exit(f"Error: The expected json ground truth file path {args.path_gt} does not exist.")


    # Read the JSON files with the predictions    
    json_file_paths = []
    for root, dirs, files in os.walk(args.path_pred):
        # For diagnostic purposes
        # print(f"root: {root}")
        # print(f"dirs: {len(dirs)}")
        # print(f"dirs: {dirs}")

        # Check if the directory is a leaf directory (no subdirectories)
        if not dirs:
            # Check if there are any JSON files in the directory
            files_found = any(f.endswith('.json') for f in files)
            if files_found:
                # Add all .json files in this leaf directory to the list
                json_file_paths.extend(os.path.join(root, f) for f in files if f.endswith('.json'))

    print(f"json files found: {len(json_file_paths)}")
    # print("json_file_paths example:\n" + "\n".join(json_file_paths[:3]))

    # Check if the args.path_res feather files already exists. If yes, then stop with an error message.
    if os.path.exists(args.path_res):
        sys.exit(f"Error: The expected feather file {args.path_res} already exists. Please delete it or choose another path.")


    # Run evaluation in parallel -----------------------------------------------
    # Run evaluation for each json file - use parallel processing. 

    # Initialize empty lists to store data frames for the evaluation results
    list_df_scalars = []
    list_df_arrays = []

    # For testing purposes:
    # res = eval_json(json_file_paths[100])
    # print(res)

    # test_files = [json_file_paths[i] for i in [1, 100]]
    # for file in test_files:
    #     df_list_eval.append(eval_json(file))
    # df_eval = pd.concat(df_list_eval, ignore_index=True)
    # print(df_eval)  


    n_cpus = requested_cpus
    json_files = json_file_paths
    # For testing purposes:
    # json_files = [json_file_paths[i] for i in [1, 100]]

    # Using dask to run the evaluation in parallel (asynchronous parallel tasks)
    # Dask operates asynchronously by default when you set the dask client
    client = Client(n_workers=n_cpus)
    print(client)
    # futures = [client.submit(eval_json, path) for path in json_files] # simpler, but without a dictionary with file paths
    futures = {client.submit(eval_json, path): path for path in json_files}
    # Iterating through the futures as they are completed and look up the 
    # corresponding file path in the futures dictionary for possible error tracking.
    # And also to append the results to the list of data frames.
    for future in as_completed(futures):
        path = futures[future]
        try:
            results = future.result()
            res_scalars = results['df_scalars']
            res_arrays = results['df_arrays']
            list_df_scalars.append(res_scalars)
            list_df_arrays.append(res_arrays)
        except Exception as e:
            logging.error(f"Error processing file {path}: {e}")
            traceback.print_exc()

    df_eval_scalars = pd.concat(list_df_scalars, ignore_index=True)
    # print(df_eval_scalars)
    df_eval_scalars.to_feather(args.path_res)
    print(f"Data frame with evaluation results (scalars) saved as {args.path_res}")


    df_eval_arrays = pd.concat(list_df_arrays, ignore_index=True)
    # print(df_eval_arrays)
    path_res_arrays = os.path.splitext(args.path_res)[0] + "_arrays.feather"
    df_eval_arrays.to_feather(path_res_arrays)
    print(f"Data frame with evaluation results (arrays) saved at {path_res_arrays}")


    end_time = time.time()
    total_time = end_time - start_time
    print(f"Time taken: {total_time:.2f} seconds ({total_time/60:.2f} minutes) on {n_cpus} CPUs.")