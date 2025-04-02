"""
Helper functions for the notebook model_performance.ipynb
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.ops import box_iou
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def match_predictions_to_ground_truth(df_true, df_pred, iou_threshold):
    """
    Matches predicted bounding boxes to ground truth boxes based on IoU and
    updates the DataFrames in place. It iterates through each ground
    truth box and finds the best matching prediction based on the
    Intersection over Union (IoU) metric. It modifies df_true and `df_pred` in place:
    - Updates df_true with the indices and details of matched predictions.
    - Marks matched predictions (rows) in `df_pred` to prevent reuse.
    - Assigns NaN to unmatched ground truth boxes or when no adequate prediction
      is found.
    
    Args:
        df_true (pandas.DataFrame): 
            DataFrame containing ground truth bounding box data.
        df_pred (pandas.DataFrame): 
            DataFrame containing predicted bounding box data.
        iou_threshold (float):
            An IoU threshold value (e.g. 0.5) for determining whether a
            predicted box sufficiently matches a ground truth box.
    
    Returns:
        None:
            This function modifies `df_true` and `df_pred` in place.
    """
    # Initialise the column idx_matched with False
    df_pred['idx_matched'] = False
    
    # Iterate over each ground truth box in df_true.
    # for true_idx, row in df_true.iloc[[45, 46]].iterrows(): # for debugging purposes
    for true_idx, row in df_true.iterrows():
        filename = row['new_filename']
        # Get the corresponding predictions for this image as long as they were
        # not matched to a ground truth box already.
        df_pr = df_pred[df_pred['new_filename'] == filename]
        # Keep only the predictions that were not matched to a ground truth box
        # already.
        df_pr = df_pr[df_pr['idx_matched'] == False]
    
        # If there is at least one x coordinate as NaN it means there are no
        # predictions for this image and move to the next image. There is no
        # image where a box is NaN and another box was predicted. If there is a
        # NaN box then there are no predictions. Testing for df_pr.empty is only
        # needed if it happens that the path is not in df_pred. This indicates a
        # more serious problem, like the image path is wrong or missing images.
        if df_pr.empty or df_pr['x'].isnull().any() or df_pr['idx_matched'].all():
            df_true.loc[true_idx, 'pred_idx'] = np.nan
            df_true.loc[true_idx, 'pred_iou'] = np.nan
            df_true.loc[true_idx, 'pred_conf'] = np.nan
            df_true.loc[true_idx, 'pred_label'] = np.nan
            df_true.loc[true_idx, 'pred_x'] = np.nan
            df_true.loc[true_idx, 'pred_y'] = np.nan
            df_true.loc[true_idx, 'pred_width'] = np.nan
            df_true.loc[true_idx, 'pred_height'] = np.nan
            continue
    
        # Convert bounding box coordinates to tensors.
        # For row[] (which is a pandas series) I needed to do a different
        # transformation than for df_pr[] (which is a data frame).
        coord_truth = row[['x', 'y', 'x2', 'y2']]
        boxes_truth = torch.tensor([coord_truth.tolist()])
        
        coord_pred = df_pr[['x', 'y', 'x2', 'y2']].values
        boxes_pred = torch.tensor(coord_pred)
    
        # Calculate IoU for all combinations of ground truth and predicted
        # boxes. The resulting ious tensor has one row for each ground truth box
        # (just one here, because we iterate row by row in df_true, so one
        # ground truth box per iteration), and one column for each predicted box
        # (which can be multiple). So, the element at position (i, j) in the
        # ious tensor represents the IoU between the i-th ground truth box and
        # the j-th predicted box.
        ious = box_iou(boxes_truth, boxes_pred)
        
        is_iou_over_thresh = ious[0] >= iou_threshold
        is_iou_over_thresh = is_iou_over_thresh.numpy()
        
        # If there is no IoU over the threshold, mark that to the ground truth
        # box metadata.
        if np.sum(is_iou_over_thresh) == 0:
            df_true.loc[true_idx, 'pred_idx'] = np.nan
            df_true.loc[true_idx, 'pred_iou'] = np.nan
            df_true.loc[true_idx, 'pred_conf'] = np.nan
            df_true.loc[true_idx, 'pred_label'] = np.nan
            df_true.loc[true_idx, 'pred_x'] = np.nan
            df_true.loc[true_idx, 'pred_y'] = np.nan
            df_true.loc[true_idx, 'pred_width'] = np.nan
            df_true.loc[true_idx, 'pred_height'] = np.nan
        # If there is at least one IoU over the threshold, then from these
        # candidates find the prediction with the highest confidence and assign
        # it to the ground truth box.
        else:
            confidence_pred = df_pr['yolo_conf'].values
            max_conf = confidence_pred[is_iou_over_thresh].max()
            max_conf_idx = np.where(confidence_pred == max_conf)
            max_conf_idx = np.array(max_conf_idx).flatten()[0]
            
            # Assign the needed values to the ground truth box.
            
            pred_idx = df_pr.index[max_conf_idx]
            df_true.loc[true_idx, 'pred_idx'] = pred_idx
            df_pred.loc[pred_idx, 'idx_matched'] = True
            
            best_iou = ious[0, max_conf_idx].item()
            df_true.loc[true_idx, 'pred_iou'] = best_iou
            
            df_true.loc[true_idx, 'pred_conf'] = max_conf
            
            labels_pred = df_pr['yolo_label_name'].values
            label = labels_pred[max_conf_idx]
            df_true.loc[true_idx, 'pred_label'] = label
            
            df_true.loc[true_idx, 'pred_x'] = df_pr['x'].values[max_conf_idx]
            df_true.loc[true_idx, 'pred_y'] = df_pr['y'].values[max_conf_idx]
            df_true.loc[true_idx, 'pred_width'] = df_pr['width'].values[max_conf_idx]
            df_true.loc[true_idx, 'pred_height'] = df_pr['height'].values[max_conf_idx]
            

def compute_metrics_localization(tp, fp, fn):
    """
    Calculates precision, recall, and F1-score from true positives (tp), false
    positives (fp), and false negatives (fn) for the localization task.
    
    Args:
        tp (int):
            The number of true positive cases.
        fp (int):
            The number of false positive cases.
        fn (int):
            The number of false negative cases.
    
    Returns:
        tuple: 
            A tuple containing precision, recall, and F1-score (as floats).
    """
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    
    if precision > 0 and recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1


def compute_metrics_classification(df_seq, df_true, gt_label_col, pred_label_col, dec=4):
    """
    Computes classification metrics for each unique category in
    `df_seq[gt_label_col]`, and overall (weighted by nr. of individuals).

    Args:
        df_seq (pd.DataFrame): 
            DataFrame with ground truths and predicted labels for individual
            insect (sequence).
        df_true (pd.DataFrame): 
            DataFrame with ground truths and predicted labels for insect
            instances.
        gt_label_col (str): 
            Column name for ground truth labels. Must exist in `df_seq` and
            `df_true`.
        pred_label_col (str): 
            Column name for predicted labels. Must exist in `df_seq`.
        dec (int): 
            Number of decimal places for the results. Default is 4.

    Returns:
        pd.DataFrame:
            Metrics table with sorted results and a total summary row.
    """

    # Create a list to store results
    result = []
    
    total_n_insect = df_seq.shape[0]

    # Get unique categories and sort them
    categories = np.sort(df_seq[gt_label_col].unique())

    # Iterate over categories and compute metrics:
    for category in categories:
        # Create binary vectors for the current category. Missing values in
        # yolo_group will be converted to 0s because the filter returns False
        # and then the conversion to integer will convert them to 0.
        y_true = (df_seq[gt_label_col] == category).astype(int)
        y_pred = (df_seq[pred_label_col] == category).astype(int)

        # Compute confusion matrix and get TN, FP, FN, TP
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        n_seq = (df_seq[gt_label_col] == category).sum()
        n_seq_prop = n_seq / total_n_insect * 100
        n_boxes = (df_true[gt_label_col] == category).sum()

        precision = tp / (tp + fp)
        # Recall, also known as True positive rate (TPR), or Sensitivity.
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        # Specificity, also known as True negative rate (TNR)
        specificity = tn / (tn + fp)

        # Number of detected insects / sequences (i.e., predictions that are not NaN)
        n_detected = df_seq[(df_seq[gt_label_col] == category) & (df_seq[pred_label_col].notnull())].shape[0]

        result.append({
            'Category': category,
            'N.box': n_boxes,
            'N.insect': n_seq,
            'N.insect%': round(n_seq_prop, 2),
            'N.detected': n_detected,
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn,
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1': round(f1, 4),
            'Accuracy': round(accuracy, 4),
            'Specificity': round(specificity, 4)
        })

    # Create DataFrame from results
    res_df = pd.DataFrame(result)

    # Sort by 'N.insect' in descending order
    res_df = res_df.sort_values(by='N.insect', ascending=False)
    
    # Compute summary row
    
    # Total nr. boxes (instances)
    total_n_boxes = res_df['N.box'].sum()
    # Total nr. insects (sequences)
    total_n_seq = res_df['N.insect'].sum()
    # total_n_seq must be equal to total_n_insect above; stop with error if not
    if total_n_seq != total_n_insect:
        raise ValueError("Total number of insects in df_seq does not match the total number of insects in df_true.")
    
    total_n_seq_prop = int(round(total_n_seq / total_n_insect, 2) * 100)
    total_n_detected = res_df['N.detected'].sum()
    
    total_tp = res_df['TP'].sum()
    total_fp = res_df['FP'].sum()
    total_fn = res_df['FN'].sum()
    total_tn = res_df['TN'].sum()

    # Compute weighted metrics (weights = nr. insects)
    precision_w = (res_df['Precision'] * res_df['N.insect']).sum() / total_n_seq
    recall_w = (res_df['Recall'] * res_df['N.insect']).sum() / total_n_seq
    weighted_f1 = 2 * (precision_w * recall_w) / (precision_w + recall_w)
    accuracy_w = (res_df['Accuracy'] * res_df['N.insect']).sum() / total_n_seq
    specificity_w = (res_df['Specificity'] * res_df['N.insect']).sum() / total_n_seq

    # Append summary row
    total_row = pd.DataFrame([{
        'Category': 'Total',
        'N.box': total_n_boxes,
        'N.insect': total_n_seq,
        'N.insect%': total_n_seq_prop,
        'N.detected': total_n_detected,
        'TP': total_tp,
        'FP': total_fp,
        'FN': total_fn,
        'TN': total_tn,
        'Precision': round(precision_w, dec),
        'Recall': round(recall_w, dec),
        'F1': round(weighted_f1, dec),
        'Accuracy': round(accuracy_w, dec),
        'Specificity': round(specificity_w, dec)
    }])

    # Concatenate results with the total row
    res_df = pd.concat([res_df, total_row], ignore_index=True)
    
    return res_df