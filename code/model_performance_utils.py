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
    Intersection over Union (IoU) metric. It modifies df_true and df_pred in place:
    - Updates df_true with the indices and details of matched predictions.
    - Marks matched predictions (rows) in df_pred to prevent reuse.
    - Assigns NaN to unmatched ground truth boxes or when no adequate prediction is found.
    
    Parameters:
    ----------
    df_true : pandas.DataFrame
        DataFrame containing ground truth bounding box data.
    df_pred : pandas.DataFrame
        DataFrame containing predicted bounding box data.
    iou_threshold : float
        An IoU threshold value for determining whether a predicted box
        sufficiently matches a ground truth box.
    
    Returns:
    -------
    None : This function modifies df_true and df_pred in place.
    
    Notes:
    ------
    Ensure that both DataFrames have the appropriate columns before calling this
    function. The computation of IoU requires torch tensors, and the function
    relies on box_iou() from torchvision.ops 
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
            

def calculate_metrics(tp, fp, fn):
    """
    Calculate precision, recall, and F1-score from true positives, false
    positives, and false negatives.
    
    Parameters:
    ----------
    tp : int
        The number of true positive cases.
    fp : int
        The number of false positive cases.
    fn : int
        The number of false negative cases.
    
    Returns:
    -------
    tuple
        A tuple containing precision, recall, and F1-score as floats.
    """
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    
    if precision > 0 and recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1


def plot_confusion_matrix_custom(y_true, y_pred):
    """
    Plots a custom confusion matrix with both absolute and normalized values.
    
    Parameters:
    ----------
    y_true : array-like
        True labels for the dataset.
    y_pred : array-like
        Predicted labels, provided by the model.
    
    Notes:
    ------
    The normalization is done by dividing each entry in the rows of the confusion 
    matrix by the sum of the entries in that row.
    
    The resulting plot displays the normalized values with percentages and shows the 
    absolute counts in parentheses.
    """
    
    # Get unique labels
    unique_labels = np.unique(np.concatenate((y_true, y_pred)))

    # Create the confusion matrix (absolute values)
    cm_abs = confusion_matrix(y_true, y_pred, labels=unique_labels)
    
    # Create a normalized version of it
    cm_norm = confusion_matrix(y_true, y_pred, labels=unique_labels, normalize='true')
    cm_norm = cm_norm * 100 # convert to percentages
    
    # # We could also manually normalize the confusion matrix:
    # # Compute the sum of each row in the confusion matrix, which is the number of
    # # ground truth insects.
    # row_sums = cm.sum(axis=1)
    # # Divide each entry in a row by the sum of that row,  thereby normalizing the values. 
    # # The [:, np.newaxis] part ensures that broadcasting works correctly during division. 
    # cm_norm = (cm / row_sums[:, np.newaxis]) * 100
    # # Replaces any NaNs with zeros. 
    # # NaNs can occur if a row sum is zero, leading to a division by zero.
    # cm_norm[np.isnan(cm_norm)] = 0 

    fig, ax = plt.subplots(figsize=(4, 3))

    # Create a ConfusionMatrixDisplay object
    # Use the normalized version so that the colors match normalized values
    ConfusionMatrixDisplay(cm_norm, 
                           display_labels=unique_labels).plot(ax=ax, 
                                                              #cmap=plt.cm.Blues,
                                                              values_format='.2f',
                                                              xticks_rotation='vertical')

    # Add absolute values to the plot
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            plt.text(j, i, f"\n\n({cm_abs[i, j]:.0f})", ha="center", va="center", color="white")

    plt.show()


def compute_metrics(df_seq, df_true, label_col, common_col, dec=4):
    """
    Compute classification metrics for each unique category in df_seq[label_col].

    Parameters:
    - df_seq (pd.DataFrame): DataFrame containing sequence predictions.
    - df_true (pd.DataFrame): DataFrame containing ground truth data.
    - label_col (str): Column name for predicted labels.
    - common_col (str): Column name for ground truth labels.
    - dec (int): Number of decimal places for the results. Default is 4.

    Returns:
    - pd.DataFrame: Metrics table with sorted results and a total summary row.
    """

    # Create a list to store rows
    rows = []

    # Get the unique categories in the given label column and sort them
    categories = np.sort(df_seq[label_col].unique())

    # Iterate over the categories
    for category in categories:
        # Get total number of ground truth sequences
        n_seq = (df_seq[label_col] == category).sum()
        n_boxes = (df_true[label_col] == category).sum() # this is optional

        # Calculate TP, FP, and FN for the current category
        # FP = predictions of the given taxa order (category) wrongly assigned to another taxa
        # FN = ground truth taxa order (category) not classified at all or wrongly classified
        tp = ((df_seq[label_col] == category) & (df_seq[common_col] == category)).sum()
        fp = ((df_seq[label_col] != category) & (df_seq[common_col] == category)).sum()
        fn = ((df_seq[label_col] == category) & (df_seq[common_col] != category)).sum()

        # Calculate precision and recall for the current category
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

        # Append results as a dictionary
        rows.append({
            'Category': category,
            'N.seq': n_seq,
            'N.box': n_boxes,
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'Precision': round(precision, dec),
            'Recall': round(recall, dec),
            'F1': round(f1, dec)
        })

    # Create DataFrame from the list
    results = pd.DataFrame(rows)

    # Sort by 'N.seq' in descending order
    results = results.sort_values(by='N.seq', ascending=False)

    # Compute total row
    total_n_seq = results['N.seq'].sum()
    total_n_boxes = results['N.box'].sum()
    total_tp = results['TP'].sum()
    total_fp = results['FP'].sum()
    total_fn = results['FN'].sum()

    # Compute weighted precision, recall, and F1-score
    weighted_precision = (results['Precision'] * results['N.seq']).sum() / total_n_seq if total_n_seq > 0 else 0
    weighted_recall = (results['Recall'] * results['N.seq']).sum() / total_n_seq if total_n_seq > 0 else 0
    weighted_f1 = 2 * (weighted_precision * weighted_recall) / (weighted_precision + weighted_recall) if (weighted_precision + weighted_recall) > 0 else 0.0

    # Append total row
    total_row = pd.DataFrame([{
        'Category': 'Total',
        'N.seq': total_n_seq,
        'N.box': total_n_boxes,
        'TP': total_tp,
        'FP': total_fp,
        'FN': total_fn,
        'Precision': round(weighted_precision, dec),
        'Recall': round(weighted_recall, dec),
        'F1': round(weighted_f1, dec)
    }])

    # Concatenate results with the total row
    results = pd.concat([results, total_row], ignore_index=True)

    return results


def compute_confusion_metrics(df_seq, df_true, label_col, common_col, dec=4):
    """
    Compute confusion matrix metrics for each unique category in df_seq[label_col].

    Parameters:
    - df_seq (pd.DataFrame): DataFrame containing sequence predictions.
    - df_true (pd.DataFrame): DataFrame containing ground truth data.
    - label_col (str): Column name for predicted labels.
    - common_col (str): Column name for ground truth labels.
    - dec (int): Number of decimal places for the results. Default is 4.

    Returns:
    - pd.DataFrame: Metrics table with sorted results and a total summary row.
    """

    # Create a list to store results
    result = []
    
    total_n_insect = df_seq.shape[0]

    # Get unique categories and sort them
    categories = np.sort(df_seq[label_col].unique())

    # Iterate over categories
    for category in categories:
        y_true = (df_seq[label_col] == category).astype(int)
        y_pred = (df_seq[common_col] == category).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        n_seq = (df_seq[label_col] == category).sum()
        n_seq_prop = n_seq / total_n_insect * 100
        n_boxes = (df_true[label_col] == category).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # True positive rate (Sensitivity)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True negative rate

        # Number of detected sequences (i.e., predictions that are not NaN)
        n_detected = df_seq[(df_seq[label_col] == category) & (df_seq[common_col].notnull())].shape[0]

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
            'Recall/TPR/Sensitivity': round(recall, 4),
            'F1': round(f1, 4),
            'Accuracy': round(accuracy, 4),
            'Specificity/TNR': round(specificity, 4)
        })

    # Create DataFrame from results
    result_df = pd.DataFrame(result)

    # Sort by 'N.insect' in descending order
    result_df = result_df.sort_values(by='N.insect', ascending=False)
    
    # Compute total row
    total_n_boxes = result_df['N.box'].sum()
    total_n_seq = result_df['N.insect'].sum() # must be equal to total_n_insect above
    # stop with error if not
    if total_n_seq != total_n_insect:
        raise ValueError("Total number of insects in df_seq does not match the total number of insects in df_true.")
    total_n_seq_prop = int(round(total_n_seq / total_n_insect, 2) * 100)
    total_n_detected = result_df['N.detected'].sum()
    total_tp = result_df['TP'].sum()
    total_fp = result_df['FP'].sum()
    total_fn = result_df['FN'].sum()
    total_tn = result_df['TN'].sum()

    # Compute weighted precision, recall, and F1-score
    weighted_precision = (result_df['Precision'] * result_df['N.insect']).sum() / total_n_seq if total_n_seq > 0 else 0
    weighted_recall = (result_df['Recall/TPR/Sensitivity'] * result_df['N.insect']).sum() / total_n_seq if total_n_seq > 0 else 0
    weighted_f1 = 2 * (weighted_precision * weighted_recall) / (weighted_precision + weighted_recall) if (weighted_precision + weighted_recall) > 0 else 0.0
    weighted_accuracy = (result_df['Accuracy'] * result_df['N.insect']).sum() / total_n_seq if total_n_seq > 0 else 0
    weighted_specificity = (result_df['Specificity/TNR'] * result_df['N.insect']).sum() / total_n_seq if total_n_seq > 0 else 0

    # Append total row
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
        'Precision': round(weighted_precision, dec),
        'Recall/TPR/Sensitivity': round(weighted_recall, dec),
        'F1': round(weighted_f1, dec),
        'Accuracy': round(weighted_accuracy, dec),
        'Specificity/TNR': round(weighted_specificity, dec)
    }])

    # Concatenate results with the total row
    result_df = pd.concat([result_df, total_row], ignore_index=True)
    
    return result_df