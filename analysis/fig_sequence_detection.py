"""
Python script to visualize a sequence of images with bounding boxes (ground
truths and predictions) associated with an individual insect.

Usage:
1. Activate the corresponding environment:
    $ source ./envs/general/bin/activate

2. Run the script from the root folder of the project:
    $ python3 ./analysis/fig_sequence_detection.py --seq_id <sequence_id>
Example:
    $ python3 ./analysis/fig_sequence_detection.py --seq_id 52

Inputs:
- Feather file with results from running the notebook
`./code/model_performance.ipynb` with eval-IoU 0.1

Outputs:
- jpg file with the graph in ./results/figures/
"""

import os
import git
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import argparse 

# Get the root path of the project using Git and print it for verification
prj_path = git.Repo('.', search_parent_directories=True).working_tree_dir
print(f"Root path of the project:\n'{prj_path}'")

# Set up argument parser
parser = argparse.ArgumentParser(description="Visualize a sequence of images with bounding boxes.")
parser.add_argument("--seq_id", type=int, required=True, help="Sequence ID to visualize")
args = parser.parse_args()


def add_box_info(ax, rows, pred_font):
    """
    Helper function to add bounding box information text to the axes.
    """
    info_text = rows['box_label'].iloc[0].title()
    ax.text(0.98, 0.98, info_text, transform=ax.transAxes, fontsize=pred_font['size'], 
            fontname=pred_font['family'], va='top', ha='right', color='black', 
            bbox=dict(facecolor='white', edgecolor='none', pad=1))
    
    info_text = f"Conf: {rows['conf'].iloc[0]:.2f}"
    ax.text(0.98, 0.865, info_text, transform=ax.transAxes, fontsize=pred_font['size'], 
            fontname=pred_font['family'], va='top', ha='right', color='black', 
            bbox=dict(facecolor='white', edgecolor='none', pad=1))
    
    info_text = f"IoU: {rows['iou'].iloc[0]:.2f}"
    ax.text(0.98, 0.75, info_text, transform=ax.transAxes, fontsize=pred_font['size'], 
            fontname=pred_font['family'], va='top', ha='right', color='black', 
            bbox=dict(facecolor='white', edgecolor='none', pad=1))


def plot_images_sequence(df, n_rows, n_cols, output_path, total_width_mm=150):
    """
    Visualizes a sequence of images with bounding boxes (ground truths, true
    positives, and false positives) of insects appearing across consecutive
    images. Displays a figure with n_rows rows and n_cols columns of images.

    Args:
        df (DataFrame): Data frame containing the image and bounding box information. 
            An image can contain multiple bounding boxes, therefore multiple rows in df.
            
            It must contain the following columns:
            - path: path to the image
            - x, y, width, height: coordinates and dimensions of the bounding box
            - box_label: label of the bounding box
            - box_type: type of the bounding box (GT, TP, FP)
            - conf: confidence score of the predicted bounding box (for TP and FP)
            - iou: intersection over union of the predicted and ground truth bounding boxes
        n_rows: number of rows in the grid;
        n_cols: number of columns in the grid;
        output_path: the file path where the JPG files will be saved;
        total_width_mm: the total width of the entire multi-panel image in millimeters (default is 150 mm)

    Returns:
        Saves the image with the bounding boxes as JPG file.
    """
    
    # Determine the width of each panel (image)
    # Subtracts (n_cols - 1) mm for spacing
    panel_width_mm = (total_width_mm - (n_cols - 1)) / n_cols
    panel_width_inch = panel_width_mm / 25.4
    
    # Load the first image to determine the aspect ratio.
    # Note that this assumes that the images in the sequence have the same
    # aspect ratio.
    sample_image_path = df['path'].iloc[0]
    sample_image = cv2.imread(sample_image_path)
    image_height, image_width = sample_image.shape[:2]
    aspect_ratio = image_height / image_width
    
    # Calculate the height of each panel (image)
    panel_height_inch = panel_width_inch * aspect_ratio
    
    # Calculate the overall figure size in inches
    fig_width_inch = n_cols * panel_width_inch + (n_cols - 1) * (1 / 25.4)
    fig_height_inch = n_rows * panel_height_inch + (n_rows - 1) * (1 / 25.4)
    
    # Create a figure with a grid of subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_width_inch, fig_height_inch))
    axs = axs.flatten()  # Flatten the grid to easily iterate over it
    
    # Font settings for labels
    panel_label_font = {'family': 'sans-serif', 'size': 7}
    pred_font = {'family': 'sans-serif', 'size': 6.5}

    for i, ax in enumerate(axs):
        ax.axis('off')  # Turn off all axes (including labels and ticks)
    
    for idx, (ax, (img_path, dfi)) in enumerate(zip(axs, df.groupby('path'))):
        # Read the image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Image at path '{img_path}' could not be loaded.")
            continue
        # Convert BGR to RGB for correct display
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display the image
        ax.imshow(image)

        # Plot insect bounding boxes with different colors for GT, TP, FP
        for _, row in dfi.iterrows():
            x, y, width, height = row['x'], row['y'], row['width'], row['height']
            
            if row['box_type'] == 'GT':
                color = '#FFA500' # Orange
            elif row['box_type'] == 'TP':
                color = '#00FFFF' # Cyan 
            elif row['box_type'] == 'FP':
                color = '#800080' # Purple
            
            rect = plt.Rectangle((x, y), width, height, linewidth=1.5, 
                                 edgecolor=color, facecolor='none')
            ax.add_patch(rect)
        
        # Add panel label (e.g., "a)", "b)", etc.) to the upper left corner
        label = chr(97 + idx) + ")"  # Generates "a)", "b)", "c)", etc. 97=a, 98=b, 99=c, ...
        ax.annotate(label, xy=(0, 1), 
                    xycoords='axes fraction', 
                    fontsize=panel_label_font['size'], 
                    fontname=panel_label_font['family'], 
                    va='top', ha='left', color='black', 
                    bbox=dict(facecolor='white', edgecolor='none', pad=1))
        
        # Add timestamp from file name to the lower right corner of the image,
        # format hh:mm:ss extracted from the file name (column
        # filename_full_frame), which look like this
        # "*_IMG_yyyymmdd_hhmmss.jpg": "2021-07-06_Centaurea-scabiosa-01_IMG_0376.JPG"
        timestamp = row['filename_full_frame'].split('_')[-1].split('.')[0]
        timestamp = f"{timestamp[:2]}:{timestamp[2:4]}:{timestamp[4:]}"
        text = ax.text(0.98, 0.02, timestamp, 
                   transform=ax.transAxes, 
                   fontsize=pred_font['size'], 
                   fontname=pred_font['family'], 
                   va='bottom', ha='right', color='white')
        text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'), 
                               path_effects.Normal()])

        # Add info for TP boxes
        rows = dfi[dfi['box_type'] == 'TP'] 
        # There should be only one TP per image in this table
        # If there are more TPs per image, display warning:
        if len(rows) > 1:
            print(f"Warning: More than one TP found for image at path '{img_path}'.") 
        if not rows.empty:
            add_box_info(ax, rows, pred_font)
        
    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.15/25.4, hspace=0.2/25.4)  # 1 mm = 1/25.4 inches
    
    # Save as JPG 
    plt.savefig(output_path, format='jpg', bbox_inches='tight', pad_inches=0, dpi=300)
    

# Read the feather files with GTs, TPs and FPs. It was produced with the helper
# script fig_sequence_detection_helper.r
file_path = os.path.join(prj_path, 'data', 'processed', 'df_eval_seq_yolov5s_iou_0.1_interim.feather')
df_eval = pd.read_feather(file_path)

# Update path column.
img_dir_path = os.path.join(prj_path, 'data', 'images', 'cropped')
df_eval['path'] = df_eval['new_filename'].apply(lambda x: os.path.join(img_dir_path, x))

# Filter the dataframe to select only a given seq_id
seq_id = args.seq_id 
df = df_eval[df_eval['seq_id'] == seq_id]

# Print number of rows in df as this can help with setting the number of columns
# in the grid. Note that nr of rows is not indicating the nr of unique images in
# that sew, it indicates number and type of boxes in the sequence.
print(f"Number of box types (GT, TP, FP) in df_eval: {df.shape[0]}")

# Print number of unique images in the sequence
print(f"Number of unique images in the sequence: {df['path'].nunique()}")

# Print each file name, unique to the sequence
print(f"Unique file names in the sequence (cropped image names):")
print(df['new_filename'].unique())
print(f"Unique file names in the sequence (full-frame names):")
print(df['filename_full_frame'].unique())

# Make the figure
fig_filename = f'fig_sequence_detection_seq_id_{seq_id}.jpg'
output_path = os.path.join(prj_path, 'results', 'figures', fig_filename)
plot_images_sequence(df, n_rows=2, n_cols=3, 
                     output_path=output_path,
                     total_width_mm=150)

print(f"Figure saved at: \n{output_path}")