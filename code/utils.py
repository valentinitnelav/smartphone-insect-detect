# Helper functions for data preparation and analysis.

import os
import json
import pandas as pd
import numpy as np
import math
from PIL import ImageOps, Image
import cv2
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor


def df_to_via_json(df, project_name, attribute_cols):
    """
    Convert a DataFrame to a JSON string in the VIA format.
    
    Args:
        df (pandas.DataFrame):
            DataFrame with the following columns: path, x, y, width, height, 
            and other attribute columns.
        project_name (str): 
            Name for the VIA project stored in the JSON file.
        attribute_cols (list of str): 
            List of column names that will be used to populate the
            region_attributes section of the VIA data structure.

    Returns: 
        dict:
            JSON dictionary in the VIA format.
    """

    via_data = {
        "_via_settings": {
            "ui": {
                "annotation_editor_height": 25,
                "annotation_editor_fontsize": 0.8,
                "leftsidebar_width": 18,
                "image_grid": {
                    "img_height": 80,
                    "rshape_fill": "none",
                    "rshape_fill_opacity": 0.3,
                    "rshape_stroke": "yellow",
                    "rshape_stroke_width": 2,
                    "show_region_shape": True,
                    "show_image_policy": "all"
                },
                "image": {
                    "region_label": "__via_region_id__",
                    "region_color": "__via_default_region_color__",
                    "region_label_font": "10px Sans",
                    "on_image_annotation_editor_placement": "NEAR_REGION"
                }
            },
            "core": {
                "buffer_size": 18,
                "filepath": {},
                "default_filepath": ""
            },
            "project": {
                "name": project_name
            }
        },
        "_via_img_metadata": {},
        "_via_attributes": {
            "region": {},
            "file": {}
        },
        "_via_data_format_version": "2.0.10",
        "_via_image_id_list": []
    }

    for _, row in df.iterrows():
        file_path = row['path']
        image_id = file_path
        if image_id not in via_data["_via_img_metadata"]:
            via_data["_via_img_metadata"][image_id] = {
                "filename": file_path,
                "size": -1,  # Set to -1 as size is not available in the DataFrame
                "regions": [],
                "file_attributes": {}
            }
            via_data["_via_image_id_list"].append(image_id)
        
        # Here, we create a dictionary for the region_attributes by iterating over
        # the list of attribute columns specified by the user
        region_attributes = {col: row[col] for col in attribute_cols}

        region = {
            "shape_attributes": {
                "name": "rect",
                "x": row['x'],
                "y": row['y'],
                "width": row['width'],
                "height": row['height']
            },
            "region_attributes": region_attributes
        }

        via_data["_via_img_metadata"][image_id]['regions'].append(region)

    return via_data


def via_json_to_df(json_file_path):
    """
    Convert a JSON file in the VIA format to a pandas.DataFrame.
    
    Args:
        json_file_path (str): Path to the JSON file
    
    Returns: 
        pandas.DataFrame:
            DataFrame with data from the VIA JSON file.
    """
    with open(json_file_path) as file:
        via_data = json.load(file)

    img_metadata = via_data["_via_img_metadata"]

    rows = []
    for img_path, metadata in img_metadata.items():
        for region_id, region in enumerate(metadata["regions"]):
            shape_attributes = region["shape_attributes"]
            region_attributes = region["region_attributes"]

            row = {
                "path": img_path,
                "region_id": region_id,
            }
            row.update(shape_attributes)
            row.update(region_attributes)
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def df_to_coco(df, img_path, label_mapping, single_class=False, include_metadata=True):
    """
    Converts a DataFrame with box annotations to a COCO format as dictionary.

    This function takes a pandas DataFrame containing image annotations (boxes)
    and a path to the image directory. It returns a dictionary in COCO format
    with information about the dataset, images, annotations, and categories.
    
    Note that certain custom column names are expected to appear in the DataFrame:
    - `ew_filename`: The name of the image file.
    - `width_crop`: The width of the cropped image.
    - `height_crop`: The height of the cropped image.
    - `date`: The date when the image was captured.
    - `p1_labels`: The label of the insect as per previous study.
    - `x`: The x-origin coordinate of the bounding box (in pixels).
    - `y`: The y-origin coordinate of the bounding box (in pixels).
    - `width`: The width of the bounding box (in pixels).
    - `height`: The height of the bounding box (in pixels).
    Metadata optional columns:
    - `seq_id`: The sequence/insect ID.
    
    Args:
        df (pandas.DataFrame): 
            The DataFrame with image annotations.
        img_path (str): 
            The directory path where images are stored.
        single_class (bool): 
            If True, all annotations are assigned to a single category.
        include_metadata (bool): 
            If True, additional metadata is included in the annotations.
        label_mapping (dict): 
            A dictionary that maps the original labels to their original ids.

    Returns:
        dict:
            A dictionary in COCO dataset format. This can be later saved to a JSON file.
    """

    # Initialize the COCO dataset structure
    data = {
        "info": {
            "year": 2025,
            "version": "1.0",
            "description": "Field images",
            "contributor": "Valentin Stefan",
            "url": ""
        },
        "images": [],
        "annotations": [],
        "categories": [],
        "dir_path": img_path
    }

    # Process images and generate image IDs
    image_ids = {}
    df_img = df.drop_duplicates(subset=['new_filename'])
    for i, (index, row) in enumerate(df_img.iterrows()):
        img = {
            "id": i + 1,
            "width": row['width_crop'],
            "height": row['height_crop'],
            "file_name": row['new_filename'],
            "license": "no license",
            "date_captured": str(row['date'])
        }
        data["images"].append(img)
        image_ids[row['new_filename']] = img['id']

    # Process labels and generate label IDs
    label_ids = {}
    categories_list = []

    # If single_class is True, we only have one category. The models will be
    # evaluated as an insect/ arthropod detector. 
    # Note that for COCO category_id = 0, "it is usually considered as having no label information
    # However, categories can have any integer id value, including 0"
    # https://github.com/openvinotoolkit/datumaro/issues/156
    # For single class use id 1; but note that YOLO label IDs start from 0 
    # and I passed them as such to the JSON file.
    if single_class:
        categories_list.append({"id": 1, "name": "arthropod"})
    else:
        # Process each unique label in the dataframe
        for label in sorted(df['p1_labels'].unique()):
            category_id = label_mapping.get(label, None)
            
            # If the label is not in the pre-defined mapping, issue a warning and skip
            if category_id is None:
                print(f"Warning: Label '{label}' not found in given label_mapping. Skipping label.")
                continue
            
            category = {"id": category_id, "name": label}
            categories_list.append(category)
            label_ids[label] = category_id

    # Add categories to data
    for category in categories_list:
        data["categories"].append(category)

    # Process annotations
    for i, (index, row) in enumerate(df.iterrows()):
        annotation = {
            "id": i + 1,
            "image_id": image_ids[row['new_filename']],
            # Set category_id based on single_class value
            "category_id": 1 if (single_class is True) else label_ids[row['p1_labels']],
            "bbox": [row['x'], row['y'], row['width'], row['height']],
            "area": row['width'] * row['height'],
            "segmentation": [],
            "iscrowd": 0
        }
        
        if (include_metadata is True):
            annotation["metadata"] = {
                "file_name": row['new_filename'],
                "sequence_id": row['seq_id'],
                # Below, use str(int()) to remove the decimal point if it is float. 
                # Also it avoids:
                # "ValueError: cannot convert float NaN to integer" 
                # and uses None if needed which is JSON-safe.
                "box_id": str(int(row['id_box'])) if not math.isnan(row['insect_id']) else None
            }
            
        data["annotations"].append(annotation)

    return data


def get_image_dim(row, path_col_name):
    """
    Get the dimensions of an image using PIL.
    
    Args:
        row (pandas.Series): 
            A row of a DataFrame containing the path to the image.
        path_col_name (str):
            The name of the column containing the image path.
    
    Returns: 
        pandas.Series:
            A pandas Series containing the image dimensions and image path.
    """
    path = row[path_col_name]
    
    try:
        img = Image.open(path)
        img_width_pil, img_height_pil = img.size
    except Exception as e:
        print(f"Error opening image {path}: {e}")
        img_width_pil = float("nan")
        img_height_pil = float("nan")
    
    try:
        # If you want the exif orientation then get the value by indexing the exif dictionary.
        # This dictionary sadly doesn't come with named keys, but you can directly use the index 274 for orientation.
        # See why index 274 at https://github.com/stark-t/PAI/issues/24
        # Or the official page for exif tags at https://exiv2.org/tags.html, 274 = Exif.Image.Orientation
        exif_dict = img._getexif()
        orientation = exif_dict[274]
    except Exception as e:
        print(f"Error getting EXIF data for {path}: {e}")
        orientation = float("nan")   
        
    results = pd.Series({
        'path': path,
        'img_width_pil': img_width_pil, 
        'img_height_pil': img_height_pil,
        'img_orientation': orientation
        })
    
    return results


def adjust_img_exif(img):
    """
    Adjusts the EXIF orientation.
    
    Useful links:
    - https://sirv.com/help/articles/rotate-photos-to-be-upright/
    - https://exif.readthedocs.io/en/latest/api_reference.html?highlight=orientation#exif.Orientation
    - https://exiv2.org/tags.html, 274 = Exif.Image.Orientation
    - https://exiftool.org/TagNames/EXIF.html
    
    Args:
        img (PIL.Image.Image):
            Image object opened using PIL via Image.open()
    
    Returns:
        PIL.Image.Image:
            Adjusted image.
    """
    exif_dict = img._getexif()
    if exif_dict:
        orientation = exif_dict[274]
        if orientation == 2:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            img = img.rotate(180)
        elif orientation == 4:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        elif orientation == 5:
            img = img.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            img = img.rotate(-90, expand=True)
        elif orientation == 7:
            img = img.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            img = img.rotate(90, expand=True)
    return img


def detect_extensions(folder_path):
    """
    Extracts file extensions for given folder path. 

    Args:
        folder_path (str): A given folder path.

    Returns:
        str: 
            File extensions.
    """
    extensions = {}
    for filename in os.listdir(folder_path):
        _, ext = os.path.splitext(filename)
        if ext in extensions:
            extensions[ext] += 1
        else:
            extensions[ext] = 1
    return extensions


def read_files(folder_path, extensions):
    """
    Reads files from a specified directory with specific provided extensions and
    returns a DataFrame containing file name, path and extension information.

    Args:
        folder_path (str):
            The path to the directory containing the files. 
        extensions (list of str):
            A list of file extensions to include, such as ['.jpg', '.png'].

    Returns:
        pandas.DataFrame:
            A DataFrame with columns 'file_name_no_ext', 'file_name', and
            'file_path', containing information about the files.
    """
    files_lst = []
    for file_name in os.listdir(folder_path):
        file_name_no_ext, ext = os.path.splitext(file_name)
        file_path = os.path.join(folder_path, file_name)
        if ext in extensions:
            files_lst.append([file_name_no_ext, file_name, file_path])
    cols = ['file_name_no_ext', 'file_name', 'file_path']
    df = pd.DataFrame(files_lst, columns=cols)
    return df



def read_yolo_txt(file_paths):
    """
    Reads YOLO format text files with predictions and returns a DataFrame.
    
    Args:
        file_paths (list of str): 
            A list of file paths to the YOLO prediction text files.
    
    Returns:
        pandas.DataFrame: 
            A DataFrame with prediction data.
    """
    data = []
    for file in file_paths:
        with open(file, 'r') as f:
            lines = f.read().strip().split('\n')
            for line in lines:
                values = line.split(' ')
                values.append(file)
                data.append(values)
                
    cols = ['yolo_label', 'x_center_rel', 'y_center_rel', 
            'width_rel', 'height_rel', 'yolo_conf', 'txt_path']
    df = pd.DataFrame(data, columns=cols)
    return df


def read_single_file(file):
    """
    Reads a single YOLO format text file and extracts prediction data.
    
    Args:
        file (str): 
            The path to the YOLO prediction text file to be read.
    
    Returns:
        list of list: 
            A list where each inner list contains the prediction data for one
            line of the file.
    """
    data = []
    with open(file, 'r') as f:
        lines = f.read().strip().split('\n')
        for line in lines:
            values = line.split(' ')
            values.append(file)
            data.append(values)
    return data


def read_yolo_txt_parallel(file_paths, n_workers=4):
    """
    Reads YOLO format text files in parallel and returns a DataFrame.
    
    Args:
        file_paths (list of str): 
            Paths to the YOLO prediction files.
        n_workers (int, optional): 
            Number of parallel processes (default is 4).
    
    Returns:
        pandas.DataFrame: 
            DataFrame with YOLO prediction data.
    """
    data = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(read_single_file, file_paths))
    
    for result in results:
        data.extend(result)
        
    cols = ['yolo_label', 'x_center_rel', 'y_center_rel', 
            'width_rel', 'height_rel', 'yolo_conf', 'txt_path']
    df = pd.DataFrame(data, columns=cols)
    return df


def compute_iou(box1, box2):
    """
    Computes the Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        box1 (tuple): 
            A tuple (x, y, w, h) representing the first bounding box.
        box2 (tuple): 
            A tuple (x, y, w, h) representing the second bounding box.
        
    Returns:
        float: 
            The IoU as a float between 0 and 1.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x_min = max(x1, x2)
    y_min = max(y1, y2)
    x_max = min(x1 + w1, x2 + w2)
    y_max = min(y1 + h1, y2 + h2)
    
    intersection_area = max(0, x_max - x_min + 1) * max(0, y_max - y_min + 1)
    # Why add 1: https://stackoverflow.com/a/58108241/5193830
    
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - intersection_area
    
    iou = intersection_area / union_area
    return iou


def apply_iou_roi(row):
    """
    Computes IoU between the ROI box and the insect box in a DataFrame row.
    
    Args:
        row (pandas.Series): 
            A DataFrame row containing the bounding box coordinates for the ROI
            and insect boxes.
    
    Returns:
        float: 
            IoU value in the range [0, 1].
    """
    box1 = [row['x'], row['y'], row['width'], row['height']]
    box2 = [row['x_roi'], row['y_roi'], row['width_roi'], row['height_roi']]
    return compute_iou(box1, box2)


   
def calculate_bounding_box_crop(df_group, dtype_dict):
    """
    Calculate a bounding box for cropping an image based on insect and ROI
    boxes.
    
    This function computes a bounding box that encloses all provided insect
    boxes and the ROI box within a given image. It only considers boxes from
    sequences where the `keep_seq` flag is True, indicating that the insect
    within the sequence of frames interacts at some point with the target
    flower. It also includes the `new_filename` corresponding to each `path`.
    
    Args:
        df_group (pandas.DataFrame):
            Filtered DataFrame group with 'keep_seq' True.
        dtype_dict (dict):
            Data types for the output DataFrame columns.
    
    Returns:
        pandas.DataFrame:
            Bounding box coordinates and 'new_filename'.
    """
    # Consider only the boxes from the sequences that touch the target flower
    df_group = df_group[df_group['keep_seq'] == True]
    
    if df_group.empty:
        return pd.DataFrame(columns=dtype_dict.keys())

    # Calculate the crop bounding box. This is the bounding box that contains
    # the bounding boxes within the given image, but only boxes from the
    # sequences that touch the target flower.
    x_min = min(df_group['x'].min(), 
                df_group['x_roi'].min())
    y_min = min(df_group['y'].min(), 
                df_group['y_roi'].min())
    x_max = max((df_group['x'] + df_group['width']).max(), 
                (df_group['x_roi'] + df_group['width_roi']).max())
    y_max = max((df_group['y'] + df_group['height']).max(), 
                (df_group['y_roi'] + df_group['height_roi']).max())
    
    # Get the first new_filename (cropped iamge) for this path (new_filename is
    # the same for all rows for that given full frame image path)
    new_filename = df_group['new_filename'].iloc[0]
    
    # Return the bounding box coordinates and the new_filename as a pandas DataFrame 
    return pd.DataFrame({
        'x': [x_min], 
        'y': [y_min], 
        'width': [x_max - x_min],
        'height': [y_max - y_min],
        'new_filename': [new_filename]
    }).astype(dtype_dict)


def check_out_of_bounds(row):
    """
    Check if bounding boxes exceed image boundaries for a DataFrame row.
    
    Args:
        row (pandas.Series):
            The DataFrame row with box and image dimensions.
        
    Returns:
        pandas.Series:
            With True/False.
    """
    width_out_of_bounds = row['x'] + row['width'] > row['img_width_pil']
    height_out_of_bounds = row['y'] + row['height'] > row['img_height_pil']
    
    return pd.Series({'width_out_of_bounds': width_out_of_bounds, 
                      'height_out_of_bounds': height_out_of_bounds})


def adjust_coordinates(row):
    """    
    Adjusts the coordinates of a insect bounding box based on a given crop area.
    
    This function calculates the intersection between an insect box and a crop
    area. If the insect box is entirely within the crop area, the coordinates
    are updated by subtracting the crop area coordinates from the insect box
    coordinates. Otherwise, it adjusts the coordinates of the insect box to
    match the intersection area (overlap) with the crop area. That is, the
    original insect box is cropped to the overlapping portion with the cropping
    box.
    
    Args:
        row (pd.Series): 
            A pandas Series representing a row of a DataFrame containing the
            bounding box information.
            The row should contain the following columns:
            - `x`: The x-coordinate of the bounding box.
            - `y`: The y-coordinate of the bounding box.
            - `width`: The width of the bounding box.
            - `height`: The height of the bounding box.
            - `x_crop`: The x-coordinate of the crop area.
            - `y_crop`: The y-coordinate of the crop area.
            - `width_crop`: The width of the crop area.
            - `height_crop`: The height of the crop area.

    Returns:
        None:
            Modifies `x`, `y`, `width`, and `height` in-place.
    """
    
    # Get the coordinates of the existing insect box    
    x = row['x']
    y = row['y']
    width = row['width']
    height = row['height']

    x_crop = row['x_crop']
    y_crop = row['y_crop']
    width_crop = row['width_crop']
    height_crop = row['height_crop']

    # Check if the insect box is already included within the crop area
    x_test = x_crop <= x <= x + width <= x_crop + width_crop
    y_test = y_crop <= y <= y + height <= y_crop + height_crop

    if x_test and y_test:
        row['x'] = x - x_crop
        row['y'] = y - y_crop
    else:
        # Calculate the intersection of the insect box and the crop box
        intersection_x = max(x, x_crop)
        intersection_y = max(y, y_crop)
        intersection_width = min(x + width, x_crop + width_crop) - intersection_x
        intersection_height = min(y + height, y_crop + height_crop) - intersection_y

        # Update the insect box coordinates to match the intersection
        row['x'] = intersection_x - x_crop
        row['y'] = intersection_y - y_crop
        row['width'] = intersection_width
        row['height'] = intersection_height

    return row


def visualize_image(df, path, crop=False, show_crop_bbx=True, show_roi=True):
    """
    Visualizes an image with bounding boxes of insects and optional cropping.

    Args:
        df (DataFrame): 
            Data frame containing the image and bounding box information.
        path (str): 
            Path to the image file.
        crop (bool, optional): 
            Flag indicating whether to crop the image based on the crop
            coordinates. Defaults to False.
        show_crop_bbx (bool, optional): 
            Flag indicating whether to show the crop bounding box. 
            Defaults to True.
        show_roi (bool, optional): 
            Flag indicating whether to show the region of interest bounding box.
            Defaults to True. 
    Returns:
        None: 
            This function displays the image with bounding boxes and does not
            return anything.
    """

    # Filter rows based on the image path
    image_rows = df[df['path'] == path]

    # Read the image
    image = cv2.imread(path)

    if crop:
        # Crop the image based on the crop coordinates
        x_crop, y_crop, width_crop, height_crop = image_rows.iloc[0]['x_crop'], image_rows.iloc[0]['y_crop'], \
                                                  image_rows.iloc[0]['width_crop'], image_rows.iloc[0]['height_crop']
        image = image[y_crop:y_crop+height_crop, x_crop:x_crop+width_crop]
    
    # Convert image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(5, 5))

    # Display the image
    ax.imshow(image)

    # Plot insect bounding boxes
    for index, row in image_rows.iterrows():
        x, y, width, height = row['x'], row['y'], row['width'], row['height']
        rect = plt.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    # Plot crop area
    if show_crop_bbx:
        x_crop, y_crop, width_crop, height_crop = image_rows.iloc[0]['x_crop'], image_rows.iloc[0]['y_crop'], \
                                                image_rows.iloc[0]['width_crop'], image_rows.iloc[0]['height_crop']
        rect_crop = plt.Rectangle((x_crop, y_crop), width_crop, height_crop, linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(rect_crop)

    # Plot region of interest
    if show_roi:
        x_roi, y_roi, width_roi, height_roi = image_rows.iloc[0]['x_roi'], image_rows.iloc[0]['y_roi'], \
                                            image_rows.iloc[0]['width_roi'], image_rows.iloc[0]['height_roi']
        rect_roi = plt.Rectangle((x_roi, y_roi), width_roi, height_roi, linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect_roi)

    plt.axis('off')
    plt.show()


def crop_image(row, dir_path):
    """
    Crops and saves an image using bounding box coordinates from a DataFrame row.
    
    Args:
        row (pandas.Series): 
            A DataFrame row containing image path, cropping coordinates and
            `new_filename`.
        dir_path (str): 
            Directory path where the cropped image will be saved.
    
    Effects:
        Modifies `error_paths` global list if an error occurs during image
        processing.
    """
    global error_paths
    try:
        img = Image.open(row['path'])
    except Exception as e:
        print(f"An error occurred while opening the image: {e}")
        error_paths.append(row['path'])
        return
    try:
        # Orientation has to be considered to do the cropping accordingly.
        # See also https://sirv.com/help/articles/rotate-photos-to-be-upright/
        img = adjust_img_exif(img)
        # define the area to crop: (x, y, x + width, y + height)
        box = (row['x'], row['y'], row['x'] + row['width'], row['y'] + row['height'])
        cropped_img = img.crop(box)
    except Exception as e:
        print(f"An error occurred while adjusting the image based on orientation: {e}")
        error_paths.append(row['path'])
        return
    try:
        cropped_img.save(os.path.join(dir_path, row['new_filename']))
    except Exception as e:
        print(f"An error occurred while saving the image: {e}")
        error_paths.append(row['path'])
        return
