# Data processing code

The following scripts and jupyter notebooks (in `./data/code/`) were executed in the order below to process the data.

Some of the first steps (like 1, 2 & 3) are optional if you do not want to process the full frames, but instead want to use the provided processed OOD cropped dataset directly. These steps were offered for full transparency and reproducibility of our study, showing how the raw data was processed to arrive to the OOD dataset that is the subject of the evaluation and analysis pipeline presented in our paper.

1. `make_via_json_cropped_frames.py`: prepares the VIA JSON files with the annotations for the full frames.
2. `insect_roi_cropping_and_coordinate_adjustment.ipynb`: prepares towards the OOD annotation dataset from the full frames.
3. `crop_full_frames.py`: crops the full frame images to arrive to the OOD images.
4. `make_via_json_cropped_frames.py`: prepares the VIA JSON files with the annotations for the cropped frames.
5. `compute_sharpness.py`: 
    - computes the sharpness metric of the pixels within the insect boxes and appends them to the OOD annotation dataset.
6. `ground_truths_to_coco.py`:
    - Converts the ground truth annotations in tabular format (from a Feather file) to COCO format (JSON file).
    - The COCO format is needed for the evaluation run with `./code/coco_eval.py`.
7. `make_via_json_predictions.py`: prepares the VIA JSON files with the ground truth boxes and also the predictions of the "best" model.
    - note that this script should be run after the predictions are computed. See `./code/README.md` for more details.

