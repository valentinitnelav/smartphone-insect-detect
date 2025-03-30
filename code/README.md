This contains various scripts and notebooks needed for intermediary processes to arrive to model evaluation.

The code was executed in the following order:

- Detections on the OOD image dataset
    - download the YOLO trained weights; see detectors/weights/README.md & `download_weights.sh`;
    - `detect.sh`: runs `detect.py` of YOLOv5 and YOLOv7 detectors
        - can be time consuming when runs with a grid search for NMM-IoU and NMS-conf set at 0.001; used on a GPU cluster;
        - when the optimal NMS conf and IoU were identified for a model (see `coco_eval_check_results.ipynb` below)
            after running the NMS-IoU grid search at NMS-conf 0.001, `detect.sh` can be executed again with those
            optimal values to generate the final detection results.
        - calls `session_info.sh` at run time.
    - `bind_yolo_txt_files.sh`: binds the default YOLO detection results (individual txt files) into a single txt file.
        - can run for a given model at given NMS-conf and NMS-IoU vales
    - `yolo_txt_to_coco.py`: converts the YOLO detection results (txt file) to COCO format (JSON file).
        - the txt files needed as input are the ones generated with `bind_yolo_txt_files.sh`;
        - the COCO format as a JSON file is needed for the evaluation with `pycoctools` library.
- Model evaluation on the OOD image dataset
    - `coco_eval.py`: class-agnostic evaluation with the `pycoctools` library
    - `coco_eval_check_results.ipynb`: to view some of the results of the evaluation
    - `model_performance.ipynb`: 
        - Evaluates model performance across frames (localization and classification metrics);
        - Computes values presented in Table 2;
        - Evaluation eval-IoU is set to 0.5 (rerun with 0.1 to get value presented in Supp. Table S2).
- Prepare JSON file for VIA toll to visualize predictions with ground truths:
    - `./data/code/make_via_json_predictions.py`
- Run detections with the NMS-optimized model on background images to get FP estimates on such cases:
    - `detect_on_backgrounds.sh`
- SAHI - [Slicing Aided Hyper Inference](https://docs.ultralytics.com/guides/sahi-tiled-inference/)
    - Note that this was an not a main goal of model evaluation.
    - see `./detectors/sahi/README.md`
- Helper functions:
    - `utils.py`: Python helper functions used in the scripts and notebooks across this repository.
    - `model_performance_utils.py`: Python helper functions used in `model_performance.ipynb` 