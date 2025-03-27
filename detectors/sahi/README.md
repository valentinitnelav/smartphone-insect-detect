# SAHI

SAHI - [Slicing Aided Hyper Inference](https://docs.ultralytics.com/guides/sahi-tiled-inference/)

Note that this was an not a main goal of model evaluation. SAHI was applied to touch on a discussion paragraph in the manuscript about the possibility of using it to annotate datasets, but most probably this cannot be deployed in real time on a custom camera in field conditions.

Order of executing the code:

1. `get_img_without_tp.py`: 
    - SAHI was executed only on the images where the NMS optimized model (YOLOv5s) failed to detect insect instances. This scripts filters for the images without any TPs (at IoU 0.5) and store the corresponding annotations for further processing with SAHI.
2. `sahi_predict.sh`: run SAHI on the images filtered out with the previous script