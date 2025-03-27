# Overview

This repository contains the code repository associated with:

> Ștefan V., Stark T., Wurm M., Taubenböck H., Knight T.M. (2025). Successes and limitations of pretrained YOLO detectors applied to unseen time-lapse images for automated pollinator monitoring.

# Data

The image dataset is stored on Zenodo with a DOI at:

> 

Check `./data/README.md` for more information on the data processing steps.

# Scripts and notebooks

The scripts and notebooks needed for this study are distributed across several locations in this repository, as follows:

- `./data/code/`: data (ground truth) processing. See `./data/README.md` for more information;
- `./code/`: intermediary processes from data preparation to model evaluation. See `./code/README.md`;
- `./analysis/`: analysis of the results given by the best model, data descriptors, visualizations, and statistical tests. See `./analysis/README.md`.

# Reproducibility & Virtual Environments

For Pyhton & R environments, refer to './envs/README.md'.

NOTE: The project was developed on Linux using open-source software and libraries. 
Windows and Mac users may need to adjust commands accordingly.