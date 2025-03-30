# Overview

This repository hosts the source code accompanying the research paper:

> Ștefan V., Stark T., Wurm M., Taubenböck H., Knight T.M. (2025). Successes and limitations of pretrained YOLO detectors applied to unseen time-lapse images for automated pollinator monitoring. *(Preprint)*

# Data

The image dataset is stored on Zenodo at:

> Ştefan, V., Workman, A., Cobain, J. C., Rakosy, D., Wild Stoykova, B., Cyranka, E., Urrego Álvarez, R., & Knight, T. (2025). Dataset of arthropod flower visits captured via smartphone time-lapse photography [Data set]. Zenodo. https://doi.org/10.5281/zenodo.15096610

Check `./data/README.md` for more information on the data processing steps.

# Scripts and notebooks

Scripts and notebooks are organized as follows:

1. `./data/code/`: Ground truth data processing. See `./data/README.md`;
2. `./code/`: intermediary processes to model evaluation. See `./code/README.md`;
3. `./analysis/`: analysis of the results produced by the optimized detector, along with data descriptors, visualizations, and statistical tests. See `./analysis/README.md`.

# Reproducibility & Virtual Environments

For Pyhton & R environments, see './envs/README.md'.

This project was developed on Linux using open-source software for the following reasons:

- GPU Compatibility: The GPU workstation and cluster we had access to run on Linux,
- Costs & Open-Source benefits: Linux is free and open-source, enhancing accessibility and transparency. This aligns with our project's commitment to open science and the [FAIR principles][1], allowing others to engage with, review, build upon, and redistribute our work.

[1]: https://www.go-fair.org/fair-principles/