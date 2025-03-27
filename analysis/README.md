Code for analyzing the results from the best model, including data descriptors, visualizations, and statistical tests.

Only `prepare_data_for_analysis.r` needs to run first, because it prepares the data tables needed further for the other scripts.

- `prepare_data_for_analysis.r`:
    - this should run first; 
    - shapes the results for carrying the analysis, figures & tables; it also offers some data description;
- `data_descriptor.r`: 
    - computes the values that were reported in the manuscript when describing the dataset;
    - computes the values from Table 1 in the manuscript;
- `analysis_bbox_area_sharpness.r`:
    - Permutation tests to assess differences in bounding box area and image sharpness between instance localization and classification groups.
- `fig_nms_grid_search.r`:
    - Produces Figure 3 and computes the values presented in Supp. Table S1;
- `fig_boxplots_bbox_area_sharpness.r`:
    - Produces Figure 4;
- `fig_sequence_detection.py`:
    - Produces Figure 5;
    - execute first the helper script `fig_sequence_detection_helper.r` which shapes data needed for figures like those in Fig 5;
- `fig_boxplots_mimicry.r`:
    - Creates the multi-panel figure with distributions of misclassifications vs correctly classified cases for Diptera and Hymenoptera taxa groups;
- `binomial_test_pollinators.r`:
    - Binomial test to show that Hymenoptera - Diptera misclassifications within instances (independent frames) are significant;
    - Computes the values reported in Supp. Table S3;
- `mimicry_analysis_hymenoptera_as_diptera.r`:
    - Quantitative comparison of correctly classified Hymenoptera and those misclassified as Diptera;
    - Computes the values reported in Supp. Table S4;
- `mimicry_analysis_diptera_as_hymenoptera.r`:
    - Quantitative comparison of correctly classified Diptera and those misclassified as Hymenoptera;
    - Computes the values reported in Supp. Table S5;
    