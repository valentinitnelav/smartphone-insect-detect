# Prepare the data tables needed for analysis. One table is for instances (each
# instance is a row; bounding box = insect instance). And one table for the
# insect individuals (sequences), which aggregates data across the frames
# corresponding to that individual.


# Setup -------------------------------------------------------------------

library(data.table)
library(magrittr)
library(arrow)

# Read data:
# Evaluation results produced with ./code/model_performance.ipynb 
dt <- read_feather('./data/processed/df_eval_seq_yolov5s_iou_0.5.feather') %>% setDT()


# Data preparation --------------------------------------------------------

# Compute sharpness as relative to the max - max normalization. The sharpest is 1.
dt[, score_sobel := score_sobel / max(score_sobel)]

# Create a taxa "identifier" for each instance
dt[, taxa_combi := paste(order, suborder, infraorder, superfamily, family, 
                         clustergenera, genus, morphospecies, species, 
                         sep = '_')]

# Was the instance (box of an insect) detected?
dt[, box_detected := ifelse(box_type == "TP", TRUE, FALSE)]
dt[, .N, keyby = box_detected][, prc := round(N/sum(N) * 100, 2)][]
#    box_detected     N   prc
# 1:        FALSE 10002 40.57
# 2:         TRUE 14654 59.43

dt[, box_detected_factor := factor(box_detected, 
                                   levels = c(FALSE, TRUE), 
                                   labels = c("no", "yes"))] 

# Was the individual insect (sequence) detected?
dt[, seq_detected := any(!is.na(pred_iou)), by = seq_id]
dt[, .N, keyby = seq_detected][, prc := round(N/sum(N) * 100, 2)][]
#    seq_detected     N  prc
# 1:        FALSE  2861 11.6
# 2:         TRUE 21795 88.4

# If across a sequence the insect label matches the predicted label with the
# highest YOLO confidence score, then the sequence is correctly classified.
dt[, seq_label_matched := p1_labels == common_label]
dt[, .N, by = seq_label_matched]
#    seq_label_matched     N
#               <lgcl> <int>
# 1:              TRUE 17558
# 2:             FALSE  4237
# 3:                NA  2861

# There are NA values in pred_label. Keep those NAs because it indicate that
# instances (boxes) were not localized and there is no classification to check.
# Classification will be assesed only for localized boxes.
dt[, .N, keyby = label_matched] # computed in Python already (evaluate_sequences.ipynb)
#    label_matched     N
# 1:            NA 10002 # boxes not localized and therefore not classified
# 2:             0  2783 # FALSE
# 3:             1 11871 # TRUE

dt[!is.na(label_matched), .N, keyby = label_matched][, prc := round(N/sum(N) * 100, 2)][]
#    label_matched     N   prc
# 1:             0  2783 18.99
# 2:             1 11871 81.01

# This will aggregate across the boxes for an individual (sequence)
dt_seq <- dt[, .(
  n_box = .N,
  p1_labels = paste(unique(p1_labels), collapse = "; "),
  label_gt_groups = paste(unique(true_group), collapse = "; "),
  pred_arthropod = paste(unique(common_label), collapse = "; "),
  n_taxa_combi = uniqueN(taxa_combi),
  taxa_combi = paste(unique(taxa_combi), collapse = "; "),
  order = paste(unique(order), collapse = "; "),
  suborder = paste(unique(suborder), collapse = "; "),
  infraorder = paste(unique(infraorder), collapse = "; "),
  superfamily = paste(unique(superfamily), collapse = "; "),
  family = paste(unique(family), collapse = "; "),
  clustergenera = paste(unique(clustergenera), collapse = "; "),
  genus = paste(unique(genus), collapse = "; "),
  morphospecies = paste(unique(morphospecies), collapse = "; "),
  species = paste(unique(species), collapse = "; "),
  seq_detected = paste(unique(seq_detected), collapse = "; ") %>% as.logical(),
  seq_label_matched = paste(unique(seq_label_matched), collapse = "; ")  %>% as.logical(),
  seq_id = paste(unique(seq_id), collapse = "; "),
  mean_bbx_rel_area = mean(box_area_rel, na.rm = TRUE),
  mean_sobel = mean(score_sobel, na.rm = TRUE)
),
keyby = seq_id]

# Convert "NA" (character) to NA representation. This character "NA" happened
# because of the paste() operations above.
str(dt_seq)
dt_seq[dt_seq == "NA"] <- NA

dt_seq[, n_seq_family := .N, by = family]
dt_seq[, n_seq_taxa_combi := .N, by = taxa_combi]

# Check if there are multiple taxa_combi per sequence (individual)
dt_seq[n_taxa_combi > 1, .N] # expect 0


# Save results ------------------------------------------------------------

# Save as feather and rds binary files. The feather one is compatible with
# pandas in Python.

write_feather(dt, './data/processed/dt_analysis.feather')
saveRDS(dt, './data/processed/dt_analysis.rds')

write_feather(dt_seq, './data/processed/dt_analysis_seq_individs.feather')
saveRDS(dt_seq, './data/processed/dt_seq_analysis_seq_individs.rds')
