# Supp. Table S5. Quantitative comparison of correctly classified Diptera and
# those misclassified as Hymenoptera.
# This analysis grouped Syrphidae vs Non-Syrphidae.

library(data.table)
library(magrittr)
library(coin)

# Load the permutation_test() function. Performs a permutation test to compare
# the difference in means or medians between two samples.
source('./analysis/utils.r')

# Read data.
# Data tables produced with analysis_prep.r
dt <- readRDS('./data/processed/dt_analysis.rds') # instances (one row one box)
dt_seq <- readRDS('./data/processed/dt_seq_analysis_seq_individs.rds') 
# dt_seq: aggregated instances to individual level


# Error rate tests ------------------------------------------------------

dt_seq[p1_labels == "diptera", .N] # 145 Diptera individuals as ground truth

# Localized vs non-localized
dt_seq[p1_labels == "diptera", .N, keyby = seq_detected]
#    seq_detected     N
# 1:        FALSE    28
# 2:         TRUE   117

dt_seq[p1_labels == "diptera", .N, keyby = seq_label_matched]
# NA = not localized (nothing to classify if it is not localized)
#    seq_label_matched    N
# 1:                NA   28
# 2:             FALSE   20
# 3:              TRUE   97


gr_cols <- c("suborder", "infraorder", "superfamily", "family", "clustergenera", 
             "genus", "species")

# Diptera miscclasified as Hymenoptera
dt_mis_not_aggreg <- dt_seq[(p1_labels == "diptera") & (pred_arthropod == "hymenoptera"), 
                            .(
                              taxa_combi = paste(unique(taxa_combi), collapse = "| "),
                              n_mis_target = .N # nr individuals misclassified as hymenoptera
                            ),
                            keyby = gr_cols]
dt_mis_not_aggreg[order(-n_mis_target)]
dt_mis_not_aggreg[, sum(n_mis_target)] # 11

# Group Syrphidae vs coarsly identified Diptera
dt[p1_labels == "diptera" & (is.na(family) | family != "syrphidae"), category := "coarse_id"]
dt[p1_labels == "diptera" & family == "syrphidae", category := "syrph"]
# Check if grouping is correct
dt[p1_labels == "diptera", .N, keyby = category]
dt[is_max_conf_row == TRUE & p1_labels == "diptera" & common_label == "hymenoptera", .N, 
   keyby = .(order, suborder, infraorder, family, clustergenera, genus, species, category)]

# Apply grouping labels to aggregated data (individuals) frame too
dt_seq[p1_labels == "diptera" & (is.na(family) | family != "syrphidae"), category := "coarse_id"]
dt_seq[p1_labels == "diptera" & family == "syrphidae", category := "syrph"]
# Check if grouping is correct
dt_seq[p1_labels == "diptera", .N, keyby = category]
dt_seq[p1_labels == "diptera" & pred_arthropod == "hymenoptera", .N, 
       keyby = .(order, suborder, infraorder, family, clustergenera, genus, species, category)]

dt_seq[p1_labels == "diptera", .N] # 145 Diptera individuals (n individuals ground truth)
dt_seq[p1_labels == "diptera", n_ind := uniqueN(seq_id), by = category] 

# Diptera miscclasified as Hymenoptera by category (grouping labels).
# Compute the nr. of individuals misclassified as Hymenoptera
dt_mis <- dt_seq[(p1_labels == "diptera") & (pred_arthropod == "hymenoptera"), 
                 .(n_mis_target = .N),
                 keyby = category]
dt_mis


# Stats for all taxa categories of Diptera
dt_w_cor <- dt_seq[p1_labels == "diptera", 
                   .(
                     # N individuals misclassified as Hymenoptera
                     n_mis_target = sum(pred_arthropod == "hymenoptera", na.rm = TRUE),
                     n_ind = uniqueN(seq_id), # N individuals ground truth
                     n_loc = sum(seq_detected == TRUE), # N localized individuals
                     # Correctly classified (from those localized, so avoid counting NAs)
                     n_cor = sum(seq_label_matched == TRUE, na.rm = TRUE),
                     # All misclassifications within the localized individuals
                     n_mis_loc = sum(seq_label_matched == FALSE, na.rm = TRUE),
                     # These are all misclassifications, not just as
                     # hymenoptera, but also not localized individuals
                     n_mis_incl_not_loc = sum((seq_label_matched == FALSE) | is.na(seq_label_matched)) 
                   ), 
                   keyby = category]
dt_w_cor
dt_w_cor[, sum(n_ind)]              # 145 total Diptera individuals
dt_w_cor[, sum(n_cor)]              #  97 correctly classified
dt_w_cor[, sum(n_mis_loc)]          #  20 misclasifications (not just Hymenoptera) within the localized individuals
dt_w_cor[, sum(n_mis_incl_not_loc)] #  48 misclasifications (not just Hymenoptera), including those not localized

# Update dt_mis
dt_mis[dt_w_cor, on = "category", 
       `:=` (n_ind = i.n_ind,
             n_loc = i.n_loc,
             n_cor = i.n_cor, 
             n_mis_loc = i.n_mis_loc, 
             n_mis_incl_not_loc = i.n_mis_incl_not_loc)]

# Nr of individuals misclassified as something else than Hymenoptera, per taxa
# category.
dt_mis[, n_mis_other := n_mis_loc - n_mis_target]
# Proportion of misclassified individuals (as Hymenoptera) from total
# misclassifications including not localized, per taxa categ.
dt_mis[, prop_mis_from_mis_incl_not_loc := round(n_mis_target / n_mis_incl_not_loc * 100, 2)]
# Proportion of misclassified individuals (as Hymenoptera) from total
# misclassifications without not localized, per taxa categ.
dt_mis[, prop_mis_from_mis_loc := round(n_mis_target / n_mis_loc * 100, 2)]
# Proportion of misclassified individuals (as Hymenoptera) from total
# individuals, per taxa categ.
dt_mis[, prop_mis_from_nind := round(n_mis_target / n_ind * 100, 2)]
# Proportion of misclassified individuals (as Hymenoptera) from overall total
# individuals misclassified as Hymenoptera.
dt_mis[, prop_mis := round(n_mis_target / dt_mis[, sum(n_mis_target)] * 100, 2)]


# ~ Binomial tests --------------------------------------------------------

# Error rate - misclassifications to Hymenoptera from all misclassifications
# (that were localized)
dt_mis[, sum(n_mis_target)] # 11 total misclassifications to Hymenoptera from localized ones
dt_w_cor[, sum(n_mis_loc)]  # 20 total misclassifications as anything from localized ones
pp <- dt_mis[, sum(n_mis_target)] / dt_w_cor[, sum(n_mis_loc)] 
round(pp * 100, 2) # 55.00

# Test whether Hymenoptera misclassifications occur more than expected by
# chance. So, test whether the observed rate of “success” (error rate in my
# case) differs from 50%, which you can interpret as a “by chance alone”
# scenario (i.e., no bias toward Hymenoptera or Other).
p_expected <- 0.5

# Compute the number of expected miscclasifications for each investigated
# category
dt_mis[, n_mis_expect := round(p_expected * n_mis_loc, 0)]
dt_mis

# Perform binomial tests for each investigated category and store p-values
dt_mis[, p_binomial := binom.test(x = n_mis_target, 
                                  n = n_mis_loc, 
                                  p = p_expected, 
                                  alternative = "greater")$p.value %>% round(digits = 4), 
       by = .I]

# Calculate adjusted p-values
dt_mis[, `:=` (
  p_bi_bonferroni = p.adjust(p_binomial, method = "bonferroni") %>% round(digits = 4),
  p_bi_holm = p.adjust(p_binomial, method = "holm") %>% round(digits = 4),
  p_bi_bh = p.adjust(p_binomial, method = "BH") %>% round(digits = 4)
)]

# Add significance flags
dt_mis[, `:=` (
  is_signif_p_bi = ifelse(p_binomial < 0.05, "yes", "no"),
  is_signif_p_bi_bonferroni = ifelse(p_bi_bonferroni < 0.05, "yes", "no"),
  is_signif_p_bi_holm = ifelse(p_bi_holm < 0.05, "yes", "no"),
  is_signif_p_bi_bh = ifelse(p_bi_bh < 0.05, "yes", "no")
)]

dt_mis


# Size, blur & YOLO conf. for correctly classified vs. misclassified --------------

# NOTE: Only localized cases where discussed. Those that were not localized can
# also be considered "not classified", but for the analysis below the focus was
# on localized and misclassified.

# Filter only for those boxes that had the maximum confidence as they define the
# predicted label for the entire sequence.
dt[is_max_conf_row == TRUE, .N]  # 1110 Total number of localized insects 
dt_seq[seq_detected == TRUE, .N] # 1110 expect same as above

dt_max <- dt[is_max_conf_row == TRUE]
dt_max[, .N] # 1110
dt_max[p1_labels == "diptera", .N] # 117 localized Diptera individuals out of 
# 145 total Diptera
dt_seq[p1_labels == "diptera", .N] # 145 total Diptera individuals


# For the misclassified, get their ids and file paths for visual
# inspection. Note that this only includes localized but misclassified
# cases and only those images from a sequence where is_max_conf_row == TRUE.
dt_to_inspect <- dt_max[p1_labels == "diptera" & common_label == "hymenoptera", 
                        .(n_ind = .N,
                          seq_ids = paste(unique(seq_id), collapse = " "),
                          file = paste(unique(new_filename), collapse = " ")), 
                        keyby = gr_cols][order(-n_ind)]


# Table with correctly classified instances (those with max YOLO confidence from
# a sequence belonging to an individual insect)
dt_max_cor <- dt_max[p1_labels == "diptera" & common_label == "diptera"]

# Descriptive stats for the misclassified cases of Diptera as Hymenoptera
dt_max_mis <- dt_max[p1_labels == "diptera" & common_label == "hymenoptera"]
nrow(dt_max_mis) # expect 11 Diptera misclassified as Hymenoptera
dt_max_mis[, .N, keyby = category][order(-N)]
# The .N above will work here as nr of ind and not nr of boxes because of the
# filter is_max_conf_row

# Build a data table that will store all the quantitative comparisons and test
# results. This will gradually be populated with values across this script.
# Below it starts with descriptive stats about the misclassifications.
# This table will be shaped into a more readable form at the end of the script.
dt_compare <- dt_max_mis[, .( n_mis = .N,
                              mean_box_area = mean(box_area_rel) %>% round(digits = 4),
                              sd_box_area = sd(box_area_rel) %>% round(digits = 4),
                              mean_sobel = mean(score_sobel) %>% round(digits = 4),
                              sd_sobel = sd(score_sobel) %>% round(digits = 4),
                              mean_score = mean(pred_conf) %>% round(digits = 4),
                              sd_score = sd(pred_conf) %>% round(digits = 4),
                              med_box_area = median(box_area_rel) %>% round(digits = 4),
                              med_sobel = median(score_sobel) %>% round(digits = 4),
                              med_score = median(pred_conf) %>% round(digits = 4) ), 
                         keyby = category][order(-n_mis)]

dt_compare


# ~ Permutation tests -----------------------------------------------------

# Apply permutation tests for each investigated category to check if differences
# between correctly classified and misclassified are each significant or not.

# Variables of interest in dt_compare
categories <- dt_compare$category
var_vector <- c("box_area_rel", "score_sobel", "pred_conf")
metrics <- c("mean", "median")

# Note that the permutation_test() function applied with
# apply_permutation_test() returns the difference as sample2 - sample1, e.g.:
# permutation_test(2, 1, n_sim = 1, stat = "mean")$observed_difference 
# returns -1 for observed_difference
dt_compare <- apply_permutation_test(categories, var_vector, metrics, dt_max_mis, 
                                     dt_max_cor, dt_compare, n_sim = 1000)
dt_compare[]


# Prepare results for appendix table --------------------------------------

# Merge tables based on binding the category (it is unique, and must be ordered
# the same)
setorder(dt_compare, category)
setorder(dt_mis, category)

dt_results <- cbind(dt_mis, dt_compare[, !c("category", "n_mis")])
setorder(dt_results, -n_mis_target)

# Transpose
dt_transposed <- melt(dt_results, 
                      id.vars = "category", 
                      variable.name = "variables")
# Warning message expected because of mixture od character and numeric values.
# All numeric will be converted to character, but this is ok for the appendix
# table purpose.
dt_transposed <- dt_transposed[, dcast(.SD, variables ~ category, value.var = "value")]

order_for_cols <- dt_compare[order(-n_mis), category]
setcolorder(dt_transposed, c("variables", order_for_cols))


# Define renaming & ordering in the mapping vector for the rows in the table.
# Note to self: if you move this in a txt file (a data frame maybe), note that
# some are specific for Hymenoptera analysis and some for Diptera. So you would
# need one temporary file for each script.
rename_map <- c(
  "n_ind" = "N. individuals in dataset",
  "n_loc" = "N. localised",
  "n_cor" = "N. correctly classified",
  "n_mis_loc" = "N. misclassified, total (from those localised)",
  "n_mis_target" = "N. misclassified as Hymenoptera",
  "n_mis_other" = "N. misclassified as Other",
  "prop_mis_from_mis_loc" = "% misclassified as Hymenoptera from #4",
  "n_mis_expect" = "N. misclassified as Hymenoptera, expected",
  "p_binomial" = "p-value, binomial test for #5 & #7",
  "is_signif_p_bi" = "p-value, is significant, for #9 (< 0.05?)",
  "mean_box_area" = "Mean relative b.box area for #5",
  "sd_box_area" = "S.D. for #11",
  "mean_box_area_rel_cor" = "Mean relative b.box area for #3",
  "sd_mean_box_area_rel_cor" = "S.D. for #13",
  "mean_box_area_rel_dif_cor_mis" = "Difference means #13 - #11",
  "mean_box_area_rel_QuantPermCI" = "Permutation quantile conf. interval (C.I.) for #15",
  "mean_box_area_rel_dif_p_val" = "p-value, permutations for #15",
  "mean_box_area_rel_dif_p_val_is_singif" = "p-value is significant, for #15 (< 0.05?)",
  "med_box_area" = "Median relative b.box area for #5",
  "median_box_area_rel_cor" = "Median relative b.box area for #3",
  "median_box_area_rel_dif_cor_mis" = "Difference medians #20 - #19",
  "median_box_area_rel_QuantPermCI" = "Permutation quantile conf. interval (C.I.) for #21",
  "median_box_area_rel_dif_p_val" = "p-value, permutations for #21",
  "median_box_area_rel_dif_p_val_is_singif" = "p-value is significant, for #21 (< 0.05?)",
  "mean_sobel" = "Mean normalised sharpness for #5",
  "sd_sobel" = "S.D. for #25",
  "mean_score_sobel_cor" = "Mean normalised sharpness for #3",
  "sd_mean_score_sobel_cor" = "S.D. for #27",
  "mean_score_sobel_dif_cor_mis" = "Difference means #27 - #25",
  "mean_score_sobel_QuantPermCI" = "Permutation quantile conf. interval (C.I.) for #29",
  "mean_score_sobel_dif_p_val" = "p-value, permutations for #29",
  "mean_score_sobel_dif_p_val_is_singif" = "p-value significant, for #29 (< 0.05?)",
  "med_sobel" = "Median normalised sharpness for #5",
  "median_score_sobel_cor" = "Median normalised sharpness for #3",
  "median_score_sobel_dif_cor_mis" = "Difference medians #34 - #33",
  "median_score_sobel_QuantPermCI" = "Permutation quantile conf. interval (C.I.) for #35",
  "median_score_sobel_dif_p_val" = "p-value, permutations for #35",
  "median_score_sobel_dif_p_val_is_singif" = "p-value is significant, for #35 (< 0.05?)",
  "mean_score" = "Mean YOLO confidence for #5",
  "sd_score" = "S.D. for #39",
  "mean_pred_conf_cor" = "Mean YOLO confidence for #3",
  "sd_mean_pred_conf_cor" = "S.D. for #41",
  "mean_pred_conf_dif_cor_mis" = "Difference means #41 - #39",
  "mean_pred_conf_QuantPermCI" = "Permutation quantile conf. interval (C.I.) for #43",
  "mean_pred_conf_dif_p_val" = "p-value, permutations for #43",
  "mean_pred_conf_dif_p_val_is_singif" = "p-value is significant, for #43 (< 0.05?)",
  "med_score" = "Median YOLO confidence for #5",
  "median_pred_conf_cor" = "Median YOLO confidence for #3",
  "median_pred_conf_dif_cor_mis" = "Difference medians #48 - #47",
  "median_pred_conf_QuantPermCI" = "Permutation quantile conf. interval (C.I.) for #49",
  "median_pred_conf_dif_p_val" = "p-value, permutations for #49",
  "median_pred_conf_dif_p_val_is_singif" = "p-value is significant, for #49 (< 0.05?)",
  "box_area_percentile_rank_mis" = "Rank for #11",
  "box_area_percentile_rank_cor" = "Rank for #13",
  "sobel_percentile_rank_mis" = "Rank for #25",
  "sobel_percentile_rank_cor" = "Rank for #27",
  "conf_percentile_rank_mis" = "Rank for #39",
  "conf_percentile_rank_cor" = "Rank for #41"
)

# Filter to keep only variables in rename_map (drops unlisted ones)
dt_supp <- dt_transposed[variables %in% names(rename_map)]
str(dt_supp)
dt_supp[, variables := as.character(variables)]

# Rename the variables column using the values from rename_map
dt_supp[, Variables := rename_map[variables]]

# Reorder rows based on this order
dt_supp <- dt_supp[order(match(variables, names(rename_map)))]

dt_supp[, "variables" := NULL]
setcolorder(dt_supp, c("Variables", setdiff(names(dt_supp), "Variables")))

dt_supp[, "#" := 1:.N]
setcolorder(dt_supp, c("#", setdiff(names(dt_supp), "#")))

dt_supp

fwrite(dt_supp, "./results/tables/Supp_Table_analysis_Diptera_misclassified_as_Hymenoptera.csv")
