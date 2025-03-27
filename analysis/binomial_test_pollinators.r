# Binomial test to show that Hymenoptera - Diptera misclassifications within
# instances (independent frames) are significant.
# Computes the values reported in Supp. Table S3. 

library(data.table)
library(magrittr)
library(stringr)


# Data table produced with prepare_data_for_analysis.r
# These are all the ground truths with their predictions.
dt_gt <- readRDS('./data/processed/dt_analysis.rds')

# Prepare data for test
dt_gt_test <- dt_gt[p1_labels %in% c("diptera", "hymenoptera"), 
                    .(n_mis = .N), 
                    keyby = .(p1_labels, pred_label, label_matched)]

dt_gt_test_misclsf <- dt_gt_test[label_matched == 0]
dt_gt_test_misclsf[, label_matched := NULL]
# Sum of all misclassified cases per group
dt_gt_test_misclsf[, n_mis_sum := sum(n_mis), by = p1_labels]
dt_gt_test_misclsf

# Example of binom.test on a particular case: diptera misclassified as hymenoptera
n_mis <- dt_gt_test_misclsf[p1_labels == "diptera" & pred_label == "hymenoptera", n_mis]
n_mis_sum <- dt_gt_test_misclsf[p1_labels == "diptera" & pred_label == "hymenoptera", n_mis_sum]
my_test <- binom.test(x = n_mis, n = n_mis_sum, p = 1/7, alternative = "greater")
my_test
# umber of successes = 303, number of trials = 467, p-value < 2.2e-16
my_test$estimate


# Perform binomial tests for each row and store raw p-values

dt_gt_test_misclsf[, p_mis := round(n_mis/n_mis_sum * 100, 2)]

# expected misclassification propability: because model can predict 8 labels, so
# 7 possibilities of mislabeling (-1 when it is correct)
p_expected <- 1/7

# N misclassifications expected given the misclassification data and expected
# probability
dt_gt_test_misclsf[, n_mis_expect := round(p_expected * n_mis_sum, 0)]

dt_gt_test_misclsf[, p_binomial := binom.test(x = n_mis, 
                                              n = n_mis_sum, 
                                              p = p_expected, 
                                              alternative = "greater")$p.value %>% round(digits = 4), 
                   by = .I]

# Calculate adjusted p-values
dt_gt_test_misclsf[, p_bi_bonferroni := p.adjust(p_binomial, 
                                                 method = "bonferroni") %>% round(digits = 4)]

# Add significance flags
dt_gt_test_misclsf[, `:=` (
  is_signif_p_bi = ifelse(p_binomial <= 0.05, 1, 0),
  is_signif_p_bi_bonferroni = ifelse(p_bi_bonferroni <= 0.05, 1, 0)
)]

# Flag if is below or under expected
dt_gt_test_misclsf[, direction := ifelse(n_mis <= n_mis_expect, "below", "above")]
dt_gt_test_misclsf[(is_signif_p_bi_bonferroni == 1) & (n_mis > n_mis_expect), direction2 := "above"]

dt_gt_test_misclsf


# Values reported in Supp. Table S3. 

dt_final <- dt_gt_test_misclsf[, .(p1_labels, pred_label, n_mis, n_mis_sum, 
                                   p_mis, n_mis_expect, p_binomial)]

# Capitalize first letter of each word in both columns
dt_final[, p1_labels := str_to_title(p1_labels)]
dt_final[, pred_label := str_to_title(pred_label)]
dt_final

# Save as CSV file
fwrite(dt_final, file = "./results/tables/Supp_Table_binom_test_dip_hym_instances.csv")
