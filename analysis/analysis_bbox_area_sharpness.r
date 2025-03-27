# To assess differences in relative bounding box area and normalized image sharpness for
# localisation and classification tasks, we implemented a nonparametric
# permutation test. This test examines whether the means and medians of two
# distributions differ, assuming under the null hypothesis that the
# distributions are identical, with expected differences in these metrics being
# zero. We compared two groups: (1) ground truth arthropod boxes that were
# either localized or not, and (2) among localized instances, those correctly
# classified versus misclassified. We selected this test due to the long-tailed
# distributions, which deviate from normality.

library(data.table)
library(magrittr)
library(ggplot2)
library(coin)

# Load the permutation_test() function. Performs a permutation test to compare
# the difference in means or medians between two samples.
source('./analysis/utils.r')

# Read data table produced with analysis_prep.r
dt <- readRDS('./data/processed/dt_analysis.rds') # instances (one row one box)

# Expect 0 cases without normalized sharpness (Sobel values) or relative
# bounding box area.
dt[is.na(score_sobel), .N]  # 0
dt[is.na(box_area_rel), .N] # 0


# Means -------------------------------------------------------------------

# Means of relative bounding box area and normalized sharpness (Sobel values)
# within boxes.
dt[, .( n_box = .N,
        n_ind = uniqueN(seq_id),
        mean_bbx_rel_area = mean(box_area_rel, na.rm = TRUE) %>% round(4),
        mean_sobel = mean(score_sobel, na.rm = TRUE) %>% round(4) ),
   by = true_group][order(-n_ind)]
#     true_group n_box n_ind mean_bbx_rel_area mean_sobel
# 1: hymenoptera 13254  1013            0.1073     0.1020
# 2:     diptera  5018   145            0.0708     0.1178
# 3:       other  6384   123            0.0115     0.0321

# And overall means
dt[, .( mean_bbx_rel_area = mean(box_area_rel, na.rm = TRUE) %>% round(4),
        mean_sobel = mean(score_sobel, na.rm = TRUE) %>% round(4) )]
#      mean_bbx_rel_area mean_sobel
#   1:            0.0751     0.0871


# Exploratory histograms --------------------------------------------------

# Histograms of distributions for relative bounding box area and normalized
# sharpness (Sobel values). These are long tailed distributions.

# Histogram for relative bounding box area
gg_bbox_area <- ggplot(data = dt, 
                       aes(x = box_area_rel, 
                           fill = box_detected_factor)) +
  geom_histogram(alpha = 0.5, position = "identity") +
  theme_minimal() +
  xlab("Relative box area") +
  ylab("Counts boxes") +
  scale_fill_manual(values = c("yes" = "blue", "no" = "red"))
gg_bbox_area

# And per groups
gg_bbox_area + facet_wrap(~ p1_labels, scales = 'free')
gg_bbox_area + facet_wrap(~ true_group, scales = 'free')


# Histogram for normalized sharpness (Sobel values)
gg_sharp <- ggplot(data = dt, 
                   aes(x = score_sobel, 
                       fill = box_detected_factor)) +
  geom_histogram(alpha = 0.5, position = "identity") +
  theme_minimal() +
  xlab("Normalized sharpness") +
  ylab("Counts boxes") +
  scale_fill_manual(values = c("yes" = "blue", "no" = "red"))
gg_sharp

# And per groups
gg_sharp + facet_wrap(~ p1_labels, scales = 'free')
gg_sharp + facet_wrap(~ true_group, scales = 'free')


# Spearman’s rank correlation ---------------------------------------------

# Spearman’s rank correlation between relative bounding box area and normalized
# sharpness (Sobel values).
dt[, cor(x = box_area_rel, y = score_sobel, method = "spearman")] %>% round(digits = 2)

dt[, cor.test(x = box_area_rel, y = score_sobel, method = "spearman", exact = FALSE)]
cor.test(dt$box_area_rel, dt$score_sobel, method = "spearman", exact = FALSE)
# Used exact = FALSE to cope with ties.
# Spearman's rank correlation rho
# 
# data:  box_area_rel and score_sobel
# S = 5.1268e+11, p-value < 2.2e-16
# alternative hypothesis: true rho is not equal to 0
# sample estimates:
#       rho 
# 0.7947757 


# Permutation tests -------------------------------------------------------

# Notes:

# The permutation tests below will return two-tailed p-values of 0, with the
# observed differences falling well outside the tails of the null distribution,
# as seen when plotting the null distributions with
# permutation_test_histogram().

# The two-tailed p-value represents the probability of observing a difference as
# extreme as (or more extreme than) the observed difference, assuming the null
# hypothesis is true (i.e., no actual difference between the tow samples). A
# p-value of 0 (or near zero) suggests that such an extreme observed difference
# is extremely unlikely under the null hypothesis. This means there is strong
# evidence against the null hypothesis, indicating a statistically significant
# difference between the two samples. The p-value was computed using 1,000
# permutations.

# So, empirical data significantly deviates from what would be expected if the
# samples were coming from the same distribution. The difference in means or
# medians between the two samples is likely not due to random chance.

# Effect sizes (observed differences) are also reported together with the mean
# of the null distribution and its permuted-quantile based confidence interval
# (CI). So, the CI from the null distribution indicates the range within which
# the majority (typically 95%) of the permuted differences lie under the
# assumption that there is no true difference between the samples (the null
# hypothesis).

# If the observed difference falls within this CI interval, it suggests that the
# observed difference could reasonably occur due to random chance, according to
# the null hypothesis. Conversely, if the observed difference falls outside this
# range, it indicates that such a difference is unlikely to occur by chance
# alone, providing evidence against the null hypothesis. In the analysis below,
# the observed differences were constantly outside this confidence interval,
# supporting the conclusion of a statistically significant difference between
# the two samples.


# A) Localization case ----------------------------------------------------

# ~ for relative bounding box area ----------------------------------------

# Prepare the two samples: instance not localized vs localized.
sample_no  <- dt[box_detected_factor == "no", box_area_rel]
sample_yes <- dt[box_detected_factor == "yes", box_area_rel]

# ~~ compare means --------------------------------------------------------

# Note: observed difference is sample2 - sample1.
perm_test <- permutation_test(sample1 = sample_no, 
                              sample2 = sample_yes, 
                              n_sim = 1000, 
                              stat = "mean")
perm_test$p_value # 0
perm_test$observed_difference %>% round(4) # 0.0781
perm_test$null_distribution_mean # -2.024623e-06
perm_test$confidence_interval

permutation_test_histogram(perm_test)

# ~~ compare medians ------------------------------------------------------

# Note: observed difference is sample2 - sample1.
perm_test <- permutation_test(sample1 = sample_no, 
                              sample2 = sample_yes, 
                              n_sim = 1000, 
                              stat = "median")
perm_test$p_value # 0
perm_test$observed_difference %>% round(4) # 0.0672
perm_test$null_distribution_mean # -0.000131505

permutation_test_histogram(perm_test)


# ~ for normalized sharpness ----------------------------------------------

# Prepare the two samples: instance not localized vs localized.
sample_no  <- dt[box_detected_factor == "no", score_sobel]
sample_yes <- dt[box_detected_factor == "yes", score_sobel]

# ~~ compare means --------------------------------------------------------

# Note: observed difference is sample2 - sample1.
perm_test <- permutation_test(sample1 = sample_no, 
                              sample2 = sample_yes, 
                              n_sim = 1000, 
                              stat = "mean")
perm_test$p_value # 0
perm_test$observed_difference %>% round(4) # 0.0916
perm_test$null_distribution_mean # -2.477928e-05

permutation_test_histogram(perm_test)

# ~~ compare medians ------------------------------------------------------

# Note: observed difference is sample2 - sample1.
perm_test <- permutation_test(sample1 = sample_no, 
                              sample2 = sample_yes, 
                              n_sim = 1000, 
                              stat = "median")
perm_test$p_value # 0
perm_test$observed_difference %>% round(4) # 0.0729
perm_test$null_distribution_mean # -2.346803e-05

permutation_test_histogram(perm_test)


# B) Classification case --------------------------------------------------

dt_cls <- dt[!is.na(pred_label), .(label_matched, box_area_rel, score_sobel)]

# ~ for relative bounding box area ----------------------------------------

# Prepare the two samples: instance not localized vs localized.
sample_no  <- dt[label_matched == 0, box_area_rel]
sample_yes <- dt[label_matched == 1, box_area_rel]

# ~~ compare means --------------------------------------------------------

# Note: observed difference is sample2 - sample1.
perm_test <- permutation_test(sample1 = sample_no, 
                              sample2 = sample_yes, 
                              n_sim = 1000, 
                              stat = "mean")
perm_test$p_value # 0
perm_test$observed_difference %>% round(4) # 0.0304
perm_test$null_distribution_mean # -0.000102118

permutation_test_histogram(perm_test)

# ~~ compare medians ------------------------------------------------------

# Note: observed difference is sample2 - sample1.
perm_test <- permutation_test(sample1 = sample_no, 
                              sample2 = sample_yes, 
                              n_sim = 1000, 
                              stat = "median")
perm_test$p_value # 0
perm_test$observed_difference %>% round(4) # 0.0219
perm_test$null_distribution_mean # -0.0001523814

permutation_test_histogram(perm_test)


# ~ for normalized sharpness ----------------------------------------------

# Prepare the two samples: instance not localized vs localized.
sample_no  <- dt[label_matched == 0, score_sobel]
sample_yes <- dt[label_matched == 1, score_sobel]

# ~~ compare means --------------------------------------------------------

# Note: observed difference is sample2 - sample1.
perm_test <- permutation_test(sample1 = sample_no, 
                              sample2 = sample_yes, 
                              n_sim = 1000, 
                              stat = "mean")
perm_test$p_value # 0
perm_test$observed_difference %>% round(4) # 0.023
perm_test$null_distribution_mean # 6.554763e-05

permutation_test_histogram(perm_test)

# ~~ compare medians ------------------------------------------------------

# Note: observed difference is sample2 - sample1.
perm_test <- permutation_test(sample1 = sample_no, 
                              sample2 = sample_yes, 
                              n_sim = 1000, 
                              stat = "median")
perm_test$p_value # 0
perm_test$observed_difference %>% round(4) # 0.0133
perm_test$null_distribution_mean # -1.385243e-05

permutation_test_histogram(perm_test)
