# Script to create the multi-panel figure with distributions of
# misclassifications vs correctly classified cases for Diptera and Hymenoptera
# taxa groups.

library(data.table)
library(magrittr)
library(ggplot2)
library(ggh4x) # for facet_grid2
# see https://stackoverflow.com/a/73649205/5193830
# https://teunbrand.github.io/ggh4x/


# Read data.
# Data tables produced with analysis_prep.r
dt <- readRDS('./data/processed/dt_analysis.rds')
dt_seq <- readRDS('./data/processed/dt_seq_analysis_seq_individs.rds')


gr_cols <- c("superfamily", "family", "genus", "morphospecies", "species")

dt_mis <- dt_seq[(p1_labels == "hymenoptera") & (pred_arthropod == "diptera"), 
                 .(
                   taxa_combi = paste(unique(taxa_combi), collapse = "; "),
                   n_mis_target = .N # nr individuals misclassified as Diptera
                 ),
                 keyby = gr_cols]

dt_mis[order(-n_mis_target)]

# Re-label for figure
dt[, category := fcase(
  taxa_combi == "hymenoptera_apocrita_aculeata_apoidea_apidae_NA_apis_NA_mellifera", "Apis mellifera",
  taxa_combi == "hymenoptera_apocrita_aculeata_apoidea_apidae_NA_bombus_red_tailed_NA", "Bombus - red tail",
  taxa_combi %in% c("hymenoptera_apocrita_aculeata_apoidea_halictidae_NA_NA_NA_NA",
                    "hymenoptera_apocrita_proctotrupomorpha_cynipoidea_cynipidae_NA_NA_NA_NA",
                    "hymenoptera_apocrita_aculeata_apoidea_halictidae_NA_halictus_NA_NA",
                    "hymenoptera_apocrita_aculeata_apoidea_andrenidae_NA_NA_NA_NA",
                    "hymenoptera_apocrita_aculeata_apoidea_megachilidae_NA_megachile_NA_NA",
                    "hymenoptera_apocrita_proctotrupomorpha_cynipoidea_NA_NA_NA_NA_NA"), "Non-mimicked",
  (p1_labels == "diptera") & (is.na(family) | family != "syrphidae"), "Coarsely identified",
  (p1_labels == "diptera") & family == "syrphidae", "Syrphidae"
)]

# Filter for ground truths of interest
dt_ctg <- dt[! is.na(category)][is_max_conf_row == TRUE][common_label %in% c("hymenoptera", "diptera")]

dt_ctg[, .N, keyby = .(order, common_label)]
dt_ctg[, .N, keyby = .(order, category, common_label)]
dt_ctg[, .N, keyby = .(order, category, seq_label_matched)]

dt_ctg[, seq_label_matched_factor := factor(seq_label_matched, 
                                            levels = c(FALSE, TRUE), 
                                            labels = c("no", "yes"))]
dt_ctg[, order_fact := factor(order, 
                              levels = c("hymenoptera", "diptera"), 
                              labels = c("Hymenoptera", "Diptera"))]

# Reshape the data into long format
dt_ctg_long <- melt(dt_ctg, 
                    id.vars = c("seq_label_matched_factor", "order_fact", "category"), 
                    measure.vars = c("box_area_rel", "score_sobel", "pred_conf"), 
                    variable.name = "metric", 
                    value.name = "value")

# Rename factor levels in the 'metric' column for plotting
levels(dt_ctg_long$metric)
levels(dt_ctg_long$metric) <- c("Rel. box area", "Sharpness", "YOLO confidence")

dt_ctg_long[, category_fact := factor(category, 
                                      levels = c("Apis mellifera",
                                                 "Bombus - red tail",
                                                 "Non-mimicked",
                                                 "Syrphidae", 
                                                 "Coarsely identified"),
                                      labels = c("Apis mellifera",
                                                 "Bombus - red tail",
                                                 "Non-mimicked",
                                                 "Syrphidae", 
                                                 "Coarsely identified"))]

set.seed(42)
font_size <- 10

gg <- ggplot(data = dt_ctg_long, 
             aes(x = seq_label_matched_factor, 
                 y = value, 
                 color = factor(seq_label_matched_factor),
                 fill = factor(seq_label_matched_factor))) +
  geom_violin(alpha = 0.2, color = NA) +  # Shows distribution similar to a histogram
  geom_jitter(width = 0.3, height = 0, alpha = 0.6, shape=20, size = 0.3) +
  geom_boxplot(width = 0.6, lwd = 0.3, outlier.shape = NA, color = "black", fill = NA, alpha = 0) + 
  stat_summary(fun = mean, geom = "point", shape = 23, size = 1.5, colour="white") + # Mean point
  scale_color_brewer(palette = "Set1") + 
  scale_fill_brewer(palette = "Set1") + 
  labs(x = "Correctly classified") +
  theme_bw() +
  theme(
    legend.position = "none",
    text = element_text(size = font_size),
    axis.text.x = element_text(size = 8),
    axis.text.y = element_text(size = 8),
    axis.title.x = element_text(size = 9),
    axis.title.y = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  ) +
  # Facet by metric (rows) and category (columns)
  facet_nested(metric ~ order_fact + category_fact,
               labeller = "label_value", 
               scales = "free",
               independent = "y",
               remove_labels = "x")  

gg

ggsave("./results/figures/fig_boxplot_misclassifications_mimicry.jpg", 
       plot = gg, width = 18, height = 11, dpi = 300, units = "cm")
