# Script to produce Figure 4.

library(data.table)
library(ggplot2)
library(patchwork)


# Read and prepare data ---------------------------------------------------

dt <- readRDS('./data/processed/dt_analysis.rds')

# From the localized boxes, check the classification
dt[, box_class_cor := p1_labels == pred_label]

dt_cls <- dt[!is.na(box_class_cor), 
             .(box_class_cor, p1_labels, pred_label, box_area_rel, score_sobel)]

dt_cls[, box_class_cor_factor := factor(box_class_cor, levels = c(FALSE, TRUE), labels = c("no", "yes"))] 

dt_cls[, .N, keyby = box_class_cor_factor]
#    box_class_cor_factor     N
# 1:                   no  2783
# 2:                  yes 11871


# Panel a  ----------------------------------------------------------------

# Relative box area by localized yes vs no

set.seed(2023)

gg_boxpot_bbarea <- ggplot(dt, aes(x=box_detected_factor, 
                                   y=box_area_rel, 
                                   color=box_detected_factor,
                                   fill=box_detected_factor)) +
  geom_jitter(width = 0.3, alpha = 0.15, shape=20, size = 0.05) +
  geom_boxplot(width = 0.6, lwd = 0.3, outlier.shape = NA, color = "black", alpha=0) + 
  scale_color_brewer(palette="Set1") + 
  scale_fill_brewer(palette="Set1") + 
  labs(color="Box detection", 
       shape="Box detection",
       fill="Box detection",
       x = "Localised",
       y = "Relative box area") +
  theme_bw() +
  theme(
    legend.position = "none",
    text = element_text(size = 10),        # set global text size to 10
    axis.title = element_text(size = 10),  # set axis title size to 10
    axis.text = element_text(size = 10),   # set axis text size to 10
    panel.grid.major = element_blank(),    # remove major grid lines
    panel.grid.minor = element_blank()     # remove minor grid lines
  )

gg_boxpot_bbarea


# Panel b -----------------------------------------------------------------

set.seed(2023)

gg_boxplot_sharp <- ggplot(dt, aes(x=box_detected_factor, 
                                   y=score_sobel, 
                                   color=box_detected_factor,
                                   fill=box_detected_factor)) +
  geom_jitter(width = 0.3, alpha = 0.15, shape=20, size = 0.05) +
  geom_boxplot(width = 0.6, lwd = 0.3, outlier.shape = NA, color = "black", alpha=0) + 
  scale_color_brewer(palette="Set1") + 
  scale_fill_brewer(palette="Set1") + 
  labs(color="Box detection", 
       shape="Box detection",
       fill="Box detection",
       x = "Localised",
       y = "Normalised sharpness") +
  theme_bw() +
  theme(
    legend.position = "none",
    text = element_text(size = 10),        # set global text size to 10
    axis.title = element_text(size = 10),  # set axis title size to 10
    axis.text = element_text(size = 10),   # set axis text size to 10
    panel.grid.major = element_blank(),    # remove major grid lines
    panel.grid.minor = element_blank()     # remove minor grid lines
  )

gg_boxplot_sharp


# Panel c -----------------------------------------------------------------

set.seed(2023)

gg_cls_boxpot_bbarea <- ggplot(data = dt_cls, 
                               aes(x=box_class_cor_factor, 
                                   y=box_area_rel, 
                                   color=box_class_cor_factor,
                                   fill=box_class_cor_factor)) +
  geom_jitter(width = 0.3, alpha = 0.15, shape=20, size = 0.05) +
  geom_boxplot(width = 0.6, lwd = 0.3, outlier.shape = NA, color = "black", alpha=0) + 
  scale_color_brewer(palette="Set1") + 
  scale_fill_brewer(palette="Set1") + 
  labs(x = "Correctly classified",
       y = "Relative box area") +
  theme_bw() +
  theme(
    legend.position = "none",
    text = element_text(size = 10),        # set global text size to 10
    axis.title = element_text(size = 10),  # set axis title size to 10
    axis.text = element_text(size = 10),   # set axis text size to 10
    panel.grid.major = element_blank(),    # remove major grid lines
    panel.grid.minor = element_blank()     # remove minor grid lines
  )

gg_cls_boxpot_bbarea


# Panel d -----------------------------------------------------------------

set.seed(2023)

gg_cls_boxplot_sharp <- ggplot(data = dt_cls, 
                               aes(x=box_class_cor_factor, 
                                   y=score_sobel, 
                                   color=box_class_cor_factor,
                                   fill=box_class_cor_factor)) +
  geom_jitter(width = 0.3, alpha = 0.15, shape=20, size = 0.05) +
  geom_boxplot(width = 0.6, lwd = 0.3, outlier.shape = NA, color = "black", alpha=0) + 
  scale_color_brewer(palette="Set1") + 
  scale_fill_brewer(palette="Set1") + 
  labs(x = "Correctly classified",
       y = "Normalised sharpness") +
  theme_bw() +
  theme(
    legend.position = "none",
    text = element_text(size = 10),        # set global text size to 10
    axis.title = element_text(size = 10),  # set axis title size to 10
    axis.text = element_text(size = 10),   # set axis text size to 10
    panel.grid.major = element_blank(),    # remove major grid lines
    panel.grid.minor = element_blank()     # remove minor grid lines
  )

gg_cls_boxplot_sharp


# Combine panels ----------------------------------------------------------

# Boxplots of relative bounding box area and box image sharpness categorized by
# successful localisation and classification status ("no" vs. "yes"), and
# aggregated across all insect categories.

fig_boxplots <- (gg_boxpot_bbarea | gg_boxplot_sharp) / (gg_cls_boxpot_bbarea | gg_cls_boxplot_sharp)
fig_boxplots <- fig_boxplots + plot_annotation(tag_levels = 'a', tag_suffix = ')')

fig_boxplots

ggsave("./results/figures/fig_boxplots_bbox_area_sharpness.jpg", 
       plot = fig_boxplots, 
       width = 14, height = 10, dpi = 300, units = "cm")
