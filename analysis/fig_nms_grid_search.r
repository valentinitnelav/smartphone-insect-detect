# Script to produce Figure 3. and Supp. Table S1

library(data.table)
library(magrittr)
library(arrow)
library(ggplot2)
library(RColorBrewer)
library(viridis)
library(patchwork)


# Read data ---------------------------------------------------------------

# Read the results of the detection run at very low NMS conf = 0.001 and an
# array of NMS-IoUs (0.1 to 0.9 with a step of 0.1).
dt <- read_feather('./data/processed/df_coco_eval_conf_0.001_ious_0.1_0.9.feather') %>% setDT()
dt_arrays <- read_feather('./data/processed/df_coco_eval_conf_0.001_ious_0.1_0.9_arrays.feather') %>% setDT()

dt[, nms_iou := as.numeric(nms_iou)]
dt_arrays[, nms_iou := as.numeric(nms_iou)]

# Performance metrics of the best detector for instance localisation
dt_maxF1 <- dt[maxF1 == max(maxF1)]
dt_maxF1[, .(model, nms_iou, maxF1conf, maxF1, AP_auc, AP_mean)]
#      model nms_iou maxF1conf     maxF1    AP_auc   AP_mean
# 1: yolov5s     0.3  0.201918 0.7018849 0.6497147 0.6482324

# For printing with 4 decimals - for reporting in the manuscript
dt_maxF1[, round(maxF1, 4)]     # 0.7019
dt_maxF1[, round(AP_auc, 4)]    # 0.6497
dt_maxF1[, round(maxF1conf, 4)] # 0.2019

# Model that maximised F1, also maximised AUC
dt[AP_auc == max(AP_auc), .(model, nms_iou, maxF1conf, maxF1, AP_auc, AP_mean)]
#      model nms_iou maxF1conf     maxF1    AP_auc   AP_mean
# 1: yolov5s     0.3  0.201918 0.7018849 0.6497147 0.6482324


# Supp. Table S1 ----------------------------------------------------------

result <- dt[, .SD[which.max(maxF1)], by = model]

result[, `:=` (
  AP_mean = round(AP_mean, 4),
  AP_auc = round(AP_auc, 4),
  maxF1 = round(maxF1, 4),
  maxF1conf = round(maxF1conf, 4)
)]

result[, .(model, nms_iou, maxF1conf, maxF1, AP_auc)]
#          model nms_iou maxF1conf  maxF1 AP_auc
# 1:     yolov5s     0.3    0.2019 0.7019 0.6497
# 2: yolov7-tiny     0.3    0.2236 0.6617 0.6294
# 3:     yolov5n     0.1    0.1648 0.6539 0.6111


# Figure 3 ----------------------------------------------------------------

# Graphs with optimal NMS conf and IoU that maximizes F1 score.
# Precision-Recall and F1 curves for the results when setting NMS conf to 0.001


# Define a custom theme
custom_theme <- function() {
  theme_bw() +
    theme(
      text = element_text(size = 10),
      axis.title = element_text(size = 9),
      axis.text = element_text(size = 8),
      axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      plot.tag = element_text(size = 10)
    )
}

# Set the model as factor with the given order so that it appears like that in the legend
yolo_lev <- c("yolov5s", "yolov7-tiny", "yolov5n")
yolo_lab <- c("YOLOv5-small", "YOLOv7-tiny", "YOLOv5-nano")
dt[, model_factor := factor(model, levels = yolo_lev, labels = yolo_lab)]


# ~ Panel gg_f1_iou -------------------------------------------------------------

# Maximum attainable F1 score vs NMS IoU 
gg_f1_iou <- ggplot(dt, 
                    aes(x = nms_iou, 
                        y = maxF1, 
                        color = model_factor, 
                        group = model_factor)) +
  geom_line(linewidth = 0.5) +
  # Dashed line at NMS IoU of 0.3 (best for yolov5s)
  geom_vline(xintercept = dt_maxF1[["nms_iou"]], linetype = "dotted", color = "gray50") +
  scale_x_continuous(breaks = seq(0.1, 0.9, by = 0.1)) +
  labs(x = "NMS IoU", 
       y = "F1 score", 
       color = "Model:") +
  # color will define the legend title
  custom_theme()

gg_f1_iou


# ~ Panel gg_auc_iou ------------------------------------------------------------

# Maximum attainable AUC of PR curve vs NMS IoU 
gg_auc_iou <- ggplot(dt, 
                     aes(x = nms_iou, 
                         y = AP_auc, 
                         color = model_factor, 
                         group = model_factor)) +
  geom_line(linewidth = 0.5) +
  # Dashed line at NMS IoU of 0.3 (best for yolov5s)
  geom_vline(xintercept = dt_maxF1[["nms_iou"]], linetype = "dotted", color = "gray50") + 
  scale_x_continuous(breaks = seq(0.1, 0.9, by = 0.1)) +
  labs(x = "NMS IoU", 
       y = "AUC", 
       color = "Model:") +
  # color will define the legend title
  custom_theme()

gg_auc_iou


# ~ Panel gg_f1_conf ----------------------------------------------------


# Maximum attainable F1 score vs NMS confidence score. Results for best IoUs per
# model.

filtered_dt <- dt_arrays[
  (model == "yolov5s" & nms_iou == 0.3) |
    (model == "yolov7-tiny" & nms_iou == 0.3) |
    (model == "yolov5n" & nms_iou == 0.1)
]

filtered_dt[, model_factor := factor(model, levels = yolo_lev, labels = yolo_lab)]

gg_f1_conf <- ggplot(filtered_dt, 
                     aes(x = conf_array, 
                         y = F1_array)) +
  geom_line(aes(color = model_factor), linewidth = 0.5) +
  # Dashed line at best conf score for yolov5s
  geom_vline(xintercept = dt_maxF1[["maxF1conf"]], linetype = "dotted", color = "gray50") +
  # Ensure x-axis includes 0 to 1 by settining limits
  scale_x_continuous(breaks = seq(0, 1, by = 0.1), limits = c(0, 1)) +
  labs(x = "Confidence score",
       y = "F1 score", 
       color = "Model:") +
  custom_theme()

gg_f1_conf


# ~ Panel gg_pr_conf ----------------------------------------------------

# Max PR curves = Maximum attainable AUC vs NMS confidence score. Results for
# best IoUs per model.

gg_pr_conf <- ggplot(filtered_dt, 
                     aes(x = R_array, 
                         y = P_array)) +
  geom_line(aes(color = model_factor), linewidth = 0.5) +
  scale_x_continuous(breaks = seq(0, 1, by = 0.1), limits = c(0, 1)) +
  labs(x = "Recall",
       y = "Precision", 
       color = "Model:") +
  custom_theme()

gg_pr_conf


# ~ Combine panels --------------------------------------------------------

# Share scales for consistency. Can be added to all plots with the & operator
# via patchwork
shared_scale <- scale_color_brewer(palette = "Dark2")

# Combine the plots
final_fig <- (gg_f1_iou | gg_f1_conf) /
  (gg_auc_iou | gg_pr_conf) +
  plot_annotation(tag_levels = 'a', tag_suffix = ')') +
  plot_layout(guides = "collect") & 
  shared_scale &
  theme(legend.position = "bottom", 
        legend.direction = "horizontal")

final_fig

ggsave("./results/figures/fig_nms_grid_search.jpg", 
       plot = final_fig, 
       width = 14, height = 11, dpi = 300, units = "cm")
