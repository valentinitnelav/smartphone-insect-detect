# Helper script for shaping the data needed to make Figure 5 with
# fig_sequence_detection.py

library(data.table)
library(magrittr)
library(arrow)

# Read tables so that all the TP, FP and FN for corresponding to an eval-IoU of
# 0.1 are bought in one place.

# groudt truth table with TPs & FNs at eval-IoU 0.1
dt_eval <- read_feather('./data/processed/df_eval_seq_yolov5s_iou_0.1.feather') %>% setDT()

# FPs table (because df_eval* doesn't have all the FPs)
dt_pred <- read_feather('./data/processed/df_pred_seq_yolov5s_iou_0.1.feather') %>% setDT()
dt_pred[dt_eval, on = "new_filename", filename_full_frame := i.filename_full_frame, mult = "first"]


# Make the table with ground truths (gt) 
dt_gt <- dt_eval[, .(filename_full_frame, new_filename, x, y, width, height, p1_labels)]
dt_gt[, box_type := 'GT']
setnames(dt_gt, old = 'p1_labels', new = 'box_label')

# Make the table with TPs
dt_tp <- dt_eval[box_type == 'TP', 
                 .(filename_full_frame, new_filename, 
                   pred_x, pred_y, pred_width, pred_height, 
                   pred_label, pred_conf, pred_iou, box_type)]
setnames(dt_tp, 
         old = c('pred_x', 'pred_y', 'pred_width', 'pred_height', 'pred_label', 'pred_conf', 'pred_iou'),
         new = c('x', 'y','width', 'height', 'box_label', 'conf', 'iou'))

# Make the table with FPs
dt_fp <- dt_pred[box_type == 'FP', 
                 .(filename_full_frame, new_filename, x, y, width, height, 
                   yolo_label_name, yolo_conf, box_type)]
setnames(dt_fp,
         old = c('yolo_label_name', 'yolo_conf'),
         new = c('box_label', 'conf'))


tbl <- rbindlist(list(dt_gt, dt_tp, dt_fp), use.names = TRUE, fill = TRUE)
tbl[, .N, keyby = box_type]
# For eval-IoU 0.1:
#    box_type     N
# 1:       FP  1799
# 2:       GT 24656
# 3:       TP 15120

tbl[dt_eval, on = 'new_filename', 
    `:=`(seq_id = i.seq_id, 
         label_seq = i.common_label)]

# used for the figure
tbl[seq_id == 52] 
tbl[seq_id == 52, .N] # 9 types of boxes
tbl[seq_id == 52, uniqueN(new_filename)] # 6 images

write_feather(tbl, './data/processed/df_eval_seq_yolov5s_iou_0.1_interim.feather')
