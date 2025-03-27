# R script to compute the values that were reported in the manuscript when
# describing the dataset.

# It also computes the values from Table 1 in the manuscript.


library(data.table)
library(magrittr)
library(arrow)


# Read data ---------------------------------------------------------------

# Raw annotations dataset for full-frames (e.g. this includes extra orders like
# thysanoptera and "no_id")
dt_raw <- read_feather('./data/annotations/annotations_full_frames.feather') %>% setDT()
dt_raw[, .N, keyby = order]


# Annotations dataset with images of insects only within the ROI & selected for
# analysis. These include also the 182 instances of insects with ROI, but not
# considered as target objects as they did not visit the target flower.
dt <- read_feather('./data/processed/df_roi.feather') %>% setDT()
dt[, .N, keyby = p1_labels]


# Descriptives ------------------------------------------------------------

# How many "plant folders" in the raw dataset?
dt_raw[, .N, by = .(date, plant_folder)] %>% nrow()  
# 213 folders that we visually parsed
dt[, .N, by = .(date, plant_folder)] %>% nrow() 
# 201 end up in the cleaned dataset (cropped images)

# Nr plant species
dt_raw[, uniqueN(.SD, by = c("plant_genus", "plant_epithet"))] 
# 33 distinct plant species
dt[, uniqueN(.SD, by = c("plant_genus", "plant_epithet"))] 
# 32 remain in the cropped dataset for analysis

# Initial Nr images in dt_raw and how many are left in the OOD dataset.
# Number of annotated images and boxes in the raw dataset:
dt_raw[, .(n_img = uniqueN(filename_full_frame))] # 33502 images
nrow(dt_raw) # 35192 bounding boxes (instances)

# Number of annotated images and boxes in the OOD dataset:
dt[, .(n_img = uniqueN(filename_full_frame))] # 23899 images
nrow(dt) # 24838 bounding boxes (instances)

# Average width and height of the cropped images
dt[, .(width_crop_avg = mean(width_crop) %>% round(digits = 0),
       height_crop_avg = mean(height_crop) %>% round(digits = 0))]
#    width_crop_avg height_crop_avg
# 1:            851             796

# Average width and height of the raw images that were cropped
dt[, .(width_avg = mean(img_width_pil) %>% round(digits = 0),
       height_avg = mean(img_height_pil) %>% round(digits = 0))]
#    width_avg height_avg
# 1:      1571       1252

# Global average box area:
dt[keep_seq == TRUE, mean(box_area_rel)] %>% round(3)   # 0.075
dt[keep_seq == TRUE, median(box_area_rel)] %>% round(3) # 0.028


# Number of distinct insects
dt[!is.na(seq_id), uniqueN(seq_id)]
# 1281
dt[is.na(seq_id), .N]
# 182 cases of background insects within the ROI


# Visit durations ---------------------------------------------------------

# Range of bounding boxes (in this case frames too) There are insects that appea
# rin only one frame and others up to 1710 (a coleoptera that didn't move much)
dt[keep_seq == TRUE, .N, by = seq_id][, range(N)] # 1 1710
# Median for the number of frames
dt[keep_seq == TRUE, .N, by = seq_id][, median(N)] # 7
# Per insect groups
dt[keep_seq == TRUE, .N, by = .(p1_labels, seq_id)][, range(N) %>% paste(collapse = "-"), by = p1_labels]
#        p1_labels     V1
# 1:    coleoptera 4-1710
# 2:   hymenoptera  1-308
# 3:       araneae  2-704
# 4:       diptera  1-755
# 5:     hemiptera  2-221
# 6: hymenoptera_f 1-1254

# Median of N boxes per insect group. Hymenoptera drives the dataset anyways.
dt[keep_seq == TRUE, .N, by = .(p1_labels, seq_id)][,  median(N), by = p1_labels]
#        p1_labels   V1
# 1:    coleoptera 19.5
# 2:   hymenoptera  7.0
# 3:       araneae 33.0
# 4:       diptera 10.0
# 5:     hemiptera 29.0
# 6: hymenoptera_f  4.0

# Typical visit duration:
# = overall median nr. of frames * mean time-step (value from previous study)
7 * 1.6 # 11.2 sec.


# Nr. of insects per image ------------------------------------------------

# Most images contain just a single insect:
# To get the correct number of images with 1, 2, 3 & 4 boxes, use n_boxes and
# not id_box. If you use id_box, you inflate the numbers because an image that
# has 4 boxes, also has an id_box = 3 and it gets counted in as well, when it
# should not be.
dt[, .(n_rows = .N, n_img = uniqueN(new_filename)), keyby = n_boxes][
  , img_prc := round(n_img/sum(n_img) *100, 1)][order(-img_prc)]
#    n_boxes n_rows n_img img_prc
# 1:       1  22668 22668    94.8
# 2:       2   2011  1162     4.9
# 3:       3    157    68     0.3
# 4:       4      2     1     0.0
# double checking:
# 22668 + 2011 + 157 + 2 = 24838, which is exactly nr rows in dt
# 22668 + 1162 + 68 + 1 = 23899, which is exactly number of unique images in dt (see above)

# Same as above, but on dt_raw
dt_raw[, .(n_rows = .N, n_img = uniqueN(new_filename)), keyby = n_boxes][
  , img_prc := round(n_img/sum(n_img) *100, 1)][order(-img_prc)]
#    n_boxes n_rows n_img img_prc
# 1:       1  31901 22669    94.8
# 2:       2   3002  1163     4.9
# 3:       3    285    69     0.3
# 4:       4      4     1     0.0
# Same percent in both cases (at 1 decimal), despite some small differences in
# nr. of images


# Summary table with Nr boxes ---------------------------------------------

# Nr. boxes (instances) per order and their percents.

# Note that using n_img = uniqueN(img_path) is not accurate because the same
# image can be counted two times if it contains boxes of two different orders.
order_summary_dt <- dt[keep_seq == TRUE, .(n_box = .N), keyby = p1_labels][
  , percent := round(n_box/sum(n_box) *100, 2)][order(-percent)]

order_summary_dt[, cumul_sum := cumsum(n_box/sum(n_box)*100) %>% round(2)]
# Adding the total row
total_row <- data.table(p1_labels = "Total",
                        n_box = sum(order_summary_dt$n_box),
                        percent = round(sum(order_summary_dt$percent), 1))
# Combining the summary data.table with the total row
order_summary_dt <- rbindlist(list(order_summary_dt, total_row), fill=TRUE)
order_summary_dt
#        p1_labels n_box percent cumul_sum
# 1:   hymenoptera 13254   53.76     53.76
# 2:       diptera  5018   20.35     74.11
# 3:    coleoptera  2778   11.27     85.37
# 4: hymenoptera_f  1967    7.98     93.35
# 5:       araneae  1036    4.20     97.55
# 6:     hemiptera   603    2.45    100.00
# 7:         Total 24656  100.00        NA


# Table 1 -----------------------------------------------------------------

# Table 1 in the manuscript.

# Summary table with N boxes, N img, N unique insects and relative box area
# (mean) by order for the target pollinators.
order_summary_dt_2 <- dt[keep_seq == TRUE, 
                         .(
                           n_box = .N, 
                           rel_box_area = mean(box_area_rel) %>% round(digits = 3),
                           n_img = uniqueN(new_filename),
                           n_insect = uniqueN(seq_id)
                         ), 
                         by = p1_labels][order(-n_img)]
order_summary_dt_2[, n_insect_pr := round(n_insect/sum(n_insect) *100, 2)]
order_summary_dt_2[, cumul_sum := cumsum(n_insect_pr) %>% round(2)]

# Add the total row
total_row_2 <- data.table(p1_labels = "",
                          n_box = sum(order_summary_dt_2$n_box),
                          n_insect = sum(order_summary_dt_2$n_insect),
                          n_insect_pr = sum(order_summary_dt_2$n_insect_pr))

# Combining the summary data.table with the total row
order_summary_dt_sum <- rbindlist(list(order_summary_dt_2, total_row_2), fill=TRUE)
order_summary_dt_sum[, .(p1_labels, n_box, rel_box_area, n_img, n_insect, n_insect_pr, cumul_sum)]
#        p1_labels n_box rel_box_area n_img n_insect n_insect_pr cumul_sum
#           <char> <int>        <num> <int>    <int>       <num>     <num>
# 1:   hymenoptera 13254        0.107 13084     1013       79.08     79.08
# 2:       diptera  5018        0.071  4998      145       11.32     90.40
# 3:    coleoptera  2778        0.010  2770       20        1.56     91.96
# 4: hymenoptera_f  1967        0.013  1962       82        6.40     98.36
# 5:       araneae  1036        0.014   994       10        0.78     99.14
# 6:     hemiptera   603        0.011   603       11        0.86    100.00
# 7:               24656           NA    NA     1281      100.00        NA

# Save as CSV file to place into manuscript
fwrite(order_summary_dt_sum, file = "./results/tables/Table_1.csv")

order_summary_dt_2$n_box %>% sum() # Sum n_box = 24656 ( for target pollinators only)
order_summary_dt_2$n_img %>% sum() # Sum n_img = 24411 images for the target pollinators. 
# But if a image contains multiple orders, it will be counted multiple times.
# That is why the number is inflated. The sum is bigger than if you compute it globally:
dt[, .(n_img = uniqueN(new_filename))] # 23899
dt[keep_seq == TRUE, .(n_img = uniqueN(new_filename))] # 23899 (same)
