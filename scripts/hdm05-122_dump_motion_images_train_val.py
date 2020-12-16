import numpy as np
import cv2
import os
import time
import math
from pathlib import Path
import plotly.graph_objects as go
from mofex.preprocessing.sequence import Sequence
from sklearn.model_selection import train_test_split
from mofex.load_sequences import load_seqs_asf_amc_hdm05
from mofex.preprocessing.helpers import xyz_minmax_coords

# Root folder for Sequence files
src_root = './data/sequences/hdm05-122/amc/'
filename_asf = '*.asf'
filename_amc = '*.amc'

dump_root = 'data/motion_images'
dataset_name = 'hdm05-122_90-10'

# Indices constants for body parts that define normalized orientation of the skeleton
# left -> hip_left
LEFT_IDX = 1
# right -> hip_right
RIGHT_IDX = 6
# up -> lowerback
UP_IDX = 11

# --- Loading Sequences and preprocessing
seqs = load_seqs_asf_amc_hdm05(src_root, filename_asf, filename_amc)
labeled_sequences_dict = {}

for seq in seqs:
    print(f'Processing: {seq.name}')
    # Normalization
    # * Body Part Indices -> 1 = left hip; 6 = right hip
    seq.norm_center_positions()
    seq.norm_relative_to_positions((seq.positions[:, LEFT_IDX, :] + seq.positions[:, RIGHT_IDX, :]) * 0.5)
    seq.norm_orientation(seq.positions[0, LEFT_IDX], seq.positions[0, RIGHT_IDX], seq.positions[0, UP_IDX])

    # Fill dictionary with classes present in dataset to split sequences in each class seperately
    if seq.desc in labeled_sequences_dict.keys():
        labeled_sequences_dict[seq.desc].append(seq)
    else:
        labeled_sequences_dict[seq.desc] = [seq]

# Filter Outliers and get xyz_minmax values
iqr_factor_x = 2.5
iqr_factor_y = 2.5
iqr_factor_z = 3.5
minmax_xyz = xyz_minmax_coords(seqs, [iqr_factor_x, iqr_factor_y, iqr_factor_z], plot_histogram=True)
xmin, xmax = minmax_xyz[0]
ymin, ymax = minmax_xyz[1]
zmin, zmax = minmax_xyz[2]

print(f"Loaded Sequence Files")
train_seqs = []
val_seqs = []
# Split class sequences into train/val sets
for label in labeled_sequences_dict.keys():
    label_split = train_test_split(labeled_sequences_dict[label], test_size=0.1, random_state=42)
    train_seqs.extend(label_split[0])
    val_seqs.extend(label_split[1])

minmax_filename = f'{dump_root}/{dataset_name}/minmax_values.txt'
# Create basic text file to store(remember) the used min/max values
if not os.path.isdir(f'{dump_root}/{dataset_name}'):
    os.makedirs(f'{dump_root}/{dataset_name}')
minmax_file = open(minmax_filename, 'w')
minmax_file.write(f'x: [{xmin}, {xmax}]\ny: [{ymin}, {ymax}]\nz: [{zmin}, {zmax}]')
minmax_file.write(f'\n')
minmax_file.write(f'x_iqr_factor: {iqr_factor_x}\ny_iqr_factor: {iqr_factor_y}\nz_iqr_factor: {iqr_factor_z}')
minmax_file.close()

# --- Create Motion Images
# Train Set
for seq in train_seqs:
    # set and create directories
    class_dir = f'{dump_root}/{dataset_name}/train/{seq.desc}'
    out = f'{class_dir}/{seq.name}.png'
    if not os.path.isdir(os.path.abspath(class_dir)):
        os.makedirs(os.path.abspath(class_dir))
    # Create and save Motion Image with defined output size and X,Y,Z min/max values to map respective color channels to positions.
    # (xmin, xmax) -> RED(0, 255), (ymin, ymax) -> GREEN(0, 255), (zmin, zmax) -> BLUE(0, 255),
    img = seq.to_motionimg(output_size=(256, 256), minmax_pos_x=(xmin, xmax), minmax_pos_y=(ymin, ymax), minmax_pos_z=(zmin, zmax))
    cv2.imwrite(out, img)
# Validation Set
for seq in val_seqs:
    # set and create directories
    class_dir = f'{dump_root}/{dataset_name}/val/{seq.desc}'
    out = f'{class_dir}/{seq.name}.png'
    if not os.path.isdir(os.path.abspath(class_dir)):
        os.makedirs(os.path.abspath(class_dir))
    # Create and save Motion Image with defined output size and X,Y,Z min/max values to map respective color channels to positions.
    # (xmin, xmax) -> RED(0, 255), (ymin, ymax) -> GREEN(0, 255), (zmin, zmax) -> BLUE(0, 255),
    img = seq.to_motionimg(output_size=(256, 256), minmax_pos_x=(xmin, xmax), minmax_pos_y=(ymin, ymax), minmax_pos_z=(zmin, zmax))
    cv2.imwrite(out, img)