import numpy as np
import cv2
import os
import time
import math
from pathlib import Path
import plotly.graph_objects as go
from mofex.preprocessing.sequence import Sequence
from sklearn.model_selection import train_test_split
from mofex.load_sequences import load_seqs_asf_amc

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
seqs = load_seqs_asf_amc(src_root, filename_asf, filename_amc)
labeled_sequences_dict = {}

# Init Lists to store X,Y,Z coordinates seperately in order to determine usefull min/max values for color mapping
x = []
y = []
z = []
for seq in seqs:
    print(f'Processing: {seq.name}')
    # Normalization
    # * Body Part Indices -> 1 = left hip; 6 = right hip
    seq.norm_center_positions()
    seq.norm_relative_to_positions((seq.positions[:, LEFT_IDX, :] + seq.positions[:, RIGHT_IDX, :]) * 0.5)
    seq.norm_orientation(seq.positions[0, LEFT_IDX], seq.positions[0, RIGHT_IDX], seq.positions[0, UP_IDX])
    # Add flattened xyz values to list respectively
    x.extend(seq.positions[:, :, 0].flatten().tolist())
    y.extend(seq.positions[:, :, 1].flatten().tolist())
    z.extend(seq.positions[:, :, 2].flatten().tolist())
    # Fill dictionary with classes present in dataset to split sequences in each class seperately
    if seq.desc in labeled_sequences_dict.keys():
        labeled_sequences_dict[seq.desc].append(seq)
    else:
        labeled_sequences_dict[seq.desc] = [seq]

print(f"Loaded Sequence Files")
train_seqs = []
val_seqs = []
# Split class sequences into train/val sets
for label in labeled_sequences_dict.keys():
    label_split = train_test_split(labeled_sequences_dict[label], test_size=0.1, random_state=42)
    train_seqs.extend(label_split[0])
    val_seqs.extend(label_split[1])

x = np.array(x)
y = np.array(y)
z = np.array(z)
x.sort()
y.sort()
z.sort()


# Get overall min/max values for xyz respectively
# xmin = math.floor(x.min())
# xmax = math.ceil(x.max())
# ymin = math.floor(y.min())
# ymax = math.ceil(y.max())
# zmin = math.floor(z.min())
# zmax = math.ceil(z.max())
# Ignore outer 1%
# x = x[math.floor(0.01 * len(x)):math.floor(len(x) - 0.01 * len(x))]
# y = y[math.floor(0.01 * len(y)):math.floor(len(y) - 0.01 * len(y))]
# z = z[math.floor(0.01 * len(z)):math.floor(len(z) - 0.01 * len(z))]
# Z Score outlier filtering
# ZSCORE_ABS_THRESHOLD = 2.5
# x = x[abs(stats.zscore(x)) < 3]
# y = y[abs(stats.zscore(y)) < 3]
# z = z[abs(stats.zscore(z)) < 2.5]
# IQR outlier filtering
def filter_outliers_iqr(data: 'np.ndarray', factor: float = 1.5):
    q1 = np.quantile(x, 0.25)
    q3 = np.quantile(x, 0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return data[np.where((data > lower) & (data < upper))]


x_iqr_factor = 2.5
y_iqr_factor = 2.5
z_iqr_factor = 3.5
x = filter_outliers_iqr(x, factor=2.5)
y = filter_outliers_iqr(y, factor=2.5)
z = filter_outliers_iqr(z, factor=3.5)

xmin = x.min()
xmax = x.max()
ymin = y.min()
ymax = y.max()
zmin = z.min()
zmax = z.max()
minmax_filename = f'{dump_root}/{dataset_name}/minmax_values.txt'
# Create basic text file to store(remember) the used min/max values
if not os.path.isdir(f'{dump_root}/{dataset_name}'):
    os.makedirs(f'{dump_root}/{dataset_name}')
minmax_file = open(minmax_filename, 'w')
minmax_file.write(f'x: [{xmin}, {xmax}]\ny: [{ymin}, {ymax}]\nz: [{zmin}, {zmax}]')
minmax_file.write(f'\n')
minmax_file.write(f'x_iqr_factor: {x_iqr_factor}\ny_iqr_factor: {y_iqr_factor}\nz_iqr_factor: {z_iqr_factor}')
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

# --- Create Coordinate Histogram
# Save a histogram of all coordinate values to visualize the distribution.
xgroups = []
ygroups = []
zgroups = []
min = np.array([math.floor(xmin), math.floor(ymin), math.floor(zmin)]).min()
max = np.array([math.ceil(xmax), math.ceil(ymax), math.ceil(zmax)]).max()
for min_rng in range(min, max):
    xgroups.append(x[np.where((x > min_rng) & (x < min_rng + 1))[0]])
    ygroups.append(y[np.where((y > min_rng) & (y < min_rng + 1))[0]])
    zgroups.append(z[np.where((z > min_rng) & (z < min_rng + 1))[0]])
xgroup_sizes = [len(group) for group in xgroups]
ygroup_sizes = [len(group) for group in ygroups]
zgroup_sizes = [len(group) for group in zgroups]

group_labels = [f'[{g},{(g+1)}]' for g in range(min, max)]
fig = go.Figure(data=[
    go.Bar(name='X Coords', x=group_labels, y=xgroup_sizes),
    go.Bar(name='Y Coords', x=group_labels, y=ygroup_sizes),
    go.Bar(name='Z Coords', x=group_labels, y=zgroup_sizes)
])
fig.update_layout(barmode='group')
fig.show()
# histogram_filename = f'{dump_root}/{dataset_name}/coord_histogram.png'
# fig.write_image(histogram_filename)