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

### Load Sequences
seqs = load_seqs_asf_amc(src_root, filename_asf, filename_amc)
labeled_sequences_dict = {}

x = []
y = []
z = []
for seq in seqs:
    print(f'Processing: {seq.name}')
    # Normalization
    # * Body Part Indices -> 1 = left hip; 6 = right hip
    seq.norm_center_positions()
    seq.norm_relative_to_positions((seq.positions[:, 1, :] + seq.positions[:, 6, :]) * 0.5)
    seq.norm_orientation_first_pose_frontal_to_camera(1, 6)
    # Add flattened xyz values to list respectively
    x.extend(seq.positions[:, :, 0].flatten().tolist())
    y.extend(seq.positions[:, :, 1].flatten().tolist())
    z.extend(seq.positions[:, :, 2].flatten().tolist())
    print(seq.desc)
    if seq.desc in labeled_sequences_dict.keys():
        labeled_sequences_dict[seq.desc].append(seq)
    else:
        labeled_sequences_dict[seq.desc] = [seq]

print(f"Loaded Sequence Files")
train_seqs = []
val_seqs = []
for label in labeled_sequences_dict.keys():
    if len(labeled_sequences_dict[label]) < 10:
        print(label)
    else:
        label_split = train_test_split(labeled_sequences_dict[label], test_size=0.1, random_state=42)
        train_seqs.extend(label_split[0])
        val_seqs.extend(label_split[1])

# make array from list
x = np.array(x)
y = np.array(y)
z = np.array(z)
# sort
x.sort()
y.sort()
z.sort()
# Get overall min/max values for xyz respectively
# ignore outer 1%
xmin = x[math.floor(0.01 * len(x)):].min()
xmax = x[:math.floor(len(x) - 0.01 * len(x))].max()
ymin = y[math.floor(0.01 * len(y)):].min()
ymax = y[:math.floor(len(y) - 0.01 * len(y))].max()
zmin = z[math.floor(0.01 * len(z)):].min()
zmax = z[:math.floor(len(z) - 0.01 * len(z))].max()
minmax_filename = f'{dump_root}/{dataset_name}/minmax_values.txt'

if not os.path.isdir(f'{dump_root}/{dataset_name}'):
    os.makedirs(f'{dump_root}/{dataset_name}')
minmax_file = open(minmax_filename, 'w')
minmax_file.write(f'x: [{xmin}, {xmax}]\ny: [{ymin}, {ymax}]\nz: [{zmin}, {zmax}]')
minmax_file.close()
#----------------
# Train Set
for seq in train_seqs:
    # set and create directories
    class_dir = f'{dump_root}/{dataset_name}/train/{seq.desc}'
    out = f'{class_dir}/{seq.name}.png'
    if not os.path.isdir(os.path.abspath(class_dir)):
        os.makedirs(os.path.abspath(class_dir))
    # save motion img
    img = seq.to_motionimg(output_size=(256, 256), minmax_pos_x=(xmin, xmax), minmax_pos_y=(ymin, ymax), minmax_pos_z=(zmin, zmax))
    cv2.imwrite(out, img)
# Validation Set
for seq in val_seqs:
    # set and create directories
    class_dir = f'{dump_root}/{dataset_name}/val/{seq.desc}'
    out = f'{class_dir}/{seq.name}.png'
    if not os.path.isdir(os.path.abspath(class_dir)):
        os.makedirs(os.path.abspath(class_dir))
    # save motion img
    cv2.imwrite(out, img)
    img = seq.to_motionimg(output_size=(256, 256), minmax_pos_x=(xmin, xmax), minmax_pos_y=(ymin, ymax), minmax_pos_z=(zmin, zmax))

# Save a histogram of all coordinate values to visualize the distribution.
xgroups = []
ygroups = []
zgroups = []
for min_rng in range(math.floor(xmin), math.ceil(xmax)):
    xgroups.append(x[np.where((x > min_rng) & (x < min_rng + 1))[0]])
for min_rng in range(math.floor(ymin), math.ceil(ymax)):
    ygroups.append(y[np.where((y > min_rng) & (y < min_rng + 1))[0]])
for min_rng in range(math.floor(zmin), math.ceil(zmax)):
    zgroups.append(z[np.where((z > min_rng) & (z < min_rng + 1))[0]])
xgroup_sizes = [len(group) for group in xgroups]
ygroup_sizes = [len(group) for group in ygroups]
zgroup_sizes = [len(group) for group in zgroups]

min = np.array([math.floor(xmin), math.floor(ymin), math.floor(zmin)]).min()
max = np.array([math.ceil(xmax), math.ceil(ymax), math.ceil(zmax)]).max()
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