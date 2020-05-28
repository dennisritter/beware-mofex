import numpy as np
import cv2
import json
import os
import random
import math
import time
from scipy import stats
from datetime import datetime
from pathlib import Path
import torch
from torchvision import transforms
from mofex.preprocessing.sequence import Sequence
from mofex.preprocessing.skeleton_visualizer import SkeletonVisualizer
import mofex.models.resnet as resnet
import mofex.feature_vectors as featvec
import plotly.graph_objects as go
from mofex.load_sequences import load_seqs_asf_amc

root = './data/sequences/hdm05-3/amc/'
filename_asf = '*.asf'
filename_amc = '*.amc'
seqs = load_seqs_asf_amc(root, filename_asf, filename_amc)

x = []
y = []
z = []

for seq in seqs:
    print(f'Processing: {seq.name}')
    # Normalization
    seq.norm_center_positions()
    seq.norm_relative_to_positions((seq.positions[:, 1, :] + seq.positions[:, 6, :]) * 0.5)
    # sv_orig = SkeletonVisualizer(seq)
    # sv_orig.show()
    seq.norm_orientation_first_pose_frontal_to_camera(1, 6)
    # Add flattened xyz values to list respectively
    x.extend(seq.positions[:, :, 0].flatten().tolist())
    y.extend(seq.positions[:, :, 1].flatten().tolist())
    z.extend(seq.positions[:, :, 2].flatten().tolist())

# make array from list
x = np.array(x)
y = np.array(y)
z = np.array(z)
# sort
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
# xmin = xmin[math.floor(0.01 * len(x)):]
# xmax = xmax[:math.floor(len(x) - 0.01 * len(x))]
# ymin = ymin[math.floor(0.01 * len(y)):]
# ymax = ymax[:math.floor(len(y) - 0.01 * len(y))]
# zmin = zmin[math.floor(0.01 * len(z)):]
# zmax = zmax[:math.floor(len(z) - 0.01 * len(z))]
# Z Scores
ZSCORE_ABS_THRESHOLD = 3
shape_full_x = x.shape
shape_full_y = y.shape
shape_full_z = z.shape
x = x[abs(stats.zscore(x)) < ZSCORE_ABS_THRESHOLD]
y = y[abs(stats.zscore(y)) < ZSCORE_ABS_THRESHOLD]
z = z[abs(stats.zscore(z)) < ZSCORE_ABS_THRESHOLD]
shape_filtered_x = x.shape
shape_filtered_y = y.shape
shape_filtered_z = z.shape
xmin = math.floor(x.min())
xmax = math.ceil(x.max())
ymin = math.floor(y.min())
ymax = math.ceil(y.max())
zmin = math.floor(z.min())
zmax = math.ceil(z.max())
print(f'{(xmin, xmax)}, {(ymin, ymax)}, {(zmin, zmax)}')
xgroups = []
ygroups = []
zgroups = []
min = np.array([xmin, ymin, zmin]).min()
max = np.array([xmax, ymax, zmax]).max()
for min_rng in range(min, max):
    xgroups.append(x[np.where((x > min_rng) & (x < min_rng + 1))[0]])
    ygroups.append(y[np.where((y > min_rng) & (y < min_rng + 1))[0]])
    zgroups.append(z[np.where((z > min_rng) & (z < min_rng + 1))[0]])
xgroup_sizes = [len(group) for group in xgroups]
ygroup_sizes = [len(group) for group in ygroups]
zgroup_sizes = [len(group) for group in zgroups]

group_labels = [f'[{g},{(g+1)}]' for g in range(min, max)]
fig = go.Figure(data=[go.Bar(x=group_labels, y=xgroup_sizes), go.Bar(x=group_labels, y=ygroup_sizes), go.Bar(x=group_labels, y=zgroup_sizes)])
fig.update_layout(barmode='group')
fig.show()
# sv = SkeletonVisualizer(seqs[02])
# sv.show()
