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
from mofex.load_sequences import load_seqs_asf_amc_cmu

root = './data/sequences/cmu-30/amc/'
filename_asf = '*.asf'
# filename_amc = 'HDM_bd_walkLeft3Steps_003_120.amc'
filename_amc = '*.amc'
seqs = load_seqs_asf_amc_cmu(root, filename_asf, filename_amc)

# Indices constants for body parts that define normalized orientation of the skeleton
# left -> hip_left
LEFT_IDX = 1
# right -> hip_right
RIGHT_IDX = 6
# up -> lowerback
UP_IDX = 11

x = []
y = []
z = []

for seq in seqs:
    print(f'Processing: {seq.name}')
    # Normalization
    seq.norm_center_positions()
    seq.norm_relative_to_positions((seq.positions[:, LEFT_IDX, :] + seq.positions[:, RIGHT_IDX, :]) * 0.5)
    seq.norm_orientation(seq.positions[0, LEFT_IDX], seq.positions[0, RIGHT_IDX], seq.positions[0, UP_IDX])
    # sv = SkeletonVisualizer(seq)
    # sv.show()
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


iqr_factor = 2.5
x = filter_outliers_iqr(x, factor=iqr_factor)
y = filter_outliers_iqr(y, factor=iqr_factor)
z = filter_outliers_iqr(z, factor=3.5)

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