import numpy as np
import cv2
import json
import os
import random
import math
import time
from datetime import datetime
from pathlib import Path
import torch
from torchvision import transforms
from mofex.preprocessing.sequence import Sequence
from mofex.preprocessing.skeleton_visualizer import SkeletonVisualizer
import mofex.models.resnet as resnet
import mofex.feature_vectors as featvec
import plotly.graph_objects as go


def load_seqs(root, regex_str):
    seqs = []
    for filename in Path(root).rglob(regex_str):
        print(f'Loading Sequence from file: {filename}')
        seqs.append(Sequence.from_hdm05_c3d_file(filename))
    return seqs


root = 'data/sequences/hdm05-10/c3d/'
filename = '*.C3D'
seqs = load_seqs(root, filename)

x = []
y = []
z = []
for seq in seqs:
    seq.norm_center_positions()
    # seq.norm_relative_to_positions((seq.positions[:, 30, :] + seq.positions[:, 37, :]) * 0.5)
    seq.norm_orientation_first_pose_frontal_to_camera(30, 37)
    x.extend(seq.positions[:, :, 0].flatten().tolist())
    y.extend(seq.positions[:, :, 1].flatten().tolist())
    z.extend(seq.positions[:, :, 2].flatten().tolist())
x.sort()
y.sort()
z.sort()
x = np.array(x)
y = np.array(y)
z = np.array(z)
xmin = math.floor(x.min() / 100)
xmax = math.ceil(x.max() / 100)
ymin = math.floor(y.min() / 100)
ymax = math.ceil(y.max() / 100)
zmin = math.floor(z.min() / 100)
zmax = math.ceil(z.max() / 100)
xgroups = []
ygroups = []
zgroups = []
for min_rng in range(xmin, xmax):
    xgroups.append(x[np.where((x / 100 > min_rng) & (x / 100 < min_rng + 1)[0])])
for min_rng in range(ymin, ymax):
    ygroups.append(y[np.where((y / 100 > min_rng) & (y / 100 < min_rng + 1)[0])])
for min_rng in range(zmin, zmax):
    zgroups.append(z[np.where((z / 100 > min_rng) & (z / 100 < min_rng + 1)[0])])
xgroup_sizes = [len(group) for group in xgroups]
ygroup_sizes = [len(group) for group in ygroups]
zgroup_sizes = [len(group) for group in zgroups]

min = np.array([xmin, ymin, zmin]).min()
max = np.array([xmax, ymax, zmax]).max()
group_labels = [f'[{g * 100},{(g+1) * 100}]' for g in range(min, max)]
fig = go.Figure(data=[go.Bar(x=group_labels, y=xgroup_sizes), go.Bar(x=group_labels, y=ygroup_sizes), go.Bar(x=group_labels, y=zgroup_sizes)])
fig.update_layout(barmode='group')
fig.show()

sv = SkeletonVisualizer(seqs[-1])
sv.show()