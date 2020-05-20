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
from mofex.load_sequences import load_seqs_asf_amc

# def load_seqs(root, regex_str_asf, regex_str_amc):
#     seqs = []
#     print(f'Loading sequences from:\nroot: {root}\nASF pattern: {regex_str_asf}\nAMC pattern: {regex_str_amc}')
#     for amc_path in Path(root).rglob(regex_str_amc):
#         class_dir = '/'.join(str(amc_path).split("\\")[:-1])
#         amc_file = str(amc_path).split("\\")[-1]
#         asf_file = amc_file[0:7]
#         asf_path = class_dir + asf_file
#         seqs.append(Sequence.from_hdm05_asf_amc_files(asf_path, amc_path, name=amc_file, desc=class_dir.split('/')[-1]))
#         print(f'loaded: {seqs[-1].name} -> {seqs[-1].desc}')
#     return seqs

root = './data/sequences/hdm05-122/amc/'
filename_asf = '*.asf'
filename_amc = '*.amc'
seqs = load_seqs_asf_amc(root, filename_asf, filename_amc)

x = []
y = []
z = []
for seq in seqs:
    print(f'Processing: {seq.name}')
    seq.norm_center_positions()
    seq.norm_relative_to_positions((seq.positions[:, 1, :] + seq.positions[:, 6, :]) * 0.5)
    seq.norm_orientation_first_pose_frontal_to_camera(1, 6)
    x.extend(seq.positions[:, :, 0].flatten().tolist())
    y.extend(seq.positions[:, :, 1].flatten().tolist())
    z.extend(seq.positions[:, :, 2].flatten().tolist())

x = np.array(x)
y = np.array(y)
z = np.array(z)
x.sort()
y.sort()
z.sort()
xmin = math.floor(x.min())
xmax = math.ceil(x.max())
ymin = math.floor(y.min())
ymax = math.ceil(y.max())
zmin = math.floor(z.min())
zmax = math.ceil(z.max())
xgroups = []
ygroups = []
zgroups = []
for min_rng in range(xmin, xmax):
    xgroups.append(x[np.where((x > min_rng) & (x < min_rng + 1))[0]])
for min_rng in range(ymin, ymax):
    ygroups.append(y[np.where((y > min_rng) & (y < min_rng + 1))[0]])
for min_rng in range(zmin, zmax):
    zgroups.append(z[np.where((z > min_rng) & (z < min_rng + 1))[0]])
xgroup_sizes = [len(group) for group in xgroups]
ygroup_sizes = [len(group) for group in ygroups]
zgroup_sizes = [len(group) for group in zgroups]

min = np.array([xmin, ymin, zmin]).min()
max = np.array([xmax, ymax, zmax]).max()
group_labels = [f'[{g},{(g+1)}]' for g in range(min, max)]
fig = go.Figure(data=[go.Bar(x=group_labels, y=xgroup_sizes), go.Bar(x=group_labels, y=ygroup_sizes), go.Bar(x=group_labels, y=zgroup_sizes)])
fig.update_layout(barmode='group')
fig.show()

sv = SkeletonVisualizer(seqs[0])
sv.show()