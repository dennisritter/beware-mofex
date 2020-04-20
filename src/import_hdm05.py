import numpy as np
import cv2
import json
import os
import random
from ezc3d import c3d
import scipy.io
from datetime import datetime
from pathlib import Path
import torch
from torchvision import transforms
import plotly.graph_objects as go
from mofex.preprocessing.sequence import Sequence
import mofex.preprocessing.normalizations as norm
from mofex.preprocessing.skeleton_visualizer import SkeletonVisualizer
import mofex.models.resnet as resnet
import mofex.feature_vectors as featvec

path = 'data/sequences/hdm05/c3d/jogRightCircle6StepsRstart'
### Load Movi data

# for filename in Path(path).rglob('HDM_bd_cartwheelLHandStart1Reps_001_120.C3D'):
#     with open(filename, 'rb') as handle:
#         reader = c3d.Reader(handle)
#         for i, (points, analog) in enumerate(reader.read_frames()):
#             print('Frame {}: {}'.format(i, points.round(2)))
for filename in Path(path).rglob('*.C3D'):
    name = str(filename).split("\\")[-1]
    # with open(filename, 'rb') as handle:
    print(f"Loading [{filename}]")
    c3d_object = c3d(str(filename))
    # print(c3d_object['parameters']['POINT'])
    param = c3d_object['parameters'].keys()
    subjects = c3d_object['parameters']['SUBJECTS']
    points = c3d_object['parameters']['POINT']
    analog = c3d_object['parameters']['ANALOG']
    positions = c3d_object['data']['points']
    labels = c3d_object['data']['analogs']
    data = c3d_object['data']
    a = [
        '*0', '*1', '*2', 'Bastian:LFHD', 'Bastian:RFHD', 'Bastian:LBHD', 'Bastian:RBHD', 'Bastian:C7', 'Bastian:T10', 'Bastian:CLAV', 'Bastian:STRN',
        'Bastian:RBAC', 'Bastian:LSHO', 'Bastian:LUPA', 'Bastian:LELB', 'Bastian:LFRM', 'Bastian:LWRA', 'Bastian:LWRB', 'Bastian:LFIN', 'Bastian:RSHO',
        'Bastian:RUPA', 'Bastian:RELB', 'Bastian:RFRM', 'Bastian:RWRA', 'Bastian:RWRB', 'Bastian:RFIN', 'Bastian:LFWT', 'Bastian:RFWT', 'Bastian:LBWT',
        'Bastian:RBWT', 'Bastian:LTHI', 'Bastian:LKNE', 'Bastian:LSHN', 'Bastian:LANK', 'Bastian:LHEE', 'Bastian:LTOE', 'Bastian:LMT5', 'Bastian:RTHI',
        'Bastian:RKNE', 'Bastian:RSHN', 'Bastian:RANK', 'Bastian:RHEE', 'Bastian:RTOE', 'Bastian:RMT5'
    ]

    positions = positions.swapaxes(0, 2)[:, :, :3]
    positions = norm.center_positions(positions)
    featvec.motion_image_from_3d_positions(positions, name=name, show_img=True)
# def _make_joint_traces():
#     pos = positions
#     frame = 22
#     trace_joints = go.Scatter3d(x=pos[0, :, frame], y=pos[1, :, frame], z=pos[2, :, frame], mode="markers", marker=dict(color="royalblue", size=5))
#     return [trace_joints]

# fig = go.Figure(data=_make_joint_traces())
# fig.write_html('skeleton.html', auto_open=True)