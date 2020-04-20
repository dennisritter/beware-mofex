import numpy as np
import cv2
import json
import os
import random
import c3d
import scipy.io
from datetime import datetime
from pathlib import Path
import torch
from torchvision import transforms
from mofex.preprocessing.sequence import Sequence
from mofex.preprocessing.skeleton_visualizer import SkeletonVisualizer
import mofex.models.resnet as resnet
import mofex.feature_vectors as featvec

path = 'data/sequences/hdm05/c3d'
### Load Movi data

for filename in Path(path).rglob('HDM_bd_cartwheelLHandStart1Reps_001_120.C3D'):
    with open(filename, 'rb') as handle:
        reader = c3d.Reader(handle)
        for i, (points, analog) in enumerate(reader.read_frames()):
            print('Frame {}: {}'.format(i, points.round(2)))
# for filename in Path(path).rglob('HDM_bd_cartwheelLHandStart1Reps_001_120.C3D'):
#     name = str(filename).split("\\")[-1]
#     # with open(filename, 'rb') as handle:
#     print(f"Loading [{filename}]")
#     c = c3d(filename)
#     print(c['parameters']['POINT']['USED']['value'][0])
#     point_data = c['data']['points']
#     points_residuals = c['data']['meta_points']['residuals']
#     analog_data = c['data']['analogs']
