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
from mofex.load_sequences import load_seqs_asf_amc_hdm05
from mofex.preprocessing.helpers import xyz_minmax_coords

root = './data/sequences/hdm05-122/amc/'
filename_asf = '*.asf'
# filename_amc = 'HDM_bd_walkLeft3Steps_003_120.amc'
filename_amc = 'HDM_tr_hitRHandHead_012_120.amc'
seqs = load_seqs_asf_amc_hdm05(root, filename_asf, filename_amc)

# Indices constants for body parts that define normalized orientation of the skeleton
# left -> hip_left
LEFT_IDX = 1
# right -> hip_right
RIGHT_IDX = 6
# up -> lowerback
UP_IDX = 11

for seq in seqs:
    print(f'Processing: {seq.name}')
    # Normalization
    seq.norm_center_positions()
    seq.norm_relative_to_positions((seq.positions[:, LEFT_IDX, :] + seq.positions[:, RIGHT_IDX, :]) * 0.5)
    seq.norm_orientation(seq.positions[0, LEFT_IDX], seq.positions[0, RIGHT_IDX], seq.positions[0, UP_IDX])
    sv = SkeletonVisualizer(seq)
    sv.show()
minmax_xyz = xyz_minmax_coords(seqs, [2.5, 2.5, 3.5], plot_histogram=True)
