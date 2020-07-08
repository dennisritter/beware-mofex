import numpy as np
import cv2
import json
import os
import random
import math
from datetime import datetime
import time
from pathlib import Path
import torch
from torchvision import transforms
from mofex.preprocessing.sequence import Sequence
from mofex.preprocessing.skeleton_visualizer import SkeletonVisualizer
import mofex.models.resnet as resnet
import mofex.feature_vectors as featvec
from scipy.signal import argrelextrema, savgol_filter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mofex.acm_asf_parser.amc_parser as amc_asf_parser
from mofex.preprocessing.helpers import xyz_minmax_coords
import mofex.model_loader as model_loader

# Indices constants for body parts that define normalized orientation of the skeleton
# left -> hip_left
LEFT_IDX = 1
# right -> hip_right
RIGHT_IDX = 6
# up -> lowerback
UP_IDX = 11

# Min/Max values used for the color mapping when transforming sequences to motion images
# min values are mapped to RGB(0,0,0), max values to RGB(255,255,255)
xmin, xmax = (-14.772495736531305, 14.602030756418097)
ymin, ymax = (-14.734704969722216, 14.557769829141042)
zmin, zmax = (-19.615324010444805, 19.43983405425556)

subseq_len = 8
savgol_win = 21


def locate_reps(seq_q, seq_gt, model, feature_size, preprocess):
    ## Get first frames of Ground Truth to search for similar segments in a long sequence
    # Normalize and get Motion Images
    key_gt = seq_gt[0:subseq_len]
    key_gt.norm_center_positions()
    key_gt.norm_relative_to_positions((key_gt.positions[:, LEFT_IDX, :] + key_gt.positions[:, RIGHT_IDX, :]) * 0.5)
    key_gt.norm_orientation(key_gt.positions[0, LEFT_IDX], key_gt.positions[0, RIGHT_IDX], key_gt.positions[0, UP_IDX])
    mi_gt = key_gt.to_motionimg(output_size=(256, 256), minmax_pos_x=(xmin, xmax), minmax_pos_y=(ymin, ymax), minmax_pos_z=(zmin, zmax))

    ## Split long sequence into small segments
    # Normalize and get Motion Images
    mi_q_list = []
    seq_q_split = seq_q.split(overlap=0, subseq_size=subseq_len)
    for seq_q_part in seq_q_split:
        seq_q_part.norm_center_positions()
        seq_q_part.norm_relative_to_positions((seq_q_part.positions[:, LEFT_IDX, :] + seq_q_part.positions[:, RIGHT_IDX, :]) * 0.5)
        seq_q_part.norm_orientation(seq_q_part.positions[0, LEFT_IDX], seq_q_part.positions[0, RIGHT_IDX], seq_q_part.positions[0, UP_IDX])
        mi_q_list.append(seq_q_part.to_motionimg(output_size=(256, 256), minmax_pos_x=(xmin, xmax), minmax_pos_y=(ymin, ymax), minmax_pos_z=(zmin, zmax)))

    # Make feature vectors from Motion Images
    featvec_gt = featvec.load_from_motion_imgs(motion_images=[mi_gt], model=model, feature_size=feature_size, preprocess=preprocess)
    featvec_q_list = featvec.load_from_motion_imgs(motion_images=mi_q_list, model=model, feature_size=feature_size, preprocess=preprocess)

    # Determine distances between q_sequence and split sequences
    distances = [np.linalg.norm(featvec_gt - featvec_q) for featvec_q in featvec_q_list]

    # Smoothing and counting reps
    savgol_distances = savgol_filter(distances, savgol_win, 3, mode='nearest')
    savgol_distance_maxima = argrelextrema(savgol_distances, np.greater_equal, order=5)[0]
    savgol_distance_minima = argrelextrema(savgol_distances, np.less_equal, order=5)[0]

    # Keyframe indices per frame
    keyframes = np.array(savgol_distance_minima) * subseq_len - (subseq_len * 0.5)
    return (keyframes, np.array(savgol_distances))