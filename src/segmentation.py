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
"""
Segmentation Approach
1. Take 1/2/4/8/16/32/64 (key_len) frame motion sequence key_gt of ground truth sequence seq_gt
2. Segment long sequence (seq_q) into short sequences (key_q) of length key_len
3. Compare key_gt and key_q.
4. Show distances in line graph   
"""

### Find/Locate subsequences in long_seq that are similar to q_seq

asf_path = './data/sequences/hdm05-122/amc/squat1Reps/HDM_bd.asf'
amc_path = './data/sequences/hdm05-122/amc/squat1Reps/HDM_bd_squat1Reps_001_120.amc'
seq_gt = Sequence.from_hdm05_asf_amc_files(asf_path, amc_path)

## Analyze one sequence
asf_path = './data/sequences/hdm05-122/amc/squat3Reps/HDM_tr.asf'
# amc_path = './data/sequences/hdm05-122/amc/squat1Reps/HDM_tr_squat1Reps_049_120.amc'
amc_path = './data/sequences/hdm05-122/amc/squat3Reps/HDM_tr_squat3Reps_013_120.amc'
seq_q = Sequence.from_hdm05_asf_amc_files(asf_path, amc_path)

## Analyze long, appended sequence
# root = './data/sequences/hdm05-122/amc/squat3Reps/'
# seqs = []
# for filename in Path(root).rglob('HDM_tr*.amc'):
#     print(filename)
#     seqs.append(Sequence.from_hdm05_asf_amc_files(asf_path, filename))
# for seq in seqs[1:]:
#     seqs[0].append(seq)
# seq_q = seqs[0]

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


def split_sequence(long_seq: 'Sequence', overlap: float = 0.0, subseq_size: int = 1) -> list:
    if overlap < 0.0 or overlap > 0.99:
        raise ValueError('overlap parameter must be a value between [0.0, 0.99]')
    step_size = int(subseq_size - subseq_size * overlap)
    if step_size == 0:
        raise ValueError('The formula int(subseq_size - subseq_size * overlap) should not equal 0. Choose params that fulfill this condition.')
    n_steps = math.floor((len(long_seq) - subseq_size) / step_size)
    seqs = [long_seq[step * step_size:step * step_size + subseq_size] for step in range(0, n_steps)]
    return seqs


subseq_len_list = [8]
savgol_windows = [11, 21, 31, 41, 51]

fig_dist = make_subplots(rows=len(subseq_len_list), cols=1)
fig_savgol = make_subplots(rows=len(savgol_windows), cols=1)
for row, subseq_len in enumerate(subseq_len_list):
    start = time.time()

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
    seq_q_split = split_sequence(seq_q, overlap=0, subseq_size=subseq_len)
    for seq_q_part in seq_q_split:
        seq_q_part.norm_center_positions()
        seq_q_part.norm_relative_to_positions((seq_q_part.positions[:, LEFT_IDX, :] + seq_q_part.positions[:, RIGHT_IDX, :]) * 0.5)
        seq_q_part.norm_orientation(seq_q_part.positions[0, LEFT_IDX], seq_q_part.positions[0, RIGHT_IDX], seq_q_part.positions[0, UP_IDX])
        mi_q_list.append(seq_q_part.to_motionimg(output_size=(256, 256), minmax_pos_x=(xmin, xmax), minmax_pos_y=(ymin, ymax), minmax_pos_z=(zmin, zmax)))

    # Get Feature Vectors
    dataset_name = 'hdm05-122_90-10'
    # CNN Model name -> model_dataset-numclasses_train-val-ratio
    model_name = 'resnet101_hdm05-122_90-10'
    # The CNN Model for Feature Vector generation
    model = model_loader.load_trained_model(model_name=model_name,
                                            remove_last_layer=True,
                                            state_dict_path=f'./data/trained_models/{dataset_name}/{model_name}_e25.pt')
    # The models output size (Feature Vector length)
    feature_size = 2048
    # Transforms
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Make feature vectors from Motion Images
    featvec_gt = featvec.load_from_motion_imgs(motion_images=[mi_gt], model=model, feature_size=feature_size, preprocess=preprocess)
    featvec_q_list = featvec.load_from_motion_imgs(motion_images=mi_q_list, model=model, feature_size=feature_size, preprocess=preprocess)

    # Determine distances between q_sequence and split sequences
    distances = [np.linalg.norm(featvec_gt - featvec_q) for featvec_q in featvec_q_list]

    end = time.time()
    elapsed = end - start

    x_data = np.arange(len(distances))
    y_data = distances
    fig_dist.append_trace(go.Scatter(x=x_data, y=y_data, name=f'subseq_len = {subseq_len}'), row=row + 1, col=1)
    print(f'Subsequence length: {subseq_len} ')
    print(f'Measured distances: {len(distances)} ')
    print(f'Computation time: {elapsed}s ')

    ## Smoothing and counting reps
    for sav_i, savgol_win in enumerate(savgol_windows):
        savgol_distances = savgol_filter(distances, savgol_win, 3, mode='nearest')
        savgol_distance_maxima = argrelextrema(savgol_distances, np.greater_equal, order=5)[0]
        savgol_distance_minima = argrelextrema(savgol_distances, np.less_equal, order=5)[0]
        print(f'savgol_distance_minima: {savgol_distance_minima}')
        print(f'savgol_distance_maxima: {savgol_distance_maxima}')

        x_data_savgol = x_data = np.arange(len(distances))
        y_data_savgol = savgol_distances
        fig_savgol.append_trace(go.Scatter(x=x_data_savgol, y=y_data_savgol, name=f'savgol_win = {savgol_win}, subseq_len = {subseq_len}'),
                                row=sav_i + 1,
                                col=1)
        # Plot Minima
        min_dists = [savgol_distances[idx] for idx in savgol_distance_minima]
        fig_savgol.append_trace(
            go.Scatter(x=savgol_distance_minima, y=min_dists, mode='markers', marker_color='red'),
            row=sav_i + 1,
            col=1,
        )

fig_dist.update_layout(height=500 * len(subseq_len_list),
                       width=1000,
                       title_text="Distances between first n frames of seq_gt and all segments of seq_q for different subseq length.")
fig_dist.show()
fig_savgol.update_layout(height=300 * len(savgol_windows), width=1000, title_text="Savgol Smoothed distances and key markings. (3 Squat Repetition Sequence)")
fig_savgol.show()