import numpy as np
import cv2
import json
import os
import random
import math
from datetime import datetime
from pathlib import Path
import torch
from torchvision import transforms
from mofex.preprocessing.sequence import Sequence
from mofex.preprocessing.skeleton_visualizer import SkeletonVisualizer
import mofex.models.resnet as resnet
import mofex.feature_vectors as featvec
import plotly.graph_objects as go
import mofex.acm_asf_parser.amc_parser as amc_asf_parser

### Find/Locate subsequences in long_seq that are similar to q_seq

asf_path = './data/sequences/hdm05-122/amc/squat1Reps/HDM_bd.asf'
amc_path = './data/sequences/hdm05-122/amc/squat1Reps/HDM_bd_squat1Reps_001_120.amc'
q_seq = Sequence.from_hdm05_asf_amc_files(asf_path, amc_path)

asf_path = './data/sequences/hdm05-122/amc/squat3Reps/HDM_bd.asf'
amc_path = './data/sequences/hdm05-122/amc/squat3Reps/HDM_bd_squat3Reps_001_120.amc'
long_seq = Sequence.from_hdm05_asf_amc_files(asf_path, amc_path)

# Indices constants for body parts that define normalized orientation of the skeleton
# left -> hip_left
LEFT_IDX = 1
# right -> hip_right
RIGHT_IDX = 6
# up -> lowerback
UP_IDX = 11


def split_sequence(long_seq: 'Sequence', overlap: float = 0.8, subseq_size: int = 100) -> list:
    if overlap < 0.0 or overlap > 0.99:
        raise ValueError('overlap parameter must be a value between [0.0, 0.99]')
    step_size = int(subseq_size - subseq_size * overlap)
    n_steps = math.floor((len(long_seq) - subseq_size) / step_size)
    seqs = [long_seq[step * step_size:step * step_size + subseq_size] for step in range(0, n_steps)]
    return seqs


q_seq.norm_center_positions()
q_seq.norm_relative_to_positions((q_seq.positions[:, LEFT_IDX, :] + q_seq.positions[:, RIGHT_IDX, :]) * 0.5)
q_seq.norm_orientation(q_seq.positions[0, LEFT_IDX], q_seq.positions[0, RIGHT_IDX], q_seq.positions[0, UP_IDX])
q_seq.to_motionimg()
seqs = split_sequence(long_seq, overlap=0.8, subseq_size=len(q_seq))

distances = []
for seq in seqs:
    seq.norm_center_positions()
    seq.norm_relative_to_positions((seq.positions[:, LEFT_IDX, :] + seq.positions[:, RIGHT_IDX, :]) * 0.5)
    seq.norm_orientation(seq.positions[0, LEFT_IDX], seq.positions[0, RIGHT_IDX], seq.positions[0, UP_IDX])

# sv = SkeletonVisualizer(seq)
# sv.show()