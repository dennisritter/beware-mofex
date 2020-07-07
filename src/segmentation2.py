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
from mofex.preprocessing.helpers import xyz_minmax_coords
import mofex.model_loader as model_loader

### Find/Locate subsequences in long_seq that are similar to q_seq

asf_path = './data/sequences/hdm05-122/amc/squat1Reps/HDM_bd.asf'
amc_path = './data/sequences/hdm05-122/amc/squat1Reps/HDM_bd_squat1Reps_001_120.amc'
q_seq = Sequence.from_hdm05_asf_amc_files(asf_path, amc_path)

asf_path = './data/sequences/hdm05-122/amc/squat3Reps/HDM_bd.asf'
amc_path = './data/sequences/hdm05-122/amc/squat3Reps/HDM_bd_squat3Reps_001_120.amc'
long_seq = Sequence.from_hdm05_asf_amc_files(asf_path, amc_path)

# We need a list of all sequences to determine minmax values of coordinates
all_seqs = [q_seq, long_seq]

# Indices constants for body parts that define normalized orientation of the skeleton
# left -> hip_left
LEFT_IDX = 1
# right -> hip_right
RIGHT_IDX = 6
# up -> lowerback
UP_IDX = 11

# Determine minmax valuzes for motion image mapping
# xyz_minmax = xyz_minmax_coords(all_seqs, [2.5, 2.5, 3.5])
# xmin, xmax = xyz_minmax[0]
# ymin, ymax = xyz_minmax[1]
# zmin, zmax = xyz_minmax[2]
xmin, xmax = (-14.772495736531305, 14.602030756418097)
ymin, ymax = (-14.734704969722216, 14.557769829141042)
zmin, zmax = (-19.615324010444805, 19.43983405425556)


def split_sequence(long_seq: 'Sequence', overlap: float = 0.8, subseq_size: int = 100) -> list:
    if overlap < 0.0 or overlap > 0.99:
        raise ValueError('overlap parameter must be a value between [0.0, 0.99]')
    step_size = int(subseq_size - subseq_size * overlap)
    n_steps = math.floor((len(long_seq) - subseq_size) / step_size)
    seqs = [long_seq[step * step_size:step * step_size + subseq_size] for step in range(0, n_steps)]
    return seqs


# Normalize and get Motion Images
q_seq.norm_center_positions()
q_seq.norm_relative_to_positions((q_seq.positions[:, LEFT_IDX, :] + q_seq.positions[:, RIGHT_IDX, :]) * 0.5)
q_seq.norm_orientation(q_seq.positions[0, LEFT_IDX], q_seq.positions[0, RIGHT_IDX], q_seq.positions[0, UP_IDX])
q_mimg = q_seq.to_motionimg(output_size=(256, 256), minmax_pos_x=(xmin, xmax), minmax_pos_y=(ymin, ymax), minmax_pos_z=(zmin, zmax))
q_seqs = split_sequence(q_seq, overlap=0, subseq_size=16)
seqs = split_sequence(long_seq, overlap=0, subseq_size=16)

q_mimgs = []
for seq in q_seqs:
    seq.norm_center_positions()
    seq.norm_relative_to_positions((seq.positions[:, LEFT_IDX, :] + seq.positions[:, RIGHT_IDX, :]) * 0.5)
    seq.norm_orientation(seq.positions[0, LEFT_IDX], seq.positions[0, RIGHT_IDX], seq.positions[0, UP_IDX])
    q_mimgs.append(seq.to_motionimg(output_size=(256, 256), minmax_pos_x=(xmin, xmax), minmax_pos_y=(ymin, ymax), minmax_pos_z=(zmin, zmax)))

split_mimgs = []
for seq in seqs:
    seq.norm_center_positions()
    seq.norm_relative_to_positions((seq.positions[:, LEFT_IDX, :] + seq.positions[:, RIGHT_IDX, :]) * 0.5)
    seq.norm_orientation(seq.positions[0, LEFT_IDX], seq.positions[0, RIGHT_IDX], seq.positions[0, UP_IDX])
    split_mimgs.append(seq.to_motionimg(output_size=(256, 256), minmax_pos_x=(xmin, xmax), minmax_pos_y=(ymin, ymax), minmax_pos_z=(zmin, zmax)))

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

q_featvecs = featvec.load_from_motion_imgs(motion_images=q_mimgs, model=model, feature_size=feature_size, preprocess=preprocess)
split_featvecs = featvec.load_from_motion_imgs(motion_images=split_mimgs, model=model, feature_size=feature_size, preprocess=preprocess)

# Determine distances between q_sequence and split sequences
# distances = [np.linalg.norm(q_featvec - split_featvec) for split_featvec in split_featvecs]
distances = []
for q_featvec in q_featvecs:
    distances.append([np.linalg.norm(q_featvec - split_featvec) for split_featvec in split_featvecs])
distances = np.array(distances)
similarity_matrix = np.around(1 - distances / distances.max(), decimals=1)
print(similarity_matrix)
similarity_matrix_img = cv2.resize(similarity_matrix, (256, 256))
cv2.imshow('img', similarity_matrix_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# print(similarity_matrix)
# cv2.imshow('img', q_mimg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# for i, img in enumerate(split_mimgs):
#     print(distances[i])
#     cv2.imshow('img', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()