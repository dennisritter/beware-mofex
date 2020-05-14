import numpy as np
import cv2
import json
import os
from pathlib import Path
from mofex.preprocessing.sequence import Sequence
from sklearn.model_selection import train_test_split

import time
from mofex.preprocessing.skeleton_visualizer import SkeletonVisualizer
# Root folder for JSON Sequence files
# src_root = 'data/sequences/hdm05-122/c3d'
src_root = 'data/sequences/hdm05-122/c3d'
dump_root = 'data/motion_images'
dataset_name = 'hdm05-122_90-10'

### Load Sequences
filenames = []
sequences = []
labeled_sequences_dict = {}
for filename in Path(src_root).rglob('*.c3d'):
    print(filename)
    name = str(filename).split("\\")[-1]
    desc = str(filename).split("\\")[-2]  # desc = class
    seq = Sequence.from_hdm05_c3d_file(filename, name=name, desc=desc)
    seq.norm_center_positions()
    # * 30 and 37 are position indices somewhere near left/right hip.
    # * Adjust if using other body part model! (currently for hdm05 dataset sequences)
    seq.norm_relative_to_positions((seq.positions[:, 30, :] + seq.positions[:, 37, :]) * 0.5)
    seq.norm_orientation_first_pose_frontal_to_camera(30, 37)
    # Append sequence to label key or add new label key if not present already
    # * Change the sequence loading function depending on the MoCap format (from_mir_file, from_mka_file, ...)
    if desc in labeled_sequences_dict.keys():
        labeled_sequences_dict[desc].append(seq)
    else:
        labeled_sequences_dict[desc] = [seq]
    # sequences.append(Sequence.from_hdm05_c3d_file(filename, name=name, desc=desc))
print(f"Loaded Sequence Files")
train_seqs = []
val_seqs = []
for label in labeled_sequences_dict.keys():
    label_split = train_test_split(labeled_sequences_dict[label], test_size=0.1, random_state=42)
    train_seqs.extend(label_split[0])
    val_seqs.extend(label_split[1])

# Find min/max coordinate values in whole dataset
x = []
y = []
z = []
for seq in (train_seqs + val_seqs):
    x.extend(seq.positions[:, :, 0].flatten().tolist())
    y.extend(seq.positions[:, :, 1].flatten().tolist())
    z.extend(seq.positions[:, :, 2].flatten().tolist())
x_min, y_min, z_min = np.array(x).min(), np.array(y).min(), np.array(z).min()
x_max, y_max, z_max = np.array(x).max(), np.array(y).max(), np.array(z).max()
filename = f'{dump_root}/{dataset_name}/minmax_values.txt'
if not os.path.isdir(f'{dump_root}/{dataset_name}'):
    os.makedirs(f'{dump_root}/{dataset_name}')
minmax_file = open(filename, 'w')
minmax_file.write(f'x: [{x_min}, {x_max}]\ny: [{y_min}, {y_max}]\nz: [{z_min}, {z_max}]')
minmax_file.close()

# Train Set
for seq in train_seqs:
    # set and create directories
    class_dir = f'{dump_root}/{dataset_name}/train/{seq.desc}'
    out = f'{class_dir}/{seq.name.replace(".C3D", ".png")}'
    if not os.path.isdir(os.path.abspath(class_dir)):
        os.makedirs(os.path.abspath(class_dir))
    # save motion img
    img = seq.to_motionimg(output_size=(256, 256), minmax_pos_x=(x_min, x_max), minmax_pos_y=(y_min, y_max), minmax_pos_z=(z_min, z_max))
    cv2.imwrite(out, img)
# Validation Set
for seq in val_seqs:
    # set and create directories
    class_dir = f'{dump_root}/{dataset_name}/val/{seq.desc}'
    out = f'{class_dir}/{seq.name.replace(".C3D", ".png")}'
    if not os.path.isdir(os.path.abspath(class_dir)):
        os.makedirs(os.path.abspath(class_dir))
    # save motion img
    img = seq.to_motionimg(output_size=(256, 256))
    cv2.imwrite(out, img)
