import numpy as np
import cv2
import os
import time
import math
from pathlib import Path
import numpy
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from mofex.preprocessing.helpers import xyz_minmax_coords, xyz_minmax_coords_per_bodypart
from mofex.preprocessing.skeleton_visualizer import SkeletonVisualizer
import mofex.preprocessing.normalizations as mofex_norm
import random

import mana.utils.math.normalizations as normalizations
from mana.models.sequence import Sequence
from mana.utils.data_operations.loaders.sequence_loader_mka import SequenceLoaderMKA
from mana.models.sequence_transforms import SequenceTransforms


def to_motionimg_bp_minmax(
        seq,
        output_size=(256, 256),
        minmax_per_bp: np.ndarray = None,
        show_img=False,
) -> np.ndarray:
    """ Returns a Motion Image, that represents this sequences' positions.

            Creates an Image from 3-D position data of motion sequences.
            Rows represent a body part (or some arbitrary position instance).
            Columns represent a frame of the sequence.

            Args:
                output_size (int, int): The size of the output image in pixels
                    (height, width). Default=(200,200)
                minmax_per_bp (int, int): The minimum and maximum xyx-positions.
                    Mapped to color range (0, 255) for each body part separately.
        """
    # Create Image container
    img = np.zeros((len(seq.positions[0, :]), len(seq.positions), 3), dtype='uint8')
    # 1. Map (min_pos, max_pos) range to (0, 255) Color range.
    # 2. Swap Axes of and frames(0) body parts(1) so rows represent body
    # parts and cols represent frames.
    for i, bp in enumerate(seq.positions[0]):
        bp_positions = seq.positions[:, i]
        x_colors = np.interp(bp_positions[:, 0], [minmax_per_bp[i, 0, 0], minmax_per_bp[i, 0, 1]], [0, 255])
        img[i, :, 0] = x_colors
        img[i, :, 1] = np.interp(bp_positions[:, 1], [minmax_per_bp[i, 1, 0], minmax_per_bp[i, 1, 1]], [0, 255])
        img[i, :, 2] = np.interp(bp_positions[:, 2], [minmax_per_bp[i, 2, 0], minmax_per_bp[i, 2, 1]], [0, 255])
    img = cv2.resize(img, output_size)

    if show_img:
        cv2.imshow(seq.name, img)
        print(f'Showing motion image from [{seq.name}]. Press any key to' ' close the image and continue.')
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img


# Root folder for Sequence files
src_root = './data/sequences/mka-beware-1.1/'

dump_root = 'data/motion_images'
dataset_name = 'mka-beware-1.1'
# Indices constants for body parts that define normalized orientation of the skeleton
# center -> pelvis
CENTER_IDX = 0
# left -> hip_left
LEFT_IDX = 18
# right -> hip_right
RIGHT_IDX = 22
# up -> spinenavel
UP_IDX = 1

seq_transforms = SequenceTransforms(SequenceTransforms.mka_to_iisy())
seq_loader = SequenceLoaderMKA(seq_transforms)
seq_labeled_dict = {}
seqs = []
for filename in Path(src_root).rglob('*.json'):
    # print(str(filename).replace('\\', '/').split('/'))
    filename_split = str(filename).replace('\\', '/').split('/')
    seq_name = filename_split[-1]  # The filename (eg: 'myname.json')
    seq_class = filename_split[-2]  # The class (eG: 'squat')
    # Load
    seq = seq_loader.load(path=f'{str(filename)}', name=seq_name[:-5], desc=seq_class)
    # Normalize
    seq.positions = mofex_norm.center_positions(seq.positions)
    seq.positions = normalizations.pose_position(seq.positions, seq.positions[:, CENTER_IDX, :])
    mofex_norm.orientation(seq.positions, seq.positions[0, LEFT_IDX, :], seq.positions[0, RIGHT_IDX, :], seq.positions[0, UP_IDX, :])

    # Add Sequences to class label in dictionary
    if seq.desc in seq_labeled_dict:
        seq_labeled_dict[seq.desc].append(seq)
    else:
        seq_labeled_dict[seq.desc] = [seq]
    # Also store all seqs in a list
    seqs.append(seq)
    print(f'Loaded: {seq.name} [{seq.desc}]')

# * Split sequences into chunks
chunk_sizes = [4, 8, 16, 32, 64, 128]
merge_sizes = np.arange(1, 5)
seqs_chunked = []
chunks_labeled_dict = {}
for seq_class in seq_labeled_dict:
    for i, seq in enumerate(seq_labeled_dict[seq_class]):
        # Merge three sequences
        chunk_size = random.choice(chunk_sizes)
        # Only merge if not enough sequences in list
        if chunk_size > len(seq):
            chunk_size = len(seq)
        seq_chunks = seq.split(overlap=0, subseq_size=chunk_size)

        for j, seq_chunk in enumerate(seq_chunks):
            seq_chunk.name = f'{seq_class}_{i}_n{chunk_size}_{j}'
            seq_chunk.desc = f'{seq_class}'
            seqs_chunked.append(seq_chunk)

            if seq_chunk.desc in chunks_labeled_dict:
                chunks_labeled_dict[seq_chunk.desc].append(seq_chunk)
            else:
                chunks_labeled_dict[seq_chunk.desc] = [seq_chunk]
    idx = 0
    while idx < len(seq_labeled_dict[seq_class]) - merge_sizes.max():
        seq = seq_labeled_dict[seq_class][idx]
        n_seqs_to_merge = random.choice(merge_sizes)
        for i in range(1, n_seqs_to_merge):
            seq.append(seq_labeled_dict[seq_class][idx + i])
        seq.name = f'{seq_class}_{idx}_m{n_seqs_to_merge}'
        seq.desc = f'{seq_class}'
        if seq.desc in chunks_labeled_dict:
            chunks_labeled_dict[seq.desc].append(seq)
        else:
            chunks_labeled_dict[seq.desc] = [seq]
        idx += n_seqs_to_merge

seqs = seqs_chunked
seq_labeled_dict = chunks_labeled_dict
# Filter Outliers and get xyz_minmax values
iqr_factor_x = 2.5
iqr_factor_y = 2.5
iqr_factor_z = 2.5

minmax_xyz = xyz_minmax_coords_per_bodypart(seqs, [iqr_factor_x, iqr_factor_y, iqr_factor_z], plot_histogram=False)
# print(f'minmax_xyz:\n{minmax_xyz}')
# xmin, xmax = minmax_xyz[0]
# ymin, ymax = minmax_xyz[1]
# zmin, zmax = minmax_xyz[2]

# Split sequences into Train/Val
train_seqs = []
val_seqs = []
# Split class sequences into train/val sets
for label in seq_labeled_dict:
    label_split = train_test_split(seq_labeled_dict[label], test_size=0.1, random_state=42)
    train_seqs.extend(label_split[0])
    val_seqs.extend(label_split[1])

minmax_filename = f'{dump_root}/{dataset_name}/minmax_values.txt'
# Create basic text file to store(remember) the used min/max values
if not os.path.isdir(f'{dump_root}/{dataset_name}'):
    os.makedirs(f'{dump_root}/{dataset_name}')
minmax_file = open(minmax_filename, 'w')
# minmax_file.write(f'x: [{xmin}, {xmax}]\ny: [{ymin}, {ymax}]\nz: [{zmin}, {zmax}]')
minmax_file.write(f'{minmax_xyz}')
minmax_file.write(f'\n')
minmax_file.write(f'x_iqr_factor: {iqr_factor_x}\ny_iqr_factor: {iqr_factor_y}\nz_iqr_factor: {iqr_factor_z}')
minmax_file.close()

# --- Create Motion Images
# Train Set
for seq in train_seqs:
    # set and create directories
    class_dir = f'{dump_root}/{dataset_name}/train/{seq.desc}'
    out = f'{class_dir}/{seq.name}.png'
    if not os.path.isdir(os.path.abspath(class_dir)):
        os.makedirs(os.path.abspath(class_dir))
    # Create and save Motion Image with defined output size and X,Y,Z min/max values to map respective color channels to positions.
    # (xmin, xmax) -> RED(0, 255), (ymin, ymax) -> GREEN(0, 255), (zmin, zmax) -> BLUE(0, 255),
    # img = seq.to_motionimg(output_size=(256, 256), minmax_pos_x=(xmin, xmax), minmax_pos_y=(ymin, ymax), minmax_pos_z=(zmin, zmax))
    img = to_motionimg_bp_minmax(seq, output_size=(256, 256), minmax_per_bp=minmax_xyz)
    cv2.imwrite(out, img)
# Validation Set
for seq in val_seqs:
    # set and create directories
    class_dir = f'{dump_root}/{dataset_name}/val/{seq.desc}'
    out = f'{class_dir}/{seq.name}.png'
    if not os.path.isdir(os.path.abspath(class_dir)):
        os.makedirs(os.path.abspath(class_dir))
    # Create and save Motion Image with defined output size and X,Y,Z min/max values to map respective color channels to positions.
    # (xmin, xmax) -> RED(0, 255), (ymin, ymax) -> GREEN(0, 255), (zmin, zmax) -> BLUE(0, 255),
    # img = seq.to_motionimg(output_size=(256, 256), minmax_pos_x=(xmin, xmax), minmax_pos_y=(ymin, ymax), minmax_pos_z=(zmin, zmax))
    img = to_motionimg_bp_minmax(seq, output_size=(256, 256), minmax_per_bp=minmax_xyz)
    cv2.imwrite(out, img)