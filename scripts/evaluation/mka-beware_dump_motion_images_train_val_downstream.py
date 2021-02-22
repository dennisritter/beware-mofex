import numpy as np
import cv2
import json
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from mofex.preprocessing.helpers import xyz_minmax_coords, xyz_minmax_coords_per_bodypart
import mofex.preprocessing.normalizations as mofex_norm

import mana.utils.math.normalizations as normalizations
from mana.utils.data_operations.loaders.sequence_loader_mka import SequenceLoaderMKA
from mana.models.sequence_transforms import SequenceTransforms

_format = {
    "Pelvis": 0,
    "SpineNavel": 1,
    "SpineChest": 2,
    "Neck": 3,
    "ClavicleLeft": 4,
    "ShoulderLeft": 5,
    "ElbowLeft": 6,
    "WristLeft": 7,
    "HandLeft": 8,
    "HandTipLeft": 9,
    "ThumbLeft": 10,
    "ClavicleRight": 11,
    "ShoulderRight": 12,
    "ElbowRight": 13,
    "WristRight": 14,
    "HandRight": 15,
    "HandTipRight": 16,
    "ThumbRight": 17,
    "HipLeft": 18,
    "KneeLeft": 19,
    "AnkleLeft": 20,
    "FootLeft": 21,
    "HipRight": 22,
    "KneeRight": 23,
    "AnkleRight": 24,
    "FootRight": 25,
    "Head": 26,
    "Nose": 27,
    "EyeLeft": 28,
    "EarLeft": 29,
    "EyeRight": 30,
    "EarRight": 31
}


def positions_to_list(positions: np.ndarray):
    positions = np.reshape(positions, (len(positions), -1))
    positions = [frame.tolist() for frame in positions]
    return positions


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
    img = np.zeros((len(seq.positions[0, :]), len(seq.positions), 3),
                   dtype='uint8')
    # 1. Map (min_pos, max_pos) range to (0, 255) Color range.
    # 2. Swap Axes of and frames(0) body parts(1) so rows represent body
    # parts and cols represent frames.
    for i, bp in enumerate(seq.positions[0]):
        bp_positions = seq.positions[:, i]
        x_colors = np.interp(bp_positions[:, 0],
                             [minmax_per_bp[i, 0, 0], minmax_per_bp[i, 0, 1]],
                             [0, 255])
        img[i, :, 0] = x_colors
        img[i, :,
            1] = np.interp(bp_positions[:, 1],
                           [minmax_per_bp[i, 1, 0], minmax_per_bp[i, 1, 1]],
                           [0, 255])
        img[i, :,
            2] = np.interp(bp_positions[:, 2],
                           [minmax_per_bp[i, 2, 0], minmax_per_bp[i, 2, 1]],
                           [0, 255])
    img = cv2.resize(img, output_size)

    if show_img:
        cv2.imshow(seq.name, img)
        print(f'Showing motion image from [{seq.name}]. Press any key to'
              ' close the image and continue.')
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img


# Root folder for Sequence files
src_root = './data/mka-beware-1.1/sequences/mka-beware-1.1/'

motion_dump_root = 'data/mka-beware-1.1/motion_images'
chunk_dump_root = 'data/mka-beware-1.1/sequence_chunks'
dataset_name = 'mka-beware-1.1_cookie-2.0'
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
for filename in Path(src_root).rglob('*.json'):
    # print(str(filename).replace('\\', '/').split('/'))
    filename_split = str(filename).replace('\\', '/').split('/')
    seq_name = filename_split[-1]  # The filename (eg: 'myname.json')
    seq_class = filename_split[-2]  # The class (eG: 'squat')
    # Load
    seq = seq_loader.load(path=f'{str(filename)}',
                          name=seq_name[:-5],
                          desc=seq_class)
    # Normalize
    seq.positions = mofex_norm.center_positions(seq.positions)
    seq.positions = normalizations.pose_position(
        seq.positions, seq.positions[:, CENTER_IDX, :])
    mofex_norm.orientation(seq.positions, seq.positions[0, LEFT_IDX, :],
                           seq.positions[0, RIGHT_IDX, :],
                           seq.positions[0, UP_IDX, :])

    # Add Sequences to class label in dictionary
    if seq.desc in seq_labeled_dict:
        seq_labeled_dict[seq.desc].append(seq)
    else:
        seq_labeled_dict[seq.desc] = [seq]
    print(f'Loaded: {seq.name} [{seq.desc}]')

seqs_chunked = []
chunks_labeled_dict = {}
for seq_class in seq_labeled_dict:
    for i, seq in enumerate(seq_labeled_dict[seq_class]):
        # create copy of seq
        _seq = seq[0]

        # append first frame to list as norep
        _seq_append = _seq[:]
        _seq_append.name = f'{seq_class}_{i}_c{len(_seq_append)-1}'
        _seq_append.desc = 'norep'
        if 'norep' in chunks_labeled_dict:
            chunks_labeled_dict['norep'].append(_seq_append)
        else:
            chunks_labeled_dict['norep'] = [_seq_append]

        first = True
        # loop over sequence timesteps * 5 (5x same rep)
        # for idx, j in enumerate([1, -1, 1, -1, 1]):
        for idx, j in enumerate([1]):
            for k in range(len(seq)):
                # skip first frame
                if first:
                    first = False
                    continue

                # append k frame in j (reversed) order
                if j == -1:
                    _k = len(seq) - k
                else:
                    _k = k

                _seq.append(seq[_k])

                # each full passthrough (+- 5) is a rep
                if k >= len(seq) - 6:
                    rep = 'rep'
                elif idx > 0 and k <= 4:
                    rep = 'rep'
                # if k == len(seq) - 1:
                else:
                    rep = 'norep'

                _seq_append = _seq[:]
                _seq_append.name = f'{seq_class}_{i}_c{len(_seq_append)-1}'
                _seq_append.desc = rep

                seqs_chunked.append(_seq_append)

                if rep in chunks_labeled_dict:
                    chunks_labeled_dict[rep].append(_seq_append)
                else:
                    chunks_labeled_dict[rep] = [_seq_append]

                # for k +- 5 (rep) create more reps with start +-5
                if rep == 'rep':
                    for l in range(1, 5):
                        for m in range(1, 4):
                            _seq_append = _seq[l::m]
                            _seq_append.name = f'{seq_class}_{i}_c{len(_seq_append)-1}_f{l}_s{m}'
                            _seq_append.desc = rep

                            seqs_chunked.append(_seq_append)
                            chunks_labeled_dict[rep].append(_seq_append)

# Filter Outliers and get xyz_minmax values
iqr_factor_x = 2.5
iqr_factor_y = 2.5
iqr_factor_z = 2.5

minmax_xyz = xyz_minmax_coords_per_bodypart(
    seqs_chunked, [iqr_factor_x, iqr_factor_y, iqr_factor_z],
    plot_histogram=False)
# xmin, xmax = minmax_xyz[0]
# ymin, ymax = minmax_xyz[1]
# zmin, zmax = minmax_xyz[2]

minmax_filename = f'{motion_dump_root}/{dataset_name}/minmax_values.txt'
# Create basic text file to store(remember) the used min/max values
if not os.path.isdir(f'{motion_dump_root}/{dataset_name}'):
    os.makedirs(f'{motion_dump_root}/{dataset_name}')
minmax_file = open(minmax_filename, 'w')
# minmax_file.write(
#     f'x: [{xmin}, {xmax}]\ny: [{ymin}, {ymax}]\nz: [{zmin}, {zmax}]')
minmax_file.write(f'{minmax_xyz}')
minmax_file.write(f'\n')
minmax_file.write(
    f'x_iqr_factor: {iqr_factor_x}\ny_iqr_factor: {iqr_factor_y}\nz_iqr_factor: {iqr_factor_z}'
)
minmax_file.close()

# --- Create Motion Images

# Split sequences into Train/Val
train_seqs = []
val_seqs = []
# Split class sequences into train/val sets
for label in chunks_labeled_dict:
    label_split = train_test_split(chunks_labeled_dict[label],
                                   test_size=0.1,
                                   random_state=42)
    train_seqs.extend(label_split[0])
    val_seqs.extend(label_split[1])

# Train Set
longest_seq = 0
for idx, seq in enumerate(train_seqs):
    print(f"Saving {idx}/{len(train_seqs)} from train set")
    if len(seq) > longest_seq:
        longest_seq = len(seq)
    # set and create directories
    motion_dir = f'{motion_dump_root}/{dataset_name}/train/{seq.desc}'
    chunk_dir = f'{chunk_dump_root}/{dataset_name}/train/{seq.desc}'
    motion_out = f'{motion_dir}/{seq.name}.png'
    chunk_out = f'{chunk_dir}/{seq.name}.json'
    os.makedirs(os.path.abspath(motion_dir), exist_ok=True)
    os.makedirs(os.path.abspath(chunk_dir), exist_ok=True)
    # Create and save Motion Image with defined output size and X,Y,Z min/max values to map respective color channels to positions.
    # (xmin, xmax) -> RED(0, 255), (ymin, ymax) -> GREEN(0, 255), (zmin, zmax) -> BLUE(0, 255),
    # img = seq.to_motionimg(output_size=(256, 256), minmax_pos_x=(xmin, xmax), minmax_pos_y=(ymin, ymax), minmax_pos_z=(zmin, zmax))
    img = to_motionimg_bp_minmax(seq,
                                 output_size=(256, 256),
                                 minmax_per_bp=minmax_xyz)
    cv2.imwrite(motion_out, img)
    out_json = {
        "name": seq.name,
        "date": "2020-08-19T15:44:52.1809407+02:00",
        "format": _format,
        "timestamps": np.arange(0, len(seq)).tolist(),
        "positions": positions_to_list(seq.positions)
    }
    with open(chunk_out, 'w') as jf:
        json.dump(out_json, jf)
print(f"Longest sequence in train contains {longest_seq} timesteps.")

# Validation Set
longest_seq = 0
for idx, seq in enumerate(val_seqs):
    print(f"Saving {idx}/{len(val_seqs)} from val set")
    if len(seq) > longest_seq:
        longest_seq = len(seq)
    motion_dir = f'{motion_dump_root}/{dataset_name}/val/{seq.desc}'
    chunk_dir = f'{chunk_dump_root}/{dataset_name}/val/{seq.desc}'
    motion_out = f'{motion_dir}/{seq.name}.png'
    chunk_out = f'{chunk_dir}/{seq.name}.json'
    os.makedirs(os.path.abspath(motion_dir), exist_ok=True)
    os.makedirs(os.path.abspath(chunk_dir), exist_ok=True)
    # Create and save Motion Image with defined output size and X,Y,Z min/max values to map respective color channels to positions.
    # (xmin, xmax) -> RED(0, 255), (ymin, ymax) -> GREEN(0, 255), (zmin, zmax) -> BLUE(0, 255),
    # img = seq.to_motionimg(output_size=(256, 256), minmax_pos_x=(xmin, xmax), minmax_pos_y=(ymin, ymax), minmax_pos_z=(zmin, zmax))
    img = to_motionimg_bp_minmax(seq,
                                 output_size=(256, 256),
                                 minmax_per_bp=minmax_xyz)
    cv2.imwrite(motion_out, img)
    out_json = {
        "name": seq.name,
        "date": "2020-08-19T15:44:52.1809407+02:00",
        "format": _format,
        "timestamps": np.arange(0, len(seq)).tolist(),
        "positions": positions_to_list(seq.positions)
    }
    with open(chunk_out, 'w') as jf:
        json.dump(out_json, jf)
print(f"Longest sequence in val contains {longest_seq} timesteps.")