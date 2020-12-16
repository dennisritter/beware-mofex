import cv2
import os

from pathlib import Path

from sklearn.model_selection import train_test_split
from mofex.preprocessing.helpers import xyz_minmax_coords
import mofex.preprocessing.normalizations as mofex_norm
import mana.utils.math.normalizations as normalizations
from mana.utils.data_operations.loaders.sequence_loader_hdm05 import SequenceLoaderHDM05
from mana.models.sequence_transforms import SequenceTransforms

# Root folder for Sequence files
src_root = './data/hdm05-122/sequences/hdm05-122/amc/'
filename_asf = '*.asf'
filename_amc = '*.amc'

dump_root = 'data/hdm05-122/motion_images'
dataset_name = 'hdm05-122_90-10_downstream'


def load_seqs_asf_amc_hdm05(root, regex_str_asf, regex_str_amc):
    seqs = []
    print(
        f'Loading sequences from:\nroot: {root}\nASF pattern: {regex_str_asf}\nAMC pattern: {regex_str_amc}'
    )
    for amc_path in Path(root).rglob(regex_str_amc):
        class_dir = '/'.join(str(amc_path).split("/")[:-1])
        amc_file = str(amc_path).split("/")[-1]
        asf_file = f'{amc_file[0:6]}.asf'
        asf_path = class_dir + '/' + asf_file
        seqs.append({
            'asf': asf_path,
            'amc': str(amc_path),
            'name': amc_file,
            'desc': class_dir.split('/')[-1]
        })
        # seqs.append(Sequence.from_hdm05_asf_amc_files(asf_path=asf_path, amc_path=amc_path, name=amc_file, desc=class_dir.split('/')[-1]))
        # print(f'Loaded: {seqs[-1].name} -> Class = {seqs[-1].desc}')
    return seqs


# Indices constants for body parts that define normalized orientation of the skeleton
# left -> hip_left
LEFT_IDX = 1
# right -> hip_right
RIGHT_IDX = 6
# up -> lowerback
UP_IDX = 11

# --- Loading Sequences and preprocessing
seq_transforms = SequenceTransforms(SequenceTransforms.hdm05_to_iisy())
seq_loader = SequenceLoaderHDM05(seq_transforms)
seq_labeled_dict = {}

seqs = load_seqs_asf_amc_hdm05(src_root, filename_asf, filename_amc)

for seq_dict in seqs:
    seq = seq_loader.load(seq_dict['asf'], seq_dict['amc'], seq_dict['name'],
                          seq_dict['desc'])
    print(f'Processing: {seq.name}')
    # Normalization
    # * Body Part Indices -> 1 = left hip; 6 = right hip
    seq.positions = mofex_norm.center_positions(seq.positions)
    seq.positions = normalizations.pose_position(
        seq.positions,
        (seq.positions[:, LEFT_IDX, :] + seq.positions[:, RIGHT_IDX, :]) * 0.5)
    mofex_norm.orientation(seq.positions, seq.positions[0, LEFT_IDX],
                           seq.positions[0, RIGHT_IDX], seq.positions[0,
                                                                      UP_IDX])

    # Fill dictionary with classes present in dataset to split sequences in each class seperately
    if seq.desc in seq_labeled_dict.keys():
        seq_labeled_dict[seq.desc].append(seq)
    else:
        seq_labeled_dict[seq.desc] = [seq]

# * Split sequences into chunks
chunk_sizes = [4, 8, 16, 32, 64, 128]
merge_sizes = [1, 2, 3, 4, 5, 6]
seqs_chunked = []
chunks_labeled_dict = {}
for seq_class in seq_labeled_dict:
    for i, seq in enumerate(seq_labeled_dict[seq_class]):

        # split sequence into smaller chunks for each size
        for size in chunk_sizes:
            if size > len(seq):
                size = len(seq)
            seq_chunks = seq.split(overlap=0, subseq_size=size)

            for j, seq_chunk in enumerate(seq_chunks):
                seq_chunk.name = f'{seq_class}_{i}_c{size}-{j}'
                seq_chunk.desc = f'{seq_class}'
                seqs_chunked.append(seq_chunk)

                if seq_chunk.desc in chunks_labeled_dict:
                    chunks_labeled_dict[seq_chunk.desc].append(seq_chunk)
                else:
                    chunks_labeled_dict[seq_chunk.desc] = [seq_chunk]

        # repeat sequence for each size
        for size in merge_sizes:
            _seq = seq[:]
            for _ in range(size):
                _seq.append(seq)
            _seq.name = f'{seq_class}_{i}_m{str(size)}'
            _seq.desc = f'{seq_class}'
            seqs_chunked.append(_seq)

            if seq.desc in chunks_labeled_dict:
                chunks_labeled_dict[_seq.desc].append(_seq)
            else:
                chunks_labeled_dict[_seq.desc] = [_seq]

# Filter Outliers and get xyz_minmax values
iqr_factor_x = 2.5
iqr_factor_y = 2.5
iqr_factor_z = 3.5
minmax_xyz = xyz_minmax_coords(seqs_chunked,
                               [iqr_factor_x, iqr_factor_y, iqr_factor_z],
                               plot_histogram=False)
xmin, xmax = minmax_xyz[0]
ymin, ymax = minmax_xyz[1]
zmin, zmax = minmax_xyz[2]

minmax_filename = f'{dump_root}/{dataset_name}/minmax_values.txt'
# Create basic text file to store(remember) the used min/max values
if not os.path.isdir(f'{dump_root}/{dataset_name}'):
    os.makedirs(f'{dump_root}/{dataset_name}')
minmax_file = open(minmax_filename, 'w')
minmax_file.write(
    f'x: [{xmin}, {xmax}]\ny: [{ymin}, {ymax}]\nz: [{zmin}, {zmax}]')
minmax_file.write(f'\n')
minmax_file.write(
    f'x_iqr_factor: {iqr_factor_x}\ny_iqr_factor: {iqr_factor_y}\nz_iqr_factor: {iqr_factor_z}'
)
minmax_file.close()

# --- Create Motion Images

train_seqs = []
val_seqs = []
# Split class sequences into train/val sets
for label in chunks_labeled_dict.keys():
    label_split = train_test_split(chunks_labeled_dict[label],
                                   test_size=0.1,
                                   random_state=42)
    train_seqs.extend(label_split[0])
    val_seqs.extend(label_split[1])

# Train Set
for seq in train_seqs:
    # set and create directories
    class_dir = f'{dump_root}/{dataset_name}/train/{seq.desc}'
    out = f'{class_dir}/{seq.name}.png'
    if not os.path.isdir(os.path.abspath(class_dir)):
        os.makedirs(os.path.abspath(class_dir))
    # Create and save Motion Image with defined output size and X,Y,Z min/max values to map respective color channels to positions.
    # (xmin, xmax) -> RED(0, 255), (ymin, ymax) -> GREEN(0, 255), (zmin, zmax) -> BLUE(0, 255),
    img = seq.to_motionimg(output_size=(256, 256),
                           minmax_pos_x=(xmin, xmax),
                           minmax_pos_y=(ymin, ymax),
                           minmax_pos_z=(zmin, zmax))
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
    img = seq.to_motionimg(output_size=(256, 256),
                           minmax_pos_x=(xmin, xmax),
                           minmax_pos_y=(ymin, ymax),
                           minmax_pos_z=(zmin, zmax))
    cv2.imwrite(out, img)