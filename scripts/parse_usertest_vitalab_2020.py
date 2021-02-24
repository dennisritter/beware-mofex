import numpy as np
from pathlib import Path
from mofex.preprocessing.skeleton_visualizer import SkeletonVisualizer
import mofex.preprocessing.normalizations as mofex_norm
import mana.utils.math.normalizations as normalizations
from mana.utils.data_operations.loaders.sequence_loader_mka import SequenceLoaderMKA
from mana.models.sequence_transforms import SequenceTransforms
from mofex.solver.beware_rep_counter import RepCounter

src_root = './data/sequences/mka-vitalab-2020/processed/'

# dump_root = 'data/motion_images'
dataset_name = 'mka-vitalab-2020'
# Indices constants for body parts that define normalized orientation of the skeleton
# center -> pelvis
CENTER_IDX = 0
# left -> hip_left
LEFT_IDX = 18
# right -> hip_right
RIGHT_IDX = 22
# up -> spinenavel
UP_IDX = 1

# def _normalize_seq(seq):
#     seq.positions = mofex_norm.center_positions(seq.positions)
#     seq.positions = normalizations.pose_position(seq.positions, seq.positions[:, CENTER_IDX, :])
#     mofex_norm.orientation(seq.positions, seq.positions[0, LEFT_IDX, :], seq.positions[0, RIGHT_IDX, :], seq.positions[0, UP_IDX, :])
#     return seq

# seq_transforms = SequenceTransforms(SequenceTransforms.mka_to_iisy())
seq_loader = SequenceLoaderMKA()

# Load sequences
seqs = []
# Example name: squat_u1_n10.json
# <exercise>_<userID>_<repetitions>.json
for filename in Path(src_root).rglob('*.json'):
    filename = str(filename).replace('\\', '/')
    print(f'Load File: filename')
    filename_split = filename.replace('\\', '/').split('/')
    seq_name = filename_split[-1][:-5]  # The filename (eg: 'myname.json')
    # seq_class = filename_split[-2]  # The class (eG: 'squat')
    # Load
    seq = seq_loader.load(path=filename, name=seq_name[:-5])
    print(f'{seq.name}: frames = {len(seq)} ')
    seqs.append(seq)
#Append Sequences to one set (one sequence)
# for seq in seqs[1:]:
#     seqs[0].append(seq)
# seq_q = seqs[0]

# sv = SkeletonVisualizer(_normalize_seq(seq_q))
# sv.show(auto_open=True)

# sv = SkeletonVisualizer(_normalize_seq(seq_q[12:72]))
# sv = SkeletonVisualizer(_normalize_seq(seq_q[72:128]))
# sv = SkeletonVisualizer(_normalize_seq(seq_q[128:184]))
# sv = SkeletonVisualizer(_normalize_seq(seq_q[184:240]))
# sv = SkeletonVisualizer(_normalize_seq(seq_q[240:296]))

# sv = SkeletonVisualizer(_normalize_seq(seq_q[296:352]))
# sv = SkeletonVisualizer(_normalize_seq(seq_q[352:408]))
# sv = SkeletonVisualizer(_normalize_seq(seq_q[408:460]))
# sv = SkeletonVisualizer(_normalize_seq(seq_q[460:516]))
# sv = SkeletonVisualizer(_normalize_seq(seq_q[516:572]))

# sv = SkeletonVisualizer(_normalize_seq(seq_q[572:632]))
# sv = SkeletonVisualizer(_normalize_seq(seq_q[632:688]))
# sv = SkeletonVisualizer(_normalize_seq(seq_q[688:740]))
# sv = SkeletonVisualizer(_normalize_seq(seq_q[740:796]))
# sv = SkeletonVisualizer(_normalize_seq(seq_q[796:856]))

# sv = SkeletonVisualizer(_normalize_seq(seq_q[856:916]))
# sv = SkeletonVisualizer(_normalize_seq(seq_q[916:976]))
# sv = SkeletonVisualizer(_normalize_seq(seq_q[976:1036]))
# sv = SkeletonVisualizer(_normalize_seq(seq_q[1036:1092]))
# sv = SkeletonVisualizer(_normalize_seq(seq_q[1092:]))
# sv.show(auto_open=True)

# for filename in Path(src_root).rglob('squat_255.json'):
#     # print(str(filename).replace('\\', '/').split('/'))
#     filename_split = str(filename).replace('\\', '/').split('/')
#     seq_name = filename_split[-1]  # The filename (eg: 'myname.json')
#     seq_class = filename_split[-2]  # The class (eG: 'squat')
#     seq = seq_loader.load(path=f'{str(filename)}', name=seq_name[:-5], desc=seq_class)
#     # Normalize
#     seq.positions = mofex_norm.center_positions(seq.positions)
#     seq.positions = normalizations.pose_position(seq.positions, seq.positions[:, CENTER_IDX, :])
#     mofex_norm.orientation(seq.positions, seq.positions[0, LEFT_IDX, :], seq.positions[0, RIGHT_IDX, :], seq.positions[0, UP_IDX, :])

#     sv = SkeletonVisualizer(seq)
#     sv.show(auto_open=True)