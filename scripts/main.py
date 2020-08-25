import numpy as np
from pathlib import Path
from mofex.preprocessing.skeleton_visualizer import SkeletonVisualizer
import mofex.preprocessing.normalizations as mofex_norm
import mana.utils.math.normalizations as normalizations
from mana.models.sequence import Sequence
from mana.utils.data_operations.loaders.sequence_loader_mka import SequenceLoaderMKA
from mana.models.sequence_transforms import SequenceTransforms

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

for filename in Path(src_root).rglob('overheadpress_1.json'):
    # print(str(filename).replace('\\', '/').split('/'))
    filename_split = str(filename).replace('\\', '/').split('/')
    seq_name = filename_split[-1]  # The filename (eg: 'myname.json')
    seq_class = filename_split[-2]  # The class (eG: 'squat')
    seq = seq_loader.load(path=f'{str(filename)}', name=seq_name[:-5], desc=seq_class)
    # Normalize
    seq.positions = mofex_norm.center_positions(seq.positions)
    seq.positions = normalizations.pose_position(seq.positions, seq.positions[:, CENTER_IDX, :])
    mofex_norm.orientation(seq.positions, seq.positions[0, LEFT_IDX, :], seq.positions[0, RIGHT_IDX, :], seq.positions[0, UP_IDX, :])

    sv = SkeletonVisualizer(seq)
    sv.show(auto_open=True)