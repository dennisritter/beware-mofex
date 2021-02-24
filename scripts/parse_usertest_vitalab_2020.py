import numpy as np
from pathlib import Path
import json
from mana.utils.data_operations.loaders.sequence_loader_mka import SequenceLoaderMKA


def positions_to_list(positions: np.ndarray):
    positions = np.reshape(positions, (len(positions), -1))
    positions = [frame.tolist() for frame in positions]
    return positions


src_root = './data/sequences/mka-vitalab-2020/recordings/10_frame_batches'

dump_root = './data/sequences/mka-vitalab-2020/recordings/10_frame_batches/merged'
dataset_name = 'mka-vitalab-2020'
# Indices constants for body parts that define normalized orientation of the skeleton
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
last_date_dir = None
count = 0
for filename in Path(src_root).rglob('*.json'):
    filename = str(filename).replace('\\', '/')
    print(f'Load File: {filename}')
    filename_split = filename.replace('\\', '/').split('/')
    seq_name = filename_split[-1][:-5]  # The filename (eg: 'myname.json')
    date_dir = filename_split[-2]  # The class (eG: 'squat')
    # Load
    try:
        seq = seq_loader.load(path=filename, name=seq_name[:-5], desc=date_dir)
    except:
        continue
    print(f'{seq.name}: frames = {len(seq)} ')
    if date_dir != last_date_dir and last_date_dir != None or len(seqs) >= 50:
        for i, s in enumerate(seqs[1:]):
            if len(s.positions) > 0:
                seqs[0].append(s)
        if len(seqs) > 0:
            seq_out = seqs[0]
            out_json = {
                "name": seq_out.name,
                "format": _format,
                "timestamps": np.arange(0, len(seq_out)).tolist(),  # aufsteigendes array (0 - X)
                "positions": positions_to_list(seq_out.positions)
            }
            with open(f'{dump_root}/session_{date_dir}_{count}.json', 'w') as jf:
                json.dump(out_json, jf)
                print(f'Saved merged sequence: {str(jf)}')
            seqs = []
            count += 1
    else:
        if len(seq.positions) > 0:
            seqs.append(seq)
    last_date_dir = date_dir

#Append Sequences to one set (one sequence)
