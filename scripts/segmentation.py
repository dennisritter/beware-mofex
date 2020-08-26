import numpy as np
import time
from torchvision import transforms
import mofex.feature_vectors as featvec
from scipy.signal import argrelextrema, savgol_filter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import mofex.model_loader as model_loader
from mofex.preprocessing.helpers import to_motionimg_bp_minmax
import mofex.preprocessing.normalizations as mofex_norm
import mana.utils.math.normalizations as normalizations
from mana.utils.data_operations.loaders.sequence_loader_mka import SequenceLoaderMKA
from mana.models.sequence_transforms import SequenceTransforms
"""
Segmentation Approach
1. Take 1/2/4/8/16/32 (key_len) frame motion sequence key_gt of ground truth sequence seq_gt
2. Segment long sequence (seq_q) into short sequences (key_q) of length key_len
3. Compare key_gt and key_q.
4. Show distances in line graph   
"""

# Indices constants for body parts that define normalized orientation of the skeleton
# center -> pelvis
CENTER_IDX = 0
# left -> hip_left
LEFT_IDX = 1
# right -> hip_right
RIGHT_IDX = 6
# up -> lowerback
UP_IDX = 11


def _normalize_seq(seq):
    seq.positions = mofex_norm.center_positions(seq.positions)
    seq.positions = normalizations.pose_position(seq.positions, seq.positions[:, CENTER_IDX, :])
    mofex_norm.orientation(seq.positions, seq.positions[0, LEFT_IDX, :], seq.positions[0, RIGHT_IDX, :], seq.positions[0, UP_IDX, :])
    return seq


# Min/Max values for each body part respectively used for the color mapping when transforming sequences to motion images
# min values are mapped to RGB(0,0,0), max values to RGB(255,255,255)
minmax_per_bp = np.array([[[-1.000e+00, 1.000e+00], [-1.000e+00, 1.000e+00], [-1.000e+00, 1.000e+00]],
                          [[-3.100e+01, 3.300e+01], [-4.000e+01, 1.150e+02], [1.340e+02, 2.100e+02]],
                          [[-4.500e+01, 4.300e+01], [-8.500e+01, 2.300e+02], [2.150e+02, 3.770e+02]],
                          [[-7.500e+01, 6.800e+01], [-2.130e+02, 3.690e+02], [3.760e+02, 6.190e+02]],
                          [[-9.800e+01, 2.700e+01], [-1.920e+02, 3.420e+02], [3.540e+02, 5.790e+02]],
                          [[-2.270e+02, -1.060e+02], [-2.580e+02, 3.790e+02], [3.450e+02, 6.520e+02]],
                          [[-5.290e+02, -1.070e+02], [-3.630e+02, 6.300e+02], [1.380e+02, 9.610e+02]],
                          [[-7.680e+02, 0.000e+00], [-3.880e+02, 8.580e+02], [-8.000e+01, 1.219e+03]],
                          [[-8.500e+02, 2.600e+01], [-4.730e+02, 9.500e+02], [-1.550e+02, 1.303e+03]],
                          [[-9.650e+02, 8.400e+01], [-5.110e+02, 1.049e+03], [-2.700e+02, 1.410e+03]],
                          [[-8.720e+02, 7.800e+01], [-4.170e+02, 9.790e+02], [-2.030e+02, 1.342e+03]],
                          [[-3.500e+01, 9.200e+01], [-1.870e+02, 3.470e+02], [3.470e+02, 5.780e+02]],
                          [[4.700e+01, 2.250e+02], [-2.310e+02, 4.140e+02], [3.310e+02, 6.100e+02]],
                          [[-1.530e+02, 5.400e+02], [-3.220e+02, 6.720e+02], [1.190e+02, 9.120e+02]],
                          [[-3.100e+02, 7.300e+02], [-4.060e+02, 8.740e+02], [-1.050e+02, 1.177e+03]],
                          [[-2.330e+02, 8.180e+02], [-4.950e+02, 9.770e+02], [-1.940e+02, 1.269e+03]],
                          [[-2.110e+02, 9.150e+02], [-5.190e+02, 1.042e+03], [-3.080e+02, 1.375e+03]],
                          [[-2.240e+02, 8.380e+02], [-3.960e+02, 9.870e+02], [-2.080e+02, 1.294e+03]],
                          [[-1.070e+02, -8.000e+01], [-4.800e+01, 4.100e+01], [-1.400e+01, 1.200e+01]],
                          [[-3.480e+02, 1.850e+02], [-1.080e+02, 4.050e+02], [-4.710e+02, 5.000e+01]],
                          [[-3.000e+02, 1.440e+02], [-2.310e+02, 7.080e+02], [-9.190e+02, -2.760e+02]],
                          [[-3.530e+02, 1.560e+02], [-1.060e+02, 8.500e+02], [-1.052e+03, -3.130e+02]],
                          [[7.200e+01, 9.700e+01], [-3.700e+01, 4.300e+01], [-1.100e+01, 1.200e+01]],
                          [[-7.400e+01, 2.670e+02], [-1.020e+02, 4.220e+02], [-4.640e+02, 5.800e+01]],
                          [[-1.590e+02, 5.190e+02], [-3.710e+02, 7.270e+02], [-9.090e+02, -2.670e+02]],
                          [[-2.240e+02, 6.050e+02], [-3.020e+02, 8.620e+02], [-1.042e+03, -1.800e+02]],
                          [[-9.000e+01, 8.600e+01], [-2.550e+02, 4.310e+02], [4.270e+02, 7.140e+02]],
                          [[-1.790e+02, 1.790e+02], [-1.250e+02, 5.860e+02], [3.490e+02, 8.450e+02]],
                          [[-1.670e+02, 1.370e+02], [-1.730e+02, 5.670e+02], [3.970e+02, 8.550e+02]],
                          [[-1.690e+02, 1.100e+01], [-2.760e+02, 4.730e+02], [4.760e+02, 8.030e+02]],
                          [[-1.460e+02, 1.740e+02], [-1.680e+02, 5.590e+02], [3.900e+02, 8.530e+02]],
                          [[-1.300e+01, 1.600e+02], [-2.750e+02, 4.550e+02], [4.720e+02, 8.050e+02]]])

seq_transforms = SequenceTransforms(SequenceTransforms.mka_to_iisy())
seq_loader = SequenceLoaderMKA(seq_transforms)
seq_class = 'squat'
seq_name = 'squat_255.json'
seq_gt = seq_loader.load(path=f'./data/sequences/mka-beware-1.1/{seq_class}/{seq_name}', name=seq_name[:-5], desc=seq_class)

### SQUAT Christopher
# src_root = './data/sequences/mka-beware-1.1-sets/Christopher/squat'
src_root = './data/sequences/mka-beware-1.1-sets/Christopher/squat/19-08-2020-12-10-48'
# src_root = './data/sequences/mka-beware-1.1-sets/Christopher/squat/19-08-2020-12-09-43'
# src_root = './data/sequences/mka-beware-1.1-sets/Christopher/squat/19-08-2020-12-08-41'
# src_root = './data/sequences/mka-beware-1.1-sets/Christopher/squat/19-08-2020-12-07-41'
# src_root = './data/sequences/mka-beware-1.1-sets/Christopher/squat/19-08-2020-12-06-00'
# src_root = './data/sequences/mka-beware-1.1-sets/Christopher/squat/19-08-2020-12-03-55'
# src_root = './data/sequences/mka-beware-1.1-sets/Christopher/squat/19-08-2020-12-02-40'
# src_root = './data/sequences/mka-beware-1.1-sets/Christopher/squat/19-08-2020-12-01-33'
# src_root = './data/sequences/mka-beware-1.1-sets/Christopher/squat/19-08-2020-11-59-52'
# src_root = './data/sequences/mka-beware-1.1-sets/Christopher/squat/19-08-2020-11-58-05'

seqs = []
for filename in Path(src_root).rglob('*.json'):
    # print(str(filename).replace('\\', '/').split('/'))
    filename_split = str(filename).replace('\\', '/').split('/')
    seq_name = filename_split[-1]  # The filename (eg: 'squat_1.json')
    seq_class = filename_split[-2]  # The class (eG: 'squat')
    # Load
    seq = seq_loader.load(path=f'{str(filename)}', name=seq_name[:-5], desc=seq_class)
    print(f'{seq.name}: frames = {len(seq)} ')
    seqs.append(seq)
# Append Sequences to one set (one sequence)
for seq in seqs[1:1]:
    seqs[0].append(seq)
seq_q = seqs[0]
print(f'Created query sequence of {len(seq_q)} frames.')

# subseq_len_list = [2, 4, 8, 16, 32]
subseq_len_list = [4]
savgol_windows = [11, 21, 31, 41, 51]
savgol_order = 7

fig_dist = make_subplots(rows=len(subseq_len_list), cols=1)
fig_savgol = make_subplots(rows=len(savgol_windows), cols=1)
for row, subseq_len in enumerate(subseq_len_list):
    start = time.time()

    ## Get first frames of Ground Truth to search for similar segments in a long sequence
    # Normalize and get Motion Images
    key_gt = seq_gt[0:subseq_len]
    key_gt = _normalize_seq(key_gt)
    mi_gt = to_motionimg_bp_minmax(key_gt, output_size=(256, 256), minmax_per_bp=minmax_per_bp)

    ## Split long sequence into small segments
    # Normalize and get Motion Images
    mi_q_list = []
    seq_q_split = seq_q.split(overlap=0, subseq_size=subseq_len)
    for seq_q_part in seq_q_split:
        seq_q_part = _normalize_seq(seq_q_part)
        mi_q_list.append(to_motionimg_bp_minmax(seq_q_part, output_size=(256, 256), minmax_per_bp=minmax_per_bp))

    dataset_name = 'mka-beware-1.1'
    # CNN Model name -> model_dataset-numclasses_train-val-ratio
    model_name = 'resnet101_mka-beware-1.1'
    # The CNN Model for Feature Vector generation
    model = model_loader.load_trained_model(model_name=model_name,
                                            remove_last_layer=True,
                                            state_dict_path=f'./data/trained_models/{dataset_name}/{model_name}_e5.pt')
    # The models output size (Feature Vector length
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
        savgol_distances = savgol_filter(distances, savgol_win, savgol_order, mode='nearest')
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