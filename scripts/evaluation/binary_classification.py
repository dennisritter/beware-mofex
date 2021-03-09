"""Contains code to evaluate a model with the cookie binary classification
problem. I.e. given a sequence when is a full rep."""

import torch
import glob
import numpy as np
import json
import cv2

from mana.utils.math.normalizations import pose_mean, pose_orientation, pose_position
from mana.models.sequence_transforms import SequenceTransforms
from mana.utils.data_operations.loaders.sequence_loader_mka import SequenceLoaderMKA

from mofex.mka_loader import MKAToIISYNorm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
from pathlib import Path
import mofex.model_loader as model_loader
import mofex.model_saver as model_saver
import mofex.model_plotter as model_plotter
import mofex.model_trainer as model_trainer
import mana.utils.math.normalizations as normalizations
from mana.utils.data_operations.loaders.sequence_loader_mka import SequenceLoaderMKA
from mana.models.sequence_transforms import SequenceTransforms
import mofex.preprocessing.normalizations as mofex_norm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def simple_evaluate(model, path):

    # Data augmentation and normalization for training and validation repectively
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create training and validation datasets
    image_dataset = datasets.ImageFolder(path, data_transform)

    # Create training and validation dataloaders
    data_loader = torch.utils.data.DataLoader(image_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=12)

    results = []
    model.eval()
    with torch.no_grad():

        last_pred = -1
        for idx, item in enumerate(data_loader):

            inputs, _ = item[0].to(device), item[1].to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            pred = preds.item()
            if pred == 0:
                results.append('rep')
                if not last_pred == pred:
                    print(f'idx: {idx} -> rep')
                last_pred = pred
                print(f'idx: {idx} -> rep')
            else:
                results.append('norep')
                if not last_pred == pred:
                    print(f'idx: {idx} -> norep')
                last_pred = pred
                print(f'idx: {idx} -> norep')

    with open('preds.log', 'w') as _file:
        _file.writelines(results)


def evaluate(model, path):
    seq_name = path.split('/')[-1].split('.')[0]
    # 1. create/ load long sequence (10-20 reps)
    # 2. create motion images on the fly (every next frame)
    #       - don't forget normalization

    # Indices constants for body parts that define normalized orientation of the skeleton
    # center -> pelvis
    CENTER_IDX = 0
    # left -> hip_left
    LEFT_IDX = 18
    # right -> hip_right
    RIGHT_IDX = 22
    # up -> spinenavel
    UP_IDX = 1

    # Normalize
    seq_transforms = SequenceTransforms(SequenceTransforms.mka_to_iisy())
    seq_loader = SequenceLoaderMKA(seq_transforms)
    seq = seq_loader.load(path=path, name=seq_name)
    seq.positions = mofex_norm.center_positions(seq.positions)
    seq.positions = normalizations.pose_position(
        seq.positions, seq.positions[:, CENTER_IDX, :])
    mofex_norm.orientation(seq.positions, seq.positions[0, LEFT_IDX, :],
                           seq.positions[0, RIGHT_IDX, :],
                           seq.positions[0, UP_IDX, :])

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    def to_motionimg(seq) -> np.ndarray:
        """ Returns a Motion Image, that represents this sequences' positions."""

        minmax_per_bp = np.array([[[-1.000e+00, 1.000e+00],
                                   [-1.000e+00, 1.000e+00],
                                   [-1.000e+00, 1.000e+00]],
                                  [[-3.100e+01, 3.300e+01],
                                   [-4.200e+01, 1.150e+02],
                                   [1.340e+02, 2.100e+02]],
                                  [[-4.500e+01, 4.300e+01],
                                   [-8.500e+01, 2.300e+02],
                                   [2.150e+02, 3.770e+02]],
                                  [[-7.500e+01, 6.800e+01],
                                   [-2.130e+02, 3.690e+02],
                                   [3.760e+02, 6.190e+02]],
                                  [[-9.800e+01, 2.700e+01],
                                   [-1.920e+02, 3.420e+02],
                                   [3.540e+02, 5.790e+02]],
                                  [[-2.270e+02, -1.060e+02],
                                   [-2.580e+02, 3.790e+02],
                                   [3.450e+02, 6.520e+02]],
                                  [[-5.290e+02, -1.070e+02],
                                   [-3.630e+02, 6.300e+02],
                                   [1.380e+02, 9.610e+02]],
                                  [[-7.680e+02, 0.000e+00],
                                   [-3.880e+02, 8.580e+02],
                                   [-8.000e+01, 1.219e+03]],
                                  [[-8.500e+02, 2.600e+01],
                                   [-4.730e+02, 9.500e+02],
                                   [-1.550e+02, 1.303e+03]],
                                  [[-9.650e+02, 8.400e+01],
                                   [-5.110e+02, 1.049e+03],
                                   [-2.700e+02, 1.410e+03]],
                                  [[-8.720e+02, 7.800e+01],
                                   [-4.170e+02, 9.790e+02],
                                   [-2.030e+02, 1.342e+03]],
                                  [[-3.500e+01, 9.200e+01],
                                   [-1.870e+02, 3.470e+02],
                                   [3.470e+02, 5.780e+02]],
                                  [[4.700e+01, 2.240e+02],
                                   [-2.310e+02, 4.140e+02],
                                   [3.310e+02, 6.100e+02]],
                                  [[-1.530e+02, 5.400e+02],
                                   [-3.220e+02, 6.720e+02],
                                   [1.190e+02, 9.120e+02]],
                                  [[-3.100e+02, 7.300e+02],
                                   [-4.060e+02, 8.740e+02],
                                   [-1.050e+02, 1.177e+03]],
                                  [[-2.330e+02, 8.180e+02],
                                   [-4.950e+02, 9.770e+02],
                                   [-1.940e+02, 1.269e+03]],
                                  [[-2.110e+02, 9.150e+02],
                                   [-5.190e+02, 1.042e+03],
                                   [-3.080e+02, 1.375e+03]],
                                  [[-2.240e+02, 8.380e+02],
                                   [-3.960e+02, 9.870e+02],
                                   [-2.080e+02, 1.294e+03]],
                                  [[-1.070e+02, -8.000e+01],
                                   [-4.800e+01, 4.100e+01],
                                   [-1.400e+01, 1.200e+01]],
                                  [[-3.460e+02, 1.850e+02],
                                   [-9.300e+01, 4.040e+02],
                                   [-4.710e+02, 5.000e+01]],
                                  [[-3.000e+02, 1.440e+02],
                                   [-2.230e+02, 7.080e+02],
                                   [-9.190e+02, -2.760e+02]],
                                  [[-3.530e+02, 1.560e+02],
                                   [-1.060e+02, 8.500e+02],
                                   [-1.052e+03, -3.130e+02]],
                                  [[7.200e+01, 9.700e+01],
                                   [-3.700e+01, 4.300e+01],
                                   [-1.100e+01, 1.200e+01]],
                                  [[-7.400e+01, 2.670e+02],
                                   [-1.020e+02, 4.220e+02],
                                   [-4.640e+02, 5.800e+01]],
                                  [[-1.590e+02, 5.190e+02],
                                   [-3.710e+02, 7.270e+02],
                                   [-9.090e+02, -2.670e+02]],
                                  [[-2.240e+02, 6.050e+02],
                                   [-3.020e+02, 8.620e+02],
                                   [-1.042e+03, -1.800e+02]],
                                  [[-9.000e+01, 8.600e+01],
                                   [-2.550e+02, 4.310e+02],
                                   [4.270e+02, 7.140e+02]],
                                  [[-1.790e+02, 1.790e+02],
                                   [-1.250e+02, 5.860e+02],
                                   [3.490e+02, 8.450e+02]],
                                  [[-1.670e+02, 1.370e+02],
                                   [-1.730e+02, 5.670e+02],
                                   [3.970e+02, 8.500e+02]],
                                  [[-1.690e+02, 1.100e+01],
                                   [-2.760e+02, 4.730e+02],
                                   [4.760e+02, 8.030e+02]],
                                  [[-1.460e+02, 1.740e+02],
                                   [-1.680e+02, 5.590e+02],
                                   [3.900e+02, 8.510e+02]],
                                  [[-1.300e+01, 1.600e+02],
                                   [-2.750e+02, 4.550e+02],
                                   [4.720e+02, 8.050e+02]]])
        # Create Image container
        img = np.zeros((len(seq.positions[0, :]), len(seq.positions), 3),
                       dtype='uint8')
        # 1. Map (min_pos, max_pos) range to (0, 255) Color range.
        # 2. Swap Axes of and frames(0) body parts(1) so rows represent body
        # parts and cols represent frames.
        for i, bp in enumerate(seq.positions[0]):
            bp_positions = seq.positions[:, i]
            x_colors = np.interp(
                bp_positions[:, 0],
                [minmax_per_bp[i, 0, 0], minmax_per_bp[i, 0, 1]], [0, 255])
            img[i, :, 0] = x_colors
            img[i, :,
                1] = np.interp(bp_positions[:, 1],
                               [minmax_per_bp[i, 1, 0], minmax_per_bp[i, 1, 1]],
                               [0, 255])
            img[i, :,
                2] = np.interp(bp_positions[:, 2],
                               [minmax_per_bp[i, 2, 0], minmax_per_bp[i, 2, 1]],
                               [0, 255])
        return cv2.resize(img, (256, 256))

    # 3. let model check if input is rep
    start_idx = 0
    results = []
    model.eval()
    with torch.no_grad():

        last_pred = -1
        for idx in range(len(seq)):
            _motion_image = to_motionimg(seq[start_idx:idx + 1])
            motion_image = to_motionimg(seq[start_idx:idx + 1])
            motion_image = data_transform(motion_image).unsqueeze(0).to(device)

            outputs = model(motion_image)
            _, preds = torch.max(outputs, 1)

            pred = preds.item()
            if pred == 0:
                results.append('norep')
                # if not last_pred == pred:
                #     print(f'idx: {idx} -> norep')
                last_pred = pred

            # 4. if rep -> reset motion image (sequence)
            #       - maybe use a threshold (if 4 times rep)
            #
            else:
                results.append('rep')
                if not last_pred == pred:
                    print(f'idx: {idx} -> rep')

                    cv2.imwrite(
                        f'output/motion_images/{seq_name}_{start_idx}-{idx}.png',
                        _motion_image)

                last_pred = pred
                # print(f'idx: {idx} -> rep')

                #! setting start idx does not work
                #! dataset wasn't generated this way -> only extended from start
                start_idx = idx

    # 5. redo for combined long seq (squat into shoulder press?)
    #       - use mka tool for creating new rep


### Motion Images
# model_list = [
#     # 'output/finetuned/mka-beware-1.1_cookie/resnet101_hdm05-122_90-10_cookie_sgd_e50_mka-beware-1.1_cookie_sgd_e50.pt'
#     # 'output/finetuned/mka-beware-1.1_cookie_dropout-0.2/resnet101_hdm05-122_90-10_cookie_sgd_e50_mka-beware-1.1_cookie_dropout-0.2_sgd_e50.pt',
#     # 'output/finetuned/mka-beware-1.1_cookie-2.0/resnet101_hdm05-122_90-10_cookie_sgd_e50_mka-beware-1.1_cookie-2.0_sgd_e50.pt',
#     'output/finetuned/mka-beware-1.1_cookie-2.0/resnet101_hdm05-122_90-10_cookie_sgd_e50_mka-beware-1.1_cookie-2.0_sgd_e1.pt',
# ]

# for model_path in model_list:

#     # Initialize the model
#     model = model_loader.load_trained_model(
#         model_name="resnet101_hdm05-122_90-10_cookie_downstream",
#         state_dict_path=model_path)
#     model.to(device)

#     # ##### Simple eval ######
#     # dataset_path = 'data/test/simple'
#     # print(f'------------------------')
#     # print(f'Start evaluating: {model_path.split("/")[-1]}')
#     # print(f'  using: {dataset_path}')
#     # print(f'------------------------')
#     # simple_evaluate(model, dataset_path)
#     # print(f'\n')

#     ##### TODO: Real-World eval ######
#     dataset_pathes = [
#         # 'data/test/merged/overheadpress_combined.json',
#         # 'data/test/merged/squat_combined.json',
#         # 'data/test/merged/tiptoestand_combined.json',
#         # 'data/test/merged/diagonalpull_mixed_combined.json',
#         # 'data/test/merged/diagonalpull_combined.json',
#         'data/test/complex/unnamed_trackingsession_0.json',
#         'data/test/complex/unnamed_trackingsession_1.json',
#         'data/test/complex/unnamed_trackingsession_2.json',
#     ]

#     for path in dataset_pathes:
#         print(f'------------------------')
#         print(f'Start evaluating: {model_path.split("/")[-1]}')
#         print(f'  using: {path}')
#         print(f'------------------------')
#         evaluate(model, path)
#         print(f'\n')

### Sequences


def save_json(filename, sequence):
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

    out_json = {
        "name": sequence.name,
        "date": "2020-08-19T15:44:52.1809407+02:00",
        "format": _format,
        "timestamps": np.arange(0, len(sequence)).tolist(),
        "positions": positions_to_list(sequence.positions)
    }
    with open(filename, 'w') as jf:
        json.dump(out_json, jf)


def norm_sequence(sequence):
    # reshape to (#frames, flatten_bodypart_3d)
    sequence.positions = np.reshape(sequence.positions,
                                    (len(sequence.positions), -1))

    # interpolate to 30 frames with 96 values
    # create 30 steps between 0 and max len
    _steps = np.linspace(0, len(sequence), num=30)
    # for each value of the coordinates (#96 <- 32joints*3d) interpolate the steps (didn't find 3d interp function..)
    # create an array of the list comp. and transpose it to retrieve original (30,96) shape
    sequence.positions = np.array([
        np.interp(_steps, np.arange(len(sequence.positions)),
                  sequence.positions[:, idx])
        for idx in range(sequence.positions.shape[1])
    ]).T

    # repeat array 3 times to re-create 3 channels
    sequence.positions = np.repeat(np.expand_dims(sequence.positions, 0),
                                   3,
                                   axis=0)
    return sequence


def evaluate_sequence(model, path):

    sequence_transforms = SequenceTransforms(
        SequenceTransforms.mka_to_iisy(body_parts=False))
    sequence_transforms.transforms.append(MKAToIISYNorm())
    # sequence_transforms = SequenceTransforms([MKAToIISYNorm()])
    sequence_loader = SequenceLoaderMKA(sequence_transforms)

    sequence = sequence_loader.load(path=path)

    start_idx = 0
    results = []
    model.eval()
    with torch.no_grad():

        last_pred = -1
        for idx in range(len(sequence)):
            _sequence = sequence[start_idx:idx + 1]
            _sequence = norm_sequence(_sequence)

            positions = torch.tensor(
                _sequence.positions).unsqueeze(0).float().to(device)

            outputs = model(positions)

            _, preds = torch.max(outputs, 1)

            pred = preds.item()
            if pred == 1:
                results.append('norep')
                # if not last_pred == pred:
                #     print(f'idx: {idx} -> rep')
                last_pred = pred
            else:
                results.append('rep')
                if not last_pred == pred:
                    if not (idx + 1 - start_idx) < 20:
                        print(
                            f'idx: {idx} -> rep  -  len: {idx + 1 - start_idx}')
                        start_idx = idx
                last_pred = pred

    with open('preds.log', 'w') as _file:
        _file.writelines(results)


model_list = [
    'output/pretrained/mka-beware-1.1_cookie-3.0/resnet101_mka-beware-1.1_cookie-3.0_sgd_e5.pt',
]

for model_path in model_list:
    model = model_loader.load_trained_model(
        model_name="resnet101_hdm05-122_90-10_cookie_downstream",
        state_dict_path=model_path)
    model = model.to(device)

    # for path in sorted(glob.glob('data/test/merged/*.json')):
    # for path in sorted(glob.glob('data/test/complex/*.json')):
    for path in sorted(
            glob.glob('data/test/complex/05-03-2021-12-17-26/*.json')):
        print(f'------------------------')
        print(f'Start evaluating: {model_path.split("/")[-1]}')
        print(f'  using: {path}')
        print(f'------------------------')
        evaluate_sequence(model, path)
        print(f'\n')
