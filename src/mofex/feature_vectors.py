""" Calculates/loads feature vectors from 3-D Motion Sequences.
"""
import numpy as np
import cv2
import json
import os
import random
from datetime import datetime
from pathlib import Path
from mofex.preprocessing.sequence import Sequence
from mofex.preprocessing.skeleton_visualizer import SkeletonVisualizer
import mofex.models.resnet as resnet
import torch
from torchvision import transforms


def load_from_sequences(sequences: list, cnn_model, cnn_preprocess) -> list:
    start = datetime.now()
    feature_vectors = []

    for seq in sequences:
        motion_img = _motion_img_from_sequence(seq)
        # ? What exactly happens here?
        input_tensor = cnn_preprocess(motion_img)
        # ? What exactly happens here?
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')

        output = cnn_model(input_batch)
        # ? What exactly happens here?
        output = output.cpu().detach().numpy().reshape((512))
        feature_vectors.append((seq.name, output))
    print(f"Loaded Feature Vectors from [{len(sequences)}] Sequences [{datetime.now() - start}]")
    return feature_vectors


def load_from_sequences_dir(path: str, tracking_type: str, cnn_model, cnn_preprocess) -> list:
    start = datetime.now()
    ### Load Sequences
    sequences = []
    for filename in Path(path).rglob('*.json'):
        print(f"Loading [{tracking_type}] Sequence file [{filename}]")
        # Use filename without folders as reference
        name = str(filename).split("\\")[-1]
        if tracking_type.lower() == 'mir':
            sequences.append(Sequence.from_mir_file(filename, name=name))
        elif tracking_type.lower() == 'mka':
            sequences.append(Sequence.from_mka_file(filename, name=name))
        else:
            print(f"Tracking Type: [{tracking_type}] is not supported.")
            return
    return load_from_sequences(sequences, cnn_model, cnn_preprocess)


def load_from_file(path: str) -> list:
    print(f"Loading Feature maps from file [{path}]")
    with open(Path(path), 'r') as featvec_file:
        featvec_json_str = featvec_file.read()
        featvec_dict = json.loads(featvec_json_str)
    # Converting into list of tuple
    feature_vectors = [(k, v) for k, v in featvec_dict.items()]
    return feature_vectors


# def load_from_3d_positions(positions: 'np.ndarray') -> list:
#     return feature_vectors


# TODO: Calculate 'smart' minmax_pos values
def motion_image_from_3d_positions(positions: 'np.ndarray',
                                   output_size: (int, int) = (256, 256),
                                   minmax_pos_x: (int, int) = (-1000, 1000),
                                   minmax_pos_y: (int, int) = (-1000, 1000),
                                   minmax_pos_z: (int, int) = (-1000, 1000),
                                   name: str = 'Motion Image',
                                   show_img: bool = False) -> 'np.ndarray':
    # Create Image container
    img = np.zeros((len(positions[0, :]), len(positions), 3), dtype='uint8')
    # 1. Map (min_pos, max_pos) range to (0, 255) Color range.
    # 2. Swap Axes of and frames(0) body parts(1) so rows represent body parts and cols represent frames.
    img[:, :, 0] = np.interp(positions[:, :, 0], [minmax_pos_x[0], minmax_pos_x[1]], [0, 255]).swapaxes(0, 1)
    img[:, :, 1] = np.interp(positions[:, :, 1], [minmax_pos_y[0], minmax_pos_y[1]], [0, 255]).swapaxes(0, 1)
    img[:, :, 2] = np.interp(positions[:, :, 2], [minmax_pos_z[0], minmax_pos_z[1]], [0, 255]).swapaxes(0, 1)
    img = cv2.resize(img, output_size)
    if show_img:
        cv2.imshow(name, img)
        print(f"Showing motion image from [{name}]. Press any key to close the image and continue.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img


def _motion_img_from_sequence(
    seq: 'Sequence',
    output_size: tuple = (256, 256),
    show_img: bool = False,
    show_skeleton: bool = False,
):
    """ Returns a Motion Image, that represents this sequences' positions.

        Creates an Image from 3-D position data of motion sequences.
        Rows represent a body part (or some arbitrary position instance).
        Columns represent a frame of the sequence.

        Args:
            output_size (int, int): The size of the output image in pixels (height, width). Default=(200,200)
            minmax_pos_x (int, int): The minimum and maximum x-position values. Mapped to color range (0, 255).
            minmax_pos_y (int, int): The minimum and maximum y-position values. Mapped to color range (0, 255).
            minmax_pos_z (int, int): The minimum and maximum z-position values. Mapped to color range (0, 255).
    """
    img = seq.to_motionimg(output_size=output_size)
    if show_img:
        cv2.imshow(seq.name, img)
        print(f"Showing motion image from [{seq.name}]. Press any key to close the image and continue.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if show_skeleton:
        sv = SkeletonVisualizer(seq)
        sv.show()
    return img
