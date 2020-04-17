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
        feature_vectors.append(output)
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
    pass


def _motion_img_from_sequence(
    seq: 'Sequence',
    output_size: tuple = (256, 256),
    show_img: bool = False,
    show_skeleton: bool = False,
):
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
