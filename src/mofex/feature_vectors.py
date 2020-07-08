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
import torch
from torchvision import transforms


# ! Deprecated, don't use until refactored
# TODO: Refactor and make sure correct parameters etc. are chosen
def load_from_sequences(sequences: list, cnn_model, cnn_preprocess, cnn_output_size=512) -> list:
    start = datetime.now()
    feature_vectors = []

    cnn_model.eval()
    if torch.cuda.is_available():
        cnn_model = cnn_model.to('cuda')

    for seq in sequences:
        motion_img = seq.to_motionimg(show_img=False, show_skeleton=False)
        # ? What exactly happens here?
        input_tensor = cnn_preprocess(motion_img)
        # ? What exactly happens here?
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')

        output = cnn_model(input_batch)
        # ? What exactly happens here?
        output = output.cpu().detach().numpy().reshape((cnn_output_size))
        feature_vectors.append(output)
    print(f"Loaded Feature Vectors from [{len(sequences)}] Sequences [{datetime.now() - start}]")
    return feature_vectors


# ! Deprecated, don't use until refactored
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
        elif tracking_type.lower() == 'hdm05':
            sequences.append(Sequence.from_hdm05_c3d_file(filename, name=name))
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
    feature_vectors = [(k, v[0], v[1]) for k, v in featvec_dict.items()]
    return feature_vectors


def load_from_motion_imgs(motion_images, model, feature_size, preprocess):
    ### Get feature Vectors from CNN
    model.eval()
    # Detect if GPU available and add model to it
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    feature_vectors = []
    for img in motion_images:
        img_tensor = torch.from_numpy(img)
        input_tensor = preprocess(img)
        # ? What exactly happens here?
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
        input_batch = input_batch.to(device)

        output = model(input_batch)
        output = output.cpu().detach().numpy().reshape((feature_size))
        feature_vectors.append(output)
    return feature_vectors


# ! Deprecated, don't use until refactored
def dump_from_motion_images_train_val(in_path, model, model_name, feature_size, dataset_name, preprocess):
    """ Saves Feature Vector JSON files under ./data/feature_vectors/<dataset_name>/<model_name>/<model_name>_[train,val].json

        Args:
            in_path: The root path to motion image .png files. Must include folders [train, val], which must include folders that represent classes, which then include the motion image .png files.
            model: The CNN model that will be used to extract the feature vectors.
            model_name: The name of the model. Used for directory and file naming.
            feature_size: The level 0 output dimension of the model, which is the size of generated feature vectors.
            dataset_name: The nameof the dataset. Used for directory and file naming.
            preprocess: The input transformations (torchvision.transforms.Compose).
    """
    PHASES = ['train', 'val']
    for phase in PHASES:
        in_path_phase = f'{in_path}/{phase}'
        # Where to store the feature_vectors.json file?
        out_dir = f'data/feature_vectors/{dataset_name}/{model_name}/{phase}'
        out_filename = f'{model_name}_{phase}.json'
        # Check that out_path is save
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        out_path = f'{out_dir}/{out_filename}'

        # ----- Actual Functionality
        filenames = []
        motion_images = []
        labels = []
        for filename in Path(in_path_phase).rglob('*.png'):
            print(filename)
            # * Ensure that the in_path_phase path contains directories with the names of the present labels. These 'label directories' then contain the motion images.
            label = str(filename).split('\\')[-2]
            img = cv2.imread(os.path.abspath(filename))
            # TODO: Check if we need to convert colorspace
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            motion_images.append(img)
            filenames.append(filename)
            labels.append(label)

        model.eval()
        # Detect if GPU available and add model to it
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        ### Get feature Vectors from CNN
        feature_vectors = []
        for img in motion_images:
            img_tensor = torch.from_numpy(img)
            input_tensor = preprocess(img)
            # ? What exactly happens here?
            input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
            input_batch = input_batch.to(device)

            output = model(input_batch)
            output = output.cpu().detach().numpy().reshape((feature_size))
            feature_vectors.append(output)

        ### Map motion image filename to [feature vectors, labels] list in json file. (tuples are not supported by json and will be converted to list)
        filename_featvec_dict = {}
        for i, name in enumerate(filenames):
            # We use the motion images filename as an ID for the feature vector dictionary.
            # * Make sure each filename is unique or this will override keys that are present already
            seq_id = str(name).split("\\")[-1]
            filename_featvec_dict[str(seq_id)] = (feature_vectors[i].tolist(), labels[i])

        with open(out_path, 'w') as data_file:
            json.dump(filename_featvec_dict, data_file)

        print(f'Finished feature vector file creation. [dataset={dataset_name}, model={model_name}, phase={phase}, n={len(feature_vectors)}]')
