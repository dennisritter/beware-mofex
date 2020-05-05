""" Compares feature vectors in order to match the types of motions they represent
"""
import numpy as np
import cv2
import json
import os
import random
from datetime import datetime
from pathlib import Path
import torch
from torchvision import transforms
from mofex.preprocessing.sequence import Sequence
from mofex.preprocessing.skeleton_visualizer import SkeletonVisualizer
import mofex.models.resnet as resnet
import mofex.feature_vectors as featvec


def evaluate_matching_top1(train_featvecs: list, val_featvecs: list, make_graph=False, result_dir_path=None):
    """ Returns a list of estimated most similar motion sequence tuples including feature vectors and respective classes.
        
        Args:
            train_featvecs (list): A list of tuples, which contain feature vectors and motion classes of a motion image the feature extracting CNN has been trained with (train set).
            val_featvecs (list): A list of tuples, which contain feature vectors and motion classes of a motion image the feature extracting CNN has been validated with (val/test set).        
    """
    # for featvec_tuple in val_featvecs:
    #     id_name = featvec_tuple[0]
    #     featvec = featvec_tuple[1]
    #     label = featvec_tuple[2]

    # TODO: Perform top1 matching evaluation
    # TODO: Create Graph and save image if make_graph=True
    # TODO: Write result string
    result = 'add result here'
    # Check if result log path given and ensure directory is present/created
    if result_dir_path:
        if not os.path.isdir(result_dir_path):
            os.makedirs(result_dir_path)
        filename = f'{result_dir_path}/top1_matching_result.txt'
        result_file = open(filename, 'w')
        result_file.write(result)
        result_file.close()
