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


def evaluate_matching_top1(train_featvecs: list, val_featvecs: list):
    """ Returns a list of estimated most similar motion sequence tuples including feature vectors and respective classes.
        
        Args:
            train_featvecs (list): A list of tuples, which contain feature vectors and motion classes of a motion image the feature extracting CNN has been trained with (train set).
            val_featvecs (list): A list of tuples, which contain feature vectors and motion classes of a motion image the feature extracting CNN has been validated with (val/test set).        
    """
    pass