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

featvec_path = 'data/feature_vectors/hdm05-122/resnet18_hdm05-122_50-50/train/resnet18_hdm05-122_50-50_train.json'

featvecs = featvec.load_from_file(featvec_path)

# Make two lists of list<tuples> for feature vectors and corresponding names
featvecs_list = list(map(list, zip(*featvecs)))

names = featvecs_list[0]
feature_vectors = np.array(featvecs_list[1])
labels = featvecs_list[2]

for i, name in enumerate(names):
    print(f'{names[i]}: {labels[i]} {feature_vectors[i]}')
