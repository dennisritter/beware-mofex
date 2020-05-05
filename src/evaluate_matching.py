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

featvec_path = 'data/feature_vectors/hdm05-122/resnet18-512_hdm05-122_50-50-e10.json'

featvecs = featvec.load_from_file(featvec_path)

# Make two lists of list<tuples> for feature vectors and corresponding names
featvecs_list = list(map(list, zip(*featvecs)))
names = featvecs_list[0]
classes = [x[0].split("/")[-1] for x in os.walk('data/motion_images/hdm05-122/val/')]
feature_vectors = np.array(featvecs_list[1])
