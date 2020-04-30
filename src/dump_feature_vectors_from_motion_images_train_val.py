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
import mofex.model_loader as model_loader
import mofex.feature_vectors as feature_vectors

# ----- Params
# Root folder of motion images. Make sure it includes 'train' and 'val' folder
in_path = 'data/motion_images/hdm05-122'
# CNN Model name -> model_dataset-numclasses_train-val-ratio
model_name = 'resnet18_hdm05-122_50-50'
# The CNN Model for Feature Vector generation
model = model_loader.load_trained_model(model_name=model_name,
                                        remove_last_layer=True,
                                        state_dict_path='./data/trained_models/hdm05-122/resnet18_hdm05-122_e10.pt')
# The models output size (Feature Vector length
feature_size = 512
# Transforms
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Dataset name
dataset_name = 'hdm05-122'

feature_vectors.dump_from_motion_images_train_val(in_path=in_path,
                                                  model=model,
                                                  model_name=model_name,
                                                  feature_size=feature_size,
                                                  dataset_name=dataset_name,
                                                  preprocess=preprocess)
