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
import mofex.feature_vectors as featvec
import mofex.preprocessing.normalizations as norm
import torch
from ezc3d import c3d
from torchvision import transforms
""" 
1. Reads Sequence files
2. Creates Motion Images
3. Creates Feature vectors from motion images
4. Stores Filename(key) Feature Vectors(value) in dict
5. Saves .JSON file that represents the dict  
"""
start = datetime.now()

# Root folder for JSON Sequence files
sequences_path = 'data/sequences/hdm05-122/c3d/'
# Filepath for feature dict
dump_path = 'data/feature_vectors/hdm05-122/resnet18-512-hdm05-c3d-bp44-120hz__2.json'
# The CNN Model for Feature Vector generation
model = resnet.load_resnet18(pretrained=True, remove_last_layer=True)
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

### Load Sequences
sequences = []
for filename in Path(sequences_path).rglob('*.C3D'):
    print(f"Loading [{filename}]")
    seq = Sequence.from_hdm05_c3d_file(path=filename, name=str(filename).split("\\")[-1])
    # The point in between hip_l and hip_r in HDM05 C3D files (44 tracked points/frame)
    seq.norm_relative_to_positions((seq.positions[:, 30, :] + seq.positions[:, 37, :]) * 0.5)
    sequences.append(seq)

feature_vectors = featvec.load_from_sequences(sequences, model, preprocess, 512)
print(f"Created Feature Vectors: {datetime.now() - start}")

### Store feature vectors in file
filename_featvec_dict = {}
for i, seq in enumerate(sequences):
    filename_featvec_dict[seq.name] = feature_vectors[i].tolist()
with open(dump_path, 'w') as data_file:
    json.dump(filename_featvec_dict, data_file)

print(f"Runtime of script: {datetime.now() - start}")
