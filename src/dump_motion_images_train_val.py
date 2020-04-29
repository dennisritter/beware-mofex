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
from sklearn.model_selection import train_test_split
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
# root = 'data/sequences/hdm05-122/c3d'
root = 'data/sequences/hdm05-122/c3d'

### Load Sequences
filenames = []
sequences = []
for filename in Path(root).rglob('*.c3d'):
    print(filename)
    # ! Change the loading function depending on the MoCap format (from_mir_file, from_mka_file, ...)
    name = str(filename).split("\\")[-1]
    desc = str(filename).split("\\")[-2]  # desc = class
    sequences.append(Sequence.from_hdm05_c3d_file(filename, name=name, desc=desc))
print(f"Loaded Sequence Files: {datetime.now() - start}")
train_seqs, val_seqs = train_test_split(sequences, test_size=0.5, random_state=42)

# Train Set
for seq in train_seqs:
    # set and create directories
    class_dir = f'data/motion_images/hdm05-122/train/{seq.desc}'
    out = f'{class_dir}/{seq.name.replace(".C3D", ".png")}'
    if not os.path.isdir(os.path.abspath(class_dir)):
        os.makedirs(os.path.abspath(class_dir))
    # save motion img
    img = seq.to_motionimg(output_size=(256, 256))
    cv2.imwrite(out, img)
# Validation Set
for seq in val_seqs:
    # set and create directories
    class_dir = f'data/motion_images/hdm05-122/val/{seq.desc}'
    out = f'{class_dir}/{seq.name.replace(".C3D", ".png")}'
    if not os.path.isdir(os.path.abspath(class_dir)):
        os.makedirs(os.path.abspath(class_dir))
    # save motion img
    img = seq.to_motionimg(output_size=(256, 256))
    cv2.imwrite(out, img)

print(f"Created, split and dumped Motion Images: {datetime.now() - start}")
