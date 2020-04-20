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
path = 'data/sequences/hdm05/c3d/'
# Filepath for feature dict
dump_path = 'data/feature_vectors/hdm05/resnet18-512-hdm05-c3d-bp44-120hz.json'
# The CNN Model for Feature Vector generation
model = resnet.load_resnet18(pretrained=True, remove_last_layer=True)

### Load Sequences
filenames = []
motion_images = []
for filename in Path(path).rglob('*.C3D'):
    name = str(filename).split("\\")[-1]
    filenames.append(filename)
    # with open(filename, 'rb') as handle:
    print(f"Loading [{filename}]")
    c3d_object = c3d(str(filename))
    positions = c3d_object['data']['points']
    positions = positions.swapaxes(0, 2)[:, :, :3]
    positions = norm.center_positions(positions)
    motion_images.append(featvec.motion_image_from_3d_positions(positions, name=name, show_img=False))

print(f"Created Motion Images: {datetime.now() - start}")
### Get feature Vectors from CNN

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

feature_vectors = []
for img in motion_images:
    model.eval()
    img_tensor = torch.from_numpy(img)

    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    output = model(input_batch)
    output = output.cpu().detach().numpy().reshape((512))
    feature_vectors.append(output)

print(f"Created Feature Vectors: {datetime.now() - start}")

### Store feature vectors in file
filename_featvec_dict = {}
for i, name in enumerate(filenames):
    seq_name = str(name).split("\\")[-1]
    filename_featvec_dict[str(seq_name)] = feature_vectors[i].tolist()

path = dump_path
with open(path, 'w') as data_file:
    json.dump(filename_featvec_dict, data_file)

print(f"Runtime of script: {datetime.now() - start}")
