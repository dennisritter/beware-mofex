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

path = 'data/sequences/191024_mir/single/squat/user-1'

### Load Sequences
sequences = []
for filename in Path(path).rglob('*.json'):
    name = str(filename).split("\\")[-1]
    print(f"Loading [{filename}]")
    sequences.append(Sequence.from_mir_file(filename, name=name))

### Specify Model
model = resnet.load_resnet18(pretrained=True, remove_last_layer=True)
if torch.cuda.is_available():
    model.to('cuda')
### Specify Input Image Preprocessing
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
model.eval()

# feature_vectors = featvec.load_from_sequences(sequences, model, preprocess)
# feature_vectors = featvec.load_from_sequences_dir(path, 'mir', model, preprocess)

# Get x random sequences
random_indices = []
for i in range(5):
    random_indices.append(random.randrange(len(feature_vectors)))
for idx in random_indices:
    gt_feat = feature_vectors[idx]
    gt_filename = sequences[idx].name
    print("------------------------------")
    print(f"Distances for [{gt_filename}]")
    distances = []
    for i, test_feat in enumerate(feature_vectors):
        distance = np.linalg.norm(gt_feat - test_feat)
        distances.append(distance)
    # Print Top 5 lowest distances (But not the Ground Truth sequence itself)
    dist_top5 = sorted(range(len(distances)), key=lambda i: distances[i])[1:11]
    for dist_idx in dist_top5:
        print(f"[{sequences[dist_idx].name}] : {distances[idx]}")

### COMPARE ALL FEATURE VECTORS WITH EACH OTHER
# for i, gt_feat in enumerate(feature_vectors):
#     gt_filename = filenames[i]
#     # print("------------------------------")
#     # print(f"Distances for [{gt_filename}]")
#     for j, test_feat in enumerate(feature_vectors):
#         test_filename = filenames[j]
#         distance = np.linalg.norm(gt_feat - test_feat)
#         # print(f"[{test_filename}] : {distance}")

#     # print("------------------------------")

# print(
#     f"Creating Motion Images [{len(motion_images)}], Creating Feature Vectors [{len(feature_vectors)}], Comparing Feature Vectors [{len(feature_vectors)*len(feature_vectors)}]"
# )
# print(f"Runtime of script: {datetime.now() - start}")