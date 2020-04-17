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

seqs_path = 'data/sequences/191024_mir/single/squat/user-1'
featvec_path = 'data/feature_vectors/mir-single/resnet18-512.json'

# ### Load Sequences
# sequences = []
# for filename in Path(path).rglob('*.json'):
#     name = str(filename).split("\\")[-1]
#     print(f"Loading [{filename}]")
#     sequences.append(Sequence.from_mir_file(filename, name=name))

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

# featvecs = featvec.load_from_sequences(sequences, model, preprocess)
# featvecs = featvec.load_from_sequences_dir(seqs_path, 'mir', model, preprocess)
featvecs = featvec.load_from_file(featvec_path)

# Make two lists of list<tuples> for feature vectors and corresponding names
featvecs_list = list(map(list, zip(*featvecs)))
names = featvecs_list[0]
feature_vectors = np.array(featvecs_list[1])

### COMPARE FEATVECS DISTANCES
# Get n_gt_featvecs4 random sequences
n_gt_featvecs = 5
random_indices = []
for i in range(n_gt_featvecs):
    random_indices.append(random.randrange(len(feature_vectors)))
for idx in random_indices:
    gt_feat = feature_vectors[idx]
    gt_filename = names[idx]
    print("------------------------------")
    print(f"Distances for [{gt_filename}]")
    distances = []
    for i, test_feat in enumerate(feature_vectors):
        distance = np.linalg.norm(gt_feat - test_feat)
        distances.append(distance)
    # Print Top 5 lowest distances (But not the Ground Truth sequence itself)
    dist_top5 = sorted(range(len(distances)), key=lambda i: distances[i])[1:6]
    for dist_idx in dist_top5:
        print(f"[{names[dist_idx]}] : {distances[dist_idx]}")

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