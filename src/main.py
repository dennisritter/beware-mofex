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
featvec_path = 'data/feature_vectors/mir-single/resnet34-512.json'

# ### Load Sequences
# sequences = []
# for filename in Path(path).rglob('*.json'):
#     name = str(filename).split("\\")[-1]
#     print(f"Loading [{filename}]")
#     sequences.append(Sequence.from_mir_file(filename, name=name))

### Specify Model
model = resnet.load_resnet34(pretrained=True, remove_last_layer=True)
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
results = {}
# for idx in random_indices:
for q_idx in range(len(feature_vectors)):
    q_feat = feature_vectors[q_idx]
    q_filename = names[q_idx]
    # print("------------------------------")
    # print(f"Distances for [{q_filename}]")
    distances = []
    for i, test_feat in enumerate(feature_vectors):
        distance = np.linalg.norm(q_feat - test_feat)
        distances.append(distance)
    # Print Top 5 lowest distances (But not the Ground Truth sequence itself)
    dist_top5 = sorted(range(len(distances)), key=lambda i: distances[i])[1:6]
    # print(f"[{names[dist_idx]}] : {distances[dist_idx]}")
    top5 = [(names[i], distances[i]) for i in dist_top5]
    results[names[q_idx]] = top5

classes = [
    'biceps_curl_left', 'biceps_curl_right', 'knee_lift_left', 'knee_lift_right', 'lunge_left', 'lunge_right', 'overhead_press', 'side_step', 'squat',
    'triceps_extension_left', 'triceps_extension_right'
]
top1_correct = 0
top1_incorrect = 0
for k in results.keys():
    for class_str in classes:
        if class_str in k:
            for r_idx, r in enumerate(results[k]):
                if class_str in r[0]:
                    if r_idx == 0:
                        top1_correct += 1
                else:
                    print(f"Wrong Guess: [rank={r_idx}] {k}, {r}, {class_str}")
                    if r_idx == 0:
                        top1_incorrect += 1
print(f"Top1 Correct [n = {len(results.keys())}] {top1_correct} ({top1_correct / (top1_correct+top1_incorrect)})")
print(f"Top1 Incorrect [n = {len(results.keys())}] {top1_incorrect} ({top1_incorrect / (top1_correct+top1_incorrect)})")