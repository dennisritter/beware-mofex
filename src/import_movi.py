import numpy as np
import cv2
import json
import os
import random
import scipy.io
from datetime import datetime
from pathlib import Path
import torch
from torchvision import transforms
from mofex.preprocessing.sequence import Sequence
from mofex.preprocessing.skeleton_visualizer import SkeletonVisualizer
import mofex.models.resnet as resnet
import mofex.feature_vectors as featvec

movi_path = 'data/sequences/movi/s_run'

### Load Movi data
for filename in Path(movi_path).rglob('S_v3d_Subject_1.mat'):
    name = str(filename).split("\\")[-1]
    print(f"Loading [{filename}]")
    mat = scipy.io.loadmat(filename)

    subject_data = mat['Subject_1_S']

    print(subject_data[0, 0])
# ### Specify Model
# model = resnet.load_resnet18(pretrained=True, remove_last_layer=True)
# if torch.cuda.is_available():
#     model.to('cuda')
# ### Specify Input Image Preprocessing
# preprocess = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize(256),
#     transforms.CenterCrop(256),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
# model.eval()

# # featvecs = featvec.load_from_sequences(sequences, model, preprocess)
# # featvecs = featvec.load_from_sequences_dir(seqs_path, 'mir', model, preprocess)
# featvecs = featvec.load_from_file(featvec_path)

# # Make two lists of list<tuples> for feature vectors and corresponding names
# featvecs_list = list(map(list, zip(*featvecs)))
# names = featvecs_list[0]
# feature_vectors = np.array(featvecs_list[1])

# ### COMPARE FEATVECS DISTANCES
# # Get n_gt_featvecs4 random sequences
# n_gt_featvecs = 5
# random_indices = []
# for i in range(n_gt_featvecs):
#     random_indices.append(random.randrange(len(feature_vectors)))
# for idx in random_indices:
#     gt_feat = feature_vectors[idx]
#     gt_filename = names[idx]
#     print("------------------------------")
#     print(f"Distances for [{gt_filename}]")
#     distances = []
#     for i, test_feat in enumerate(feature_vectors):
#         distance = np.linalg.norm(gt_feat - test_feat)
#         distances.append(distance)
#     # Print Top 5 lowest distances (But not the Ground Truth sequence itself)
#     dist_top5 = sorted(range(len(distances)), key=lambda i: distances[i])[1:6]
#     for dist_idx in dist_top5:
#         print(f"[{names[dist_idx]}] : {distances[dist_idx]}")
