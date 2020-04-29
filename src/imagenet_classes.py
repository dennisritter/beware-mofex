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

### ESTIMATE IMAGENET CLASS OF "NORMAL" RGB IMAGE
# # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
# print(output[0])
# # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
# softmax_output = torch.nn.functional.softmax(output[0], dim=0)
# output_class_idx = torch.max(softmax_output, 0).indices
# output_class_probability = torch.max(softmax_output, 0)
# print(output_class_probability)

# # Print the identified class
# dir = os.path.dirname(__file__)
# with open(os.path.join(dir, 'mofex/postprocessing/imagenet-simple-labels.json')) as f:
#     labels = json.load(f)

# def class_id_to_label(i):
#     return labels[i]

# print(class_id_to_label(output_class_idx))
