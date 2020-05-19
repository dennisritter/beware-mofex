import numpy as np
import cv2
import json
import os
import random
import math
from datetime import datetime
from pathlib import Path
import torch
from torchvision import transforms
from mofex.preprocessing.sequence import Sequence
from mofex.preprocessing.skeleton_visualizer import SkeletonVisualizer
import mofex.models.resnet as resnet
import mofex.feature_vectors as featvec
import plotly.graph_objects as go
import mofex.acm_asf_parser.amc_parser as amc_asf_parser

asf_path = './data/sequences/hdm05-10/amc/cartwheelLHandStart1Reps/HDM_bd.asf'
amc_path = './data/sequences/hdm05-10/amc/cartwheelLHandStart1Reps/HDM_bd_cartwheelLHandStart1Reps_003_120.amc'
seq = Sequence.from_hdm05_asf_amc_files(asf_path, amc_path)

seq.norm_center_positions()
seq.norm_relative_to_positions((seq.positions[:, 1, :] + seq.positions[:, 6, :]) * 0.5)
seq.norm_orientation_first_pose_frontal_to_camera(1, 6)

sv = SkeletonVisualizer(seq)
sv.show(auto_open=True)

a = 1