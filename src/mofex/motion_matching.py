""" Compares feature vectors in order to match the types of motions they represent
"""
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


def evaluate_matching_top1(train_featvecs: list, val_featvecs: list, make_graph=False, result_dir_path=None):
    """ Returns a list of estimated most similar motion sequence tuples including feature vectors and respective classes.
        
        Args:
            train_featvecs (list): A list of tuples, which contain id_name, feature vector and motion class of a motion image the feature extracting CNN has been trained with (train set).
            val_featvecs (list): A list of tuples, which contain id_name, feature vector and motion class of a motion image the feature extracting CNN has been validated with (val/test set).        
            make_graph (bool): Determine whether to create a result graph or not. (default=False)
            result_dir_path (bool): Enter a path to store a result file. If None, no result file is created. (default=None) 
    """

    # Determine top1 lowest distances by comparing all val_featvecs to all train_featvecs
    top1_results = []
    correct = 0
    incorrect = 0
    for val_featvec_tuple in val_featvecs:
        val_id_name = val_featvec_tuple[0]
        val_featvec = np.array(val_featvec_tuple[1])
        val_label = val_featvec_tuple[2]
        distances = []
        for train_featvec_tuple in train_featvecs:
            train_id_name = train_featvec_tuple[0]
            train_featvec = np.array(train_featvec_tuple[1])
            train_label = train_featvec_tuple[2]

            distances.append(np.linalg.norm(val_featvec - train_featvec))
        top1_idx = distances.index(min(distances))
        top1_results.append((val_id_name, val_label, train_featvecs[top1_idx][0], train_featvecs[top1_idx][2]))
        # print(f'TOP1: {train_featvecs[top1_idx][0]}:\ndist = {distances[top1_idx]}\nlabel = {train_featvecs[top1_idx][2]}')
        # print('-' * 10)
        if val_label == train_featvecs[top1_idx][2]:
            correct += 1
        else:
            incorrect += 1
    print('-' * 10)
    print(f'Finished Top1 Matching evaluation:')
    print(f'Items Compared: n_train={len(train_featvecs)}, n_val={len(val_featvecs)}, total_comparisons={len(train_featvecs)*len(val_featvecs)}')
    print(f'Correct: {correct}/{len(val_featvecs)}')
    print(f'Incorrect: {incorrect}/{len(val_featvecs)}')
    print(f'Accuracy: {correct / len(val_featvecs)}')
    print('-' * 10)
    # TODO: Create Graph and save image if make_graph=True
    # TODO: Write result string
    result = 'add result here'
    # Check if result log path given and ensure directory is present/created
    if result_dir_path:
        if not os.path.isdir(result_dir_path):
            os.makedirs(result_dir_path)
        filename = f'{result_dir_path}/top1_matching_result.txt'
        result_file = open(filename, 'w')
        for result in top1_results:
            if result[1] == result[3]:
                result_file.write(f'CORRECT {str(result)}\n')
            else:
                result_file.write(f'WRONG {str(result)}\n')
        result_file.close()
