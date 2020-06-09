# import numpy as np
# import cv2
# import json
# import os
import random
from pathlib import Path
from queue import Queue
import time
import click
import numpy as np
from torchvision import transforms
from scipy.signal import argrelextrema  #, savgol_filter
import mofex.model_loader as model_loader
from mofex.preprocessing.sequence import Sequence
from mofex.preprocessing.skeleton_visualizer import SkeletonVisualizer
import mofex.preprocessing.normalizations as norm
import mofex.feature_vectors as feature_vectors

# ! Root directory of motion images. Make sure it includes 'train' and 'val' directory, which then include a directory for each present label in the dataset.
# ! Structure template: <in_path>/train/<class_name>/motion_img1.png
dataset_name = 'hdm05-122_90-10'
# CNN Model name -> model_dataset-numclasses_train-val-ratio
model_name = 'resnet101_hdm05-122_90-10'
# The CNN Model for Feature Vector generation
model = model_loader.load_trained_model(model_name=model_name,
                                        remove_last_layer=True,
                                        state_dict_path=f'./data/trained_models/{dataset_name}/{model_name}_e50.pt')
# The models output size (Feature Vector length
feature_size = 2048
# Transforms
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_seqs(root, regex_str):
    seqs = []
    for filename in Path(root).rglob(regex_str):
        print(f'Loading Sequence from file: {filename}')
        seqs.append(Sequence.from_hdm05_c3d_file(filename))
    return seqs


root_gt = 'data/sequences/hdm05-122/c3d/squat1Reps/'
filename_gt = 'HDM_bd_squat1Reps_001_120.C3D'
seqs_gt = load_seqs(root_gt, filename_gt)
seqs_gt_normed = []
for seq in seqs_gt:
    seq_normed = seq[:]
    # seq_normed.norm_center_positions()
    # ! 30 and 37 are position indices somewhere near left/right hip.
    # ! Adjust if using other body part model! (currently for hdm05 dataset sequences)
    # seq_normed.norm_relative_to_positions((seq_normed.positions[:, 30, :] + seq_normed.positions[:, 37, :]) * 0.5)
    # seq_normed.norm_orientation_first_pose_frontal_to_camera(30, 37)
    seqs_gt_normed.append(seq)
# * hacky.. Just take the first sequence
seq_gt_normed = seqs_gt_normed[0]
# TODO: Remove debug visualization
#sv = SkeletonVisualizer(seq_gt_normed)
#sv.show()
featvec_gt = feature_vectors.load_from_sequences([seq_gt_normed], model, preprocess, feature_size)[0]

root_q = 'data/sequences/hdm05-122/c3d/squat3Reps/'
pattern_q = '*.C3D'
seqs_q = load_seqs(root_q, pattern_q)


def fill_queue(queue: Queue, batchsize: int = 10):
    """ Just a helper function to fill the queue with sequence batches"""
    seq = seqs_q[random.randrange(len(seqs_q))]
    sv = SkeletonVisualizer(seq)
    sv.show()
    batched_seq = seq.split_into_batches(batchsize)
    for batch in batched_seq:
        queue.put(batch)
    return queue


@click.command()
@click.option("--batchsize", default=10, help="How many frames do you want to send in each package?")
@click.option("--fps", default=120, help="How many frames per second do you want to stream?")
@click.option("--delay", default=0, help="How much delay you want to add between batches?")
def stream(batchsize, fps, delay):
    state = {
        "paused": False,
    }
    click.echo(f'Simulating motion stream:\nFPS: {fps}\nBatchSize: {batchsize}')
    click.echo(f'Sending a batch of [{batchsize}] frames every [{(batchsize / fps):.2f}] seconds...')
    t = 0
    seq_q_queue = Queue(maxsize=0)
    seq_q_queue = fill_queue(seq_q_queue, batchsize)

    # Init seq_q, the stitched query sequence
    seq_q = None
    distances = []
    keyframes = []
    seqs_q_single_rep = []
    while True:
        time.sleep((batchsize / fps) + delay)
        t += batchsize / fps
        click.echo(f'[{round(t, 2):.2f}] Get next {batchsize} frames from queue.')

        # Handle empty queue
        if seq_q_queue.empty():
            seq_q = None
            seq_q_queue = fill_queue(Queue(maxsize=0))
        # Create query sequence if it is None
        if not seq_q:
            print('-' * 10)
            print(f'Keyframes: {keyframes}')
            print('-' * 10)
            print('You want to test another Query Sequence?')
            click.pause()
            distances = []
            keyframes = []
            seqs_q_single_rep = []
            seq_q = seq_q_queue.get()
        # Append next batch if query sequence exists
        else:
            seq_q.append(seq_q_queue.get())

        # * Normalize the query sequence in a copy
        # * Otherwise we normalize multiple times, which we shouldn't
        seq_q_normed = seq_q[:]
        print(f'Q LENGTH: {len(seq_q_normed)}')
        # seq_q_normed.norm_center_positions()
        # ! 30 and 37 are position indices somewhere near left/right hip.
        # ! Adjust if using other body part model! (currently for hdm05 dataset sequences)
        # seq_q_normed.norm_relative_to_positions((seq_q_normed.positions[:, 30, :] + seq_q_normed.positions[:, 37, :]) * 0.5)
        # seq_q_normed.norm_orientation_first_pose_frontal_to_camera(30, 37)
        # TODO: Remove debug visualization
        # sv = SkeletonVisualizer(seq_q_normed)
        # sv.show()
        # click.pause()

        # Make motion image from sequence
        # seq_q_normed.to_motionimg()
        # Make feature vector from motion image
        featvec_q = feature_vectors.load_from_sequences([seq_q_normed], model, preprocess, feature_size)[0]

        distance = np.linalg.norm(featvec_q - featvec_gt)
        distances.append(distance)
        minima = argrelextrema(np.array(distances), np.less, order=5)[0]
        if len(minima >= 1):
            print(f'I found a distance minimum after [{(minima[0] + 1) * batchsize}] frames.')
            keyframes.append((minima[0] + 1) * batchsize)
            seqs_q_single_rep.append(seq_gt_normed[0:minima[0]])
            seq_q = seq_q[minima[0]:]
            distances = distances[minima[0]:]


if __name__ == '__main__':
    # start_message()
    stream()
