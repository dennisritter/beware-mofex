from queue import Queue
import time
import click
import numpy as np
from torchvision import transforms
import mofex.model_loader as model_loader
from mofex.preprocessing.sequence import Sequence
from mofex.load_sequences import load_seqs_asf_amc_hdm05
from mofex.rep_counter import RepCounter
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from pathlib import Path

import mana.utils.math.normalizations as normalizations
from mana.models.sequence import Sequence
from mana.utils.data_operations.loaders.sequence_loader_mka import SequenceLoaderMKA
from mana.models.sequence_transforms import SequenceTransforms

# ! Root directory of motion images. Make sure it includes 'train' and 'val' directory, which then include a directory for each present label in the dataset.
# ! Structure template: <in_path>/train/<class_name>/motion_img1.png
dataset_name = 'mka-beware-1.1'
# CNN Model name -> model_dataset-numclasses_train-val-ratio
model_name = 'resnet101_mka-beware-1.1'
# The CNN Model for Feature Vector generation
model = model_loader.load_trained_model(model_name=model_name,
                                        remove_last_layer=True,
                                        state_dict_path=f'./data/trained_models/{dataset_name}/{model_name}_e5.pt')
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

# Root folder for Query Sequence files
### OVERHEADPRESS Dennis
# src_root = './data/sequences/mka-beware-1.1-sets/Dennis/overheadpress'
### TIPTOESTAND Dennis
# src_root = './data/sequences/mka-beware-1.1-sets/Dennis/tiptoestand'
### TIPTOESTAND Christopher
src_root = './data/sequences/mka-beware-1.1-sets/Christopher/tiptoestand'
### OVERHEADPRESS Philippe
# src_root = './data/sequences/mka-beware-1.1-sets/Philippe/overheadpress'
### OVERHEADPRESS Christopher
# src_root = './data/sequences/mka-beware-1.1-sets/Christopher/overheadpress'
# src_root = './data/sequences/mka-beware-1.1-sets/Christopher/overheadpress/19-08-2020-02-59-14'
# src_root = './data/sequences/mka-beware-1.1-sets/Christopher/overheadpress/19-08-2020-02-59-14'
# src_root = './data/sequences/mka-beware-1.1-sets/Christopher/overheadpress/19-08-2020-02-59-14'
# src_root = './data/sequences/mka-beware-1.1-sets/Christopher/overheadpress/19-08-2020-02-59-14'
# src_root = './data/sequences/mka-beware-1.1-sets/Christopher/overheadpress/19-08-2020-02-59-14'
# src_root = './data/sequences/mka-beware-1.1-sets/Christopher/overheadpress/19-08-2020-02-59-14'
# src_root = './data/sequences/mka-beware-1.1-sets/Christopher/overheadpress/19-08-2020-02-59-14'
# src_root = './data/sequences/mka-beware-1.1-sets/Christopher/overheadpress/19-08-2020-02-59-14'
# src_root = './data/sequences/mka-beware-1.1-sets/Christopher/overheadpress/19-08-2020-02-59-14'
# src_root = './data/sequences/mka-beware-1.1-sets/Christopher/overheadpress/19-08-2020-02-59-14'
### SQUAT Christopher
# src_root = './data/sequences/mka-beware-1.1-sets/Christopher/squat'
# src_root = './data/sequences/mka-beware-1.1-sets/Christopher/squat/19-08-2020-12-10-48'
# src_root = './data/sequences/mka-beware-1.1-sets/Christopher/squat/19-08-2020-12-09-43'
# src_root = './data/sequences/mka-beware-1.1-sets/Christopher/squat/19-08-2020-12-08-41'
# src_root = './data/sequences/mka-beware-1.1-sets/Christopher/squat/19-08-2020-12-07-41'
# src_root = './data/sequences/mka-beware-1.1-sets/Christopher/squat/19-08-2020-12-06-00'
# src_root = './data/sequences/mka-beware-1.1-sets/Christopher/squat/19-08-2020-12-03-55'
# src_root = './data/sequences/mka-beware-1.1-sets/Christopher/squat/19-08-2020-12-02-40'
# src_root = './data/sequences/mka-beware-1.1-sets/Christopher/squat/19-08-2020-12-01-33'
# src_root = './data/sequences/mka-beware-1.1-sets/Christopher/squat/19-08-2020-11-59-52'
# src_root = './data/sequences/mka-beware-1.1-sets/Christopher/squat/19-08-2020-11-58-05'
ground_truth = 'tiptoestand_225.json'
exercise = 'tiptoestand'

seq_transforms = SequenceTransforms(SequenceTransforms.mka_to_iisy())
seq_loader = SequenceLoaderMKA(seq_transforms)
# Load sequences
seqs = []
for filename in Path(src_root).rglob('*.json'):
    # print(str(filename).replace('\\', '/').split('/'))
    filename_split = str(filename).replace('\\', '/').split('/')
    seq_name = filename_split[-1]  # The filename (eg: 'myname.json')
    seq_class = filename_split[-2]  # The class (eG: 'squat')
    # Load
    seq = seq_loader.load(path=f'{str(filename)}', name=seq_name[:-5], desc=seq_class)
    print(f'{seq.name}: frames = {len(seq)} ')
    seqs.append(seq)
# Append Sequences to one set (one sequence)
for seq in seqs[1:]:
    seqs[0].append(seq)
seq_q = seqs[0]
print(f'Created query sequence of {len(seq_q)} frames.')


def fill_queue(queue: Queue, batchsize: int = 10):
    """ Just a helper function to fill the queue with sequence batches"""
    seq_q_batches = seq_q.split(overlap=0.0, subseq_size=batchsize)
    for batch in seq_q_batches:
        queue.put(batch)
    return queue


# Load Ground Truth Sequence
seq_name = ground_truth
seq_class = exercise
seq_gt = seq_loader.load(path=f'./data/sequences/mka-beware-1.1/{seq_class}/{seq_name}', name=seq_name[:-5], desc=seq_class)
repcounter = RepCounter(seq_gt=seq_gt[0:4], subseq_len=4, savgol_win=21, model=model, feature_size=feature_size, preprocess=preprocess)


@click.command()
@click.option("--batchsize", default=10, help="How many frames do you want to send in each package?")
@click.option("--fps", default=30, help="How many frames per second do you want to stream?")
@click.option("--delay", default=0, help="How much delay you want to add between batches?")
def stream(batchsize, fps, delay):
    click.echo(f'Simulating motion stream:\nFPS: {fps}\nBatchSize: {batchsize}')
    click.echo(f'Sending a batch of [{batchsize}] frames every [{(batchsize / fps):.2f}] seconds...')
    t = 0
    seq_q_queue = Queue(maxsize=0)
    seq_q_queue = fill_queue(seq_q_queue, batchsize)

    show_counter = 0
    while True:
        # time.sleep((batchsize / fps) + delay)
        t += batchsize / fps
        # click.echo(f'[{round(t, 2):.2f}] Get next {batchsize} frames from queue.')

        # Handle empty queue
        if seq_q_queue.empty():
            print(f'Stream stopped.')
            print(f'----------')
            print(f'RESULTS')
            print(f'Repititions: {len(repcounter.keyframes)-1}')
            print(f'keyframes: {repcounter.keyframes}')
            print(f'Creating Animated Results Plot...')
            # repcounter.show_animated()
            repcounter.show()
            # click.pause()
            return  # end program
            # refill queue and start again
            # seq_q_queue = fill_queue(Queue(maxsize=0))

        ### Do whatever you want to do here...
        repcounter.append_seq_q(seq_q_queue.get())
        show_counter += batchsize
        # if show_counter >= 300:
        #     show_counter = 0
        #     repcounter.show_animated()
        #     repcounter.show()


if __name__ == '__main__':
    # start_message()
    stream()
