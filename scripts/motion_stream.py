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

# ! Root directory of motion images. Make sure it includes 'train' and 'val' directory, which then include a directory for each present label in the dataset.
# ! Structure template: <in_path>/train/<class_name>/motion_img1.png
dataset_name = 'hdm05-122_90-10'
# CNN Model name -> model_dataset-numclasses_train-val-ratio
model_name = 'resnet101_hdm05-122_90-10'
# The CNN Model for Feature Vector generation
model = model_loader.load_trained_model(model_name=model_name,
                                        remove_last_layer=True,
                                        state_dict_path=f'./data/trained_models/{dataset_name}/{model_name}_e25.pt')
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
filename_asf = '*.asf'
filename_amc = '*.amc'
src_root = './data/sequences/hdm05-122/amc/squat3Reps/'
seqs = load_seqs_asf_amc_hdm05(src_root, filename_asf, filename_amc)
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
asf_path = './data/sequences/hdm05-122/amc/squat1Reps/HDM_bd.asf'
amc_path = './data/sequences/hdm05-122/amc/squat1Reps/HDM_bd_squat1Reps_001_120.amc'
seq_gt = Sequence.from_hdm05_asf_amc_files(asf_path, amc_path)
repcounter = RepCounter(seq_gt=seq_gt[0:8], subseq_len=8, savgol_win=21, model=model, feature_size=feature_size, preprocess=preprocess)


@click.command()
@click.option("--batchsize", default=10, help="How many frames do you want to send in each package?")
@click.option("--fps", default=120, help="How many frames per second do you want to stream?")
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
        click.echo(f'[{round(t, 2):.2f}] Get next {batchsize} frames from queue.')

        # Handle empty queue
        if seq_q_queue.empty():
            print(f'Stream stopped.')
            print(f'----------')
            print(f'RESULTS')
            print(f'Repititions: {len(repcounter.keyframes)-1}')
            print(f'keyframes: {repcounter.keyframes}')
            print(f'Creating Animated Results Plot...')
            repcounter.show_animated()
            click.pause()
            # refill queue and start again
            seq_q_queue = fill_queue(Queue(maxsize=0))

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
