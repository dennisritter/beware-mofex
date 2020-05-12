# import numpy as np
# import cv2
# import json
# import os
# from pathlib import Path
import time
import click

# TODO: 1. Load Ground Truth sequence GT (Type of motion we are looking for. e.g. a squat)
# TODO: 2. Load Long Sequence Qs (group of the searched type of motions. e.g. 10 squats)
# TODO: 3. Cut Qs into parts q of <batchsize>
# TODO: 4. Make motion image -> Feature vector from q
# TODO: 5. Compare FVq to FVgt and temp save distance
# TODO: 6. Repeat with next n batches
# TODO: 7. If distance decreases n times, we found end/start frame


@click.command()
@click.option("--batchsize", default=10, help="How many frames do you want to send in each package?")
@click.option("--fps", default=30, help="How many frames per second do you want to stream?")
def stream(batchsize, fps):
    click.echo(f'Simulating motion stream:\nFPS: {fps}\nBatchSize: {batchsize}')
    click.echo(f'Sending a batch of [{batchsize}] frames every [{batchsize / fps}] seconds...')
    t = 0
    while True:
        time.sleep(batchsize / fps)
        t += batchsize / fps
        click.echo(f'[{round(t, 2):.2f}] Sent {batchsize} frames.')


if __name__ == '__main__':
    # start_message()
    stream()
