# 1. Install Mana -> pip install src/mana-develop-788bfe8e8c7c429d4f6616326f3c800a99b111be.zip
# 2. Choose sequences to test
# 3. Finish implementation for dataloading
# 4. ..

import numpy as np

from mana.models.sequence_transforms import SequenceTransforms
from mana.utils.data_operations.loaders.sequence_loader_mka import SequenceLoaderMKA

from mofex.mka_loader import MKAToIISYNorm

import torch
import mofex.model_loader as model_loader
from mana.utils.data_operations.loaders.sequence_loader_mka import SequenceLoaderMKA
from mana.models.sequence_transforms import SequenceTransforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def norm_sequence(sequence):
    # reshape to (#frames, flatten_bodypart_3d)
    sequence.positions = np.reshape(sequence.positions,
                                    (len(sequence.positions), -1))

    # interpolate to 30 frames with 96 values
    # create 30 steps between 0 and max len
    _steps = np.linspace(0, len(sequence), num=30)
    # for each value of the coordinates (#96 <- 32joints*3d) interpolate the steps (didn't find 3d interp function..)
    # create an array of the list comp. and transpose it to retrieve original (30,96) shape
    sequence.positions = np.array([
        np.interp(_steps, np.arange(len(sequence.positions)),
                  sequence.positions[:, idx])
        for idx in range(sequence.positions.shape[1])
    ]).T

    # repeat array 3 times to re-create 3 channels
    sequence.positions = np.repeat(np.expand_dims(sequence.positions, 0),
                                   3,
                                   axis=0)
    return sequence


def evaluate_sequence(model, path):

    # setup sequence loader
    sequence_transforms = SequenceTransforms(
        SequenceTransforms.mka_to_iisy(body_parts=False))
    sequence_transforms.transforms.append(MKAToIISYNorm())
    sequence_loader = SequenceLoaderMKA(sequence_transforms)

    # preload whole sequence
    sequence = sequence_loader.load(path=path)

    # start_idx = 0
    model.eval()
    with torch.no_grad():

        for idx in range(len(sequence)):
            # TODO: which range to process from sequence?
            #       _sequence = sequence[start_idx:idx + 1]

            # norm sequence
            _sequence = norm_sequence(_sequence)

            positions = torch.tensor(
                _sequence.positions).unsqueeze(0).float().to(device)

            outputs = model(positions)

            _, preds = torch.max(outputs, 1)

            pred = preds.item()
            if pred == 0:
                print("rep")
                # TODO: update start idx for sequence window?


# TODO: your model pathes
model_list = [
    'data/resnet101_mka-beware-1.1_cookie-3.0_sgd_e50.pt',
]

for model_path in model_list:
    model = model_loader.load_trained_model(
        model_name="resnet101_hdm05-122_90-10_cookie_downstream",
        state_dict_path=model_path)
    model = model.to(device)

    # TODO: your sequence pathes
    for path in sorted(
            glob.glob('data/test/complex/05-03-2021-12-17-26/*.json')):
        print(f'------------------------')
        print(f'Start evaluating: {model_path.split("/")[-1]}')
        print(f'  using: {path}')
        print(f'------------------------')
        evaluate_sequence(model, path)
        print(f'\n')