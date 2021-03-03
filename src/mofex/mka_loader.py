"""Contains code for loading the hdmo05 dataset."""
import argparse
import glob
from typing import Tuple

from mana.models.sequence_transforms import SequenceTransforms
from mana.utils.data_operations.loaders.sequence_loader_mka import SequenceLoaderMKA
from mana.utils.math.normalizations import pose_orientation, pose_position
import numpy as np
from torch.utils.data import DataLoader, Dataset


class MKADataset(Dataset):
    """MKADataset reader"""
    def __init__(self, path: str, mode: str = 'classification') -> None:
        """
        Args:
            path (str): Path to Dataset root.
        """
        self.mode = mode
        self.path = path
        self.pathes = sorted(glob.glob(path + '/**/*.json'))
        self.targets = {
            key: idx
            for idx, key in enumerate(
                set([_path.split('/')[-2] for _path in self.pathes]))
        }
        # use mka loader, since we pre-processed mka to have mka form
        self.sequence_transforms = SequenceTransforms(
            SequenceTransforms.mka_to_iisy(body_parts=False))
        self.sequence_transforms.transforms.append(MKAToIISYNorm())
        self.sequence_loader = SequenceLoaderMKA(self.sequence_transforms)

    def __len__(self):
        return len(self.pathes)

    def __getitem__(self, idx):
        if idx > self.__len__():
            raise IndexError

        _input = self.sequence_loader.load(path=(self.pathes[idx]))
        # reshape to (#frames, flatten_bodypart_3d)
        _input = np.reshape(_input.positions, (len(_input.positions), -1))

        # interpolate to 30 frames with 96 values
        # create 30 steps between 0 and max len
        _steps = np.linspace(0, len(_input), num=30)
        # for each value of the coordinates (#96 <- 32joints*3d) interpolate the steps (didn't find 3d interp function..)
        # create an array of the list comp. and transpose it to retrieve original (30,96) shape
        _input = np.array([
            np.interp(_steps, np.arange(len(_input)), _input[:, idx])
            for idx in range(_input.shape[1])
        ]).T

        # repeat array 3 times to re-create 3 channels
        _input = np.repeat(np.expand_dims(_input, 0), 3, axis=0)

        _target = np.array(self.targets[self.pathes[idx].split('/')[-2]])
        return _input, _target


def norm_range(positions):
    positions = 2 * (positions - np.expand_dims(positions.min(axis=1), axis=1)
                     ) / (np.expand_dims(positions.max(axis=1), axis=1) -
                          np.expand_dims(positions.min(axis=1), axis=1)) - 1
    return positions


class MKAToIISYNorm(object):
    def __call__(self, positions: np.ndarray) -> np.ndarray:
        """Returns the given positions after swapping x-values with y-values

        Args:
            positions (np.ndarray): A time series of various 3-D positions
            (ndim = 3) (shape = (n_frames, n_positions, 3))

        Returns:
            np.ndarray: The transformed positions array.
        """
        # translate each frame to pelvis (= 0) position
        positions = pose_position(positions, positions[:, 0, :])
        # rotate hip vector towards x around z
        positions = pose_orientation(positions,
                                     positions[:, 22, :] - positions[:, 18, :],
                                     np.array([1, 0, 0]),
                                     np.array([0, 0, 1]),
                                     origin=positions[:, 0, :])
        # rotate hip vector towards x around y
        positions = pose_orientation(positions,
                                     positions[:, 22, :] - positions[:, 18, :],
                                     np.array([1, 0, 0]),
                                     np.array([0, 1, 0]),
                                     origin=positions[:, 0, :])
        # rotate up (pelvis-spine) vector towards z around x
        positions = pose_orientation(positions,
                                     positions[:, 1, :] - positions[:, 0, :],
                                     np.array([0, 0, 1]),
                                     np.array([1, 0, 0]),
                                     origin=positions[:, 0, :])

        # norm values between -1 and 1
        positions = norm_range(positions)

        return positions


# def pad_collate(batch: List) -> Tuple:
#     """Custom collate method which returns the current batch with padded values.

#     Args:
#         batch (List): The list (batch_size) of items of the unpadded batch.

#     Returns:
#         Tuple: Padded batch elements of the dataset/ tuple + masking arrays.
#             (padded_input, padded_target, padded_input_mask). Masking array
#             contains 1 where a True value is and 0 where padded.
#     """

#     batch_size = len(batch)

#     # split batch
#     _input, _target = zip(*batch)

#     # determine seq lengths and create padded + mask arrays
#     seq_length = [sample.shape[0] for sample in _input]
#     padded_input = torch.zeros(
#         (batch_size, max(seq_length), _input[0].shape[-1]))

#     # fill arrays
#     for sample_idx in range(batch_size):
#         padded_input[sample_idx][:seq_length[sample_idx]] = torch.from_numpy(
#             _input[sample_idx])

#     # parse to torch tensor
#     _input, _target = padded_input, torch.from_numpy(np.asarray(_target))

#     return _input, _target


def mka_loader(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    """Returns the training and validation DataLoader for the MKA Dataset.

    Args:
        path (str): The root path to the MKA Dataset.
        args (argparse.Namespace): COOKIE CLI arguments.
    """
    train_path = args.dataset_path + '/train'
    val_path = args.dataset_path + '/val'

    train_set = MKADataset(train_path)
    val_set = MKADataset(val_path)

    loader_params = {
        'batch_size': args.batch_size,
        'shuffle': args.not_shuffle,
        'num_workers': args.num_workers,
        'pin_memory': not args.preload_gpu,
        # 'collate_fn': pad_collate,
        'drop_last': True,
    }

    train_loader = DataLoader(train_set, **loader_params)
    val_loader = DataLoader(val_set, **loader_params)

    return train_loader, val_loader
