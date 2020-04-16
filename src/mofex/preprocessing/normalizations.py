"""This module contains 3-D transformation function as well as geometric calculations."""
import math
import numpy as np
import mofex.preprocessing.transformations as transformations
import sklearn.preprocessing as preprocessing


def center_positions(positions: 'np.ndarray') -> 'np.ndarray':
    """ Subtracts the mean of all positions from the given positions for each axis respectively. Returns the resulting centered positions.

        Args: 
            positions (np.ndarray): Positions of a motion sequence of shape  (n_frames, n_body_parts, 3).
    """
    normalized_positions = np.zeros(positions.shape)
    positions[:, :, 0] -= np.mean(positions[:, :, 0])
    positions[:, :, 1] -= np.mean(positions[:, :, 1])
    positions[:, :, 2] -= np.mean(positions[:, :, 2])
    return positions


def relative_to_root(positions: 'np.ndarray', root_idx: int) -> 'np.ndarray':
    """ Subtracts the mean of all positions from the given positions for each axis respectively. Returns the resulting positions.

        Args: 
            positions (np.ndarray): Positions of a motion sequence of shape  (n_frames, n_body_parts, 3).
            root_idx (int): The body part(dim=1) index of the skeletons' root joint.
    """
    # Store copy of slice to prevent subtracting the normalized root positions (which is 0,0,0 after the subtraction)
    original_root_position = np.copy(positions[0, root_idx])
    for bp_idx in range(positions.shape[1]):
        positions[:, bp_idx] -= original_root_position
    return positions


# def orientation_frontal_to_camera_all_poses(positions: 'np.ndarray', hip_l_idx: int, hip_r_idx: int) -> 'np.ndarray':
#   phis = ((positions[:, hip_l_idx, 1] - positions[:, hip_l_idx, 1]) / (positions[:, hip_r_idx, 0] - positions[:, hip_r_idx, 0]))


def orientation_first_pose_frontal_to_camera(positions: 'np.ndarray', hip_l_idx: int, hip_r_idx: int) -> 'np.ndarray':
    # TODO: Understand why this is the rotation angle. Source: 10.1007/s11042-017-4859-7 chapter 3.2.2
    # Calc roation angle
    phi = np.arctan((positions[0, hip_l_idx, 1] - positions[0, hip_r_idx, 1]) / (positions[0, hip_l_idx, 0] - positions[0, hip_r_idx, 0]))
    # Make array of rotation angles to perform batchwise matrix multiplication
    r = np.array([[math.cos(phi), 0, math.sin(phi)], [0, 1, 0], [math.sin(phi) * -1, 0, math.cos(phi)]])
    rotations = np.full((positions.shape[0], 3, 3), r)
    for bp_idx in range(positions.shape[1]):
        positions[:, bp_idx] = transformations.mat_mul_batch(positions[:, bp_idx], rotations)
    return positions