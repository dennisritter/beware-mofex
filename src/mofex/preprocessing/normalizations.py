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
    # positions -= np.mean(positions, axis=2)
    # Center Positions by subtracting the mean of each coordinate
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
    root_positions = np.copy(positions[:, root_idx, :])
    for bp_idx in range(positions.shape[1]):
        positions[:, bp_idx, :] -= root_positions

    return positions


def relative_to_positions(positions: 'np.ndarray', root_positions: 'np.ndarray') -> 'np.ndarray':
    """ Subtracts the mean of all positions from the given positions for each axis respectively. Returns the resulting positions.

        Args: 
            positions (np.ndarray): Positions of a motion sequence of shape  (n_frames, n_body_parts, 3).
            root_position (np.ndarray): An array of 3-D points, whose center represents the root position 0,0,0 (shape: (n_frames, 3))
    """

    for bp_idx in range(positions.shape[1]):
        positions[:, bp_idx, :] -= root_positions

    return positions


# ! Not Working as expected
# def orientation_first_pose_frontal_to_camera(positions: 'np.ndarray', hip_l_idx: int, hip_r_idx: int) -> 'np.ndarray':
#     # TODO: Understand why this is the rotation angle. Source: 10.1007/s11042-017-4859-7 chapter 3.2.2 chapter 3.2.2
#     # Calc roation angle about Z
#     phi = np.arctan((positions[0, hip_l_idx, 0] - positions[0, hip_r_idx, 0]) / (positions[0, hip_l_idx, 1] - positions[0, hip_r_idx, 1]))
#     rz = np.array([[math.cos(phi), -1 * math.sin(phi), 0], [math.sin(phi), math.cos(phi), 0], [0, 0, 1]])
#     # Make array of rotation angles to perform batchwise matrix multiplication
#     rotations = np.full((positions.shape[0], 3, 3), rz)
#     for bp_idx in range(positions.shape[1]):
#         # for frame_idx in range(positions.shape[0]):
#         #     positions[frame_idx, bp_idx] = positions[frame_idx, bp_idx] @ rz
#         positions[:, bp_idx] = positions[:, bp_idx] @ rz
#     return positions


# TODO Rename and PyDoc function!
def orientation(positions: 'np.ndarray', left: 'np.ndarray', right: 'np.ndarray', up: 'np.ndarray') -> 'np.ndarray':
    # --- Normalize horizontal. vec_left_right should lie on x-axis of the Global Coordinate System
    # The vector from given left position to right position
    vec_left_right = right - left
    vec_x_axis = np.array([1.0, 0.0, 0.0])
    vec_z_axis = np.array([0.0, 0.0, 1.0])
    # Get 3x3 rotation matrix that transforms vec_left_right to lie on vec_x_axis
    r = transformations.get_rotation(vec_left_right, vec_x_axis)[:3, :3]

    # --- Normalize vertical. Vector from root to 'up' should lie on XZ-Plane
    # Transform up reference position with r to calculate the rotation for vertical normalization with respect to the horizontal normalization.
    up = up @ r
    # Vector projection up onto vec_x_axis
    vec_up_on_x = vec_x_axis * np.dot(up, vec_x_axis) / np.dot(vec_x_axis, vec_x_axis)
    # Vector from
    vec_up_on_x_to_up = up - vec_up_on_x
    alpha = transformations.get_angle(vec_z_axis, vec_up_on_x_to_up)
    # Define direction of rotation. (alpha is always positive)
    r_alpha = -alpha if up[1] <= 0 else alpha
    rx = np.array([[1, 0, 0], [0, math.cos(-r_alpha), -math.sin(-r_alpha)], [0, math.sin(-r_alpha), math.cos(-r_alpha)]])
    for bp_idx in range(positions.shape[1]):
        positions[:, bp_idx] = positions[:, bp_idx] @ r
        positions[:, bp_idx] = positions[:, bp_idx] @ rx

    return positions