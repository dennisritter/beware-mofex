"""Contains the code for the sequence model including the scenegraph and angle computation."""
import json
import copy
import numpy as np
import cv2
from ezc3d import c3d
from mofex.preprocessing.skeleton_visualizer import SkeletonVisualizer
import mofex.preprocessing.transformations as transformations
import mofex.preprocessing.normalizations as norm


# Ignore pylint 'Function redefined warning' as Sequence is imported for pyright
# pylint: disable=E0102
class Sequence:
    """Represents a motion sequence.

    Attributes:
        positions (list): The tracked body part positions for each frame.
        name (str): The name of this sequence.
    """
    def __init__(self, positions: np.ndarray, name: str = 'sequence', desc: str = None):
        self.name = name
        # A Boolean mask list to exclude all frames, where all positions are 0.0
        zero_frames_filter_list = self._filter_zero_frames(positions)
        # Defines positions of each bodypart
        # 1. Dimension = Time
        # 2. Dimension = Bodypart
        # 3. Dimension = x, y, z
        # Example: [
        #           [[f1_bp1_x, f1_bp1_x, f1_bp1_x], [f1_bp2_x, f1_bp2_x, f1_bp2_x], ...],
        #           [[f2_bp1_x, f2_bp1_x, f2_bp1_x], [f2_bp2_x, f2_bp2_x, f2_bp2_x], ...],
        #           ...
        #          ]
        # shape: (num_body_parts, num_keypoints, xyz)
        self.positions = np.array(positions)[zero_frames_filter_list]
        # Description of the sequence. (used for class identification)
        self.desc = desc

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, item) -> 'Sequence':
        """Returns the sub-sequence item. You can either specifiy one element by index or use numpy-like slicing.

        Args:
            item (int/slice): Defines a particular frame or slice from all frames of this sequence.

        Raises NotImplementedError if index is given as tuple.
        Raises TypeError if item is not of type int or slice.
        """

        if isinstance(item, slice):
            if item.start is None and item.stop is None and item.step is None:
                # Return a Deepcopy to improve copy performance (sequence[:])
                return copy.deepcopy(self)
            start, stop, step = item.indices(len(self))
        elif isinstance(item, int):
            start, stop, step = item, item + 1, 1
        elif isinstance(item, tuple):
            raise NotImplementedError("Tuple as index")
        else:
            raise TypeError(f"Invalid argument type: {type(item)}")

        return Sequence(self.positions[start:stop:step], self.name, self.desc)

    def _filter_zero_frames(self, positions: np.ndarray) -> list:
        """Returns a filter mask list to filter frames where all positions equal 0.0.

        Checks whether all coordinates for a frame are 0
            True -> keep this frame
            False -> remove this frame

        Args:
            positions (np.ndarray): The positions to filter "Zero-Position-Frames" from

        Returns:
            (list<boolean>): The filter list.
        """
        return [len(pos) != len(pos[np.all(pos == 0)]) for pos in positions]

    @classmethod
    def from_mir_file(cls, path: str, name: str = 'Sequence', desc: str = None) -> 'Sequence':
        """Loads an sequence .json file in Mocap Intel RealSense format and returns an Sequence object.

        Args:
            path (str): Path to the json file

        Returns:
            Sequence: a new Sequence instance from the given input.
        """
        with open(path, 'r') as sequence_file:
            return Sequence.from_mir_json(sequence_file.read(), name)

    @classmethod
    def from_mir_json(cls, json_str: str, name: str = 'Sequence', desc: str = None) -> 'Sequence':
        """Loads an sequence from a json string in Mocap Intel RealSense format and returns an Sequence object.

        Args:
            json_str (str): The json string.

        Returns:
            Sequence: a new Sequence instance from the given input.
        """
        # load, parse file from json and return class
        json_data = json.loads(json_str)
        positions = np.array(json_data["positions"])

        # reshape positions to 3d array
        positions = np.reshape(positions, (np.shape(positions)[0], int(np.shape(positions)[1] / 3), 3))

        # Center Positions by subtracting the mean of each coordinate
        positions[:, :, 0] -= np.mean(positions[:, :, 0])
        positions[:, :, 1] -= np.mean(positions[:, :, 1])
        positions[:, :, 2] -= np.mean(positions[:, :, 2])

        # Adjust MIR data to our target Coordinate System
        # X_mocap = Left    ->  X_hma = Right   -->     Flip X-Axis
        # Y_mocap = Up      ->  Y_hma = Front   -->     Switch Y and Z; Flip (new) Y-Axis
        # Z_mocap = Back    ->  Z_hma = Up      -->     Switch Y and Z

        # Switch Y and Z axis.
        # In MIR Y points up and Z to the back -> We want Z to point up and Y to the front,
        y_positions_mocap = positions[:, :, 1].copy()
        z_positions_mocap = positions[:, :, 2].copy()
        positions[:, :, 1] = z_positions_mocap
        positions[:, :, 2] = y_positions_mocap
        # MIR coordinate system is left handed -> flip x-axis to adjust data for right handed coordinate system
        positions[:, :, 0] *= -1
        # Flip Y-Axis
        # MIR Z-Axis (our Y-Axis now) points "behind" the trainee, but we want it to point "forward"
        positions[:, :, 1] *= -1

        # # Change body part indices according to the target body part format
        # positions_mocap = positions.copy()
        # positions[:, 0, :] = positions_mocap[:, 15, :]  # "head": 0
        # positions[:, 1, :] = positions_mocap[:, 3, :]  # "neck": 1
        # positions[:, 2, :] = positions_mocap[:, 2, :]  # "shoulder_l": 2
        # positions[:, 3, :] = positions_mocap[:, 14, :]  # "shoulder_r": 3
        # positions[:, 4, :] = positions_mocap[:, 1, :]  # "elbow_l": 4
        # positions[:, 5, :] = positions_mocap[:, 13, :]  # "elbow_r": 5
        # positions[:, 6, :] = positions_mocap[:, 0, :]  # "wrist_l": 6
        # positions[:, 7, :] = positions_mocap[:, 12, :]  # "wrist_r": 7
        # positions[:, 8, :] = positions_mocap[:, 4, :]  # "torso": 8
        # positions[:, 9, :] = positions_mocap[:, 5, :]  # "pelvis": 9
        # positions[:, 10, :] = positions_mocap[:, 8, :]  # "hip_l": 10
        # positions[:, 11, :] = positions_mocap[:, 11, :]  # "hip_r": 11
        # positions[:, 12, :] = positions_mocap[:, 7, :]  # "knee_l": 12
        # positions[:, 13, :] = positions_mocap[:, 10, :]  # "knee_r": 13
        # positions[:, 14, :] = positions_mocap[:, 6, :]  # "ankle_l": 14
        # positions[:, 15, :] = positions_mocap[:, 9, :]  # "ankle_r": 15

        return cls(positions, name=name, desc=desc)

    @classmethod
    def from_mka_file(cls, path: str, name: str = 'Sequence') -> 'Sequence':
        """Loads an sequence .json file in Mocap Kinect Azure format and returns an Sequence object.

        Args:
            path (str): Path to the json file

        Returns:
            Sequence: a new Sequence instance from the given input.
        """
        with open(path, 'r') as sequence_file:
            return Sequence.from_mka_json(sequence_file.read(), name)

    @classmethod
    def from_mka_json(cls, json_str: str, name: str = 'Sequence') -> 'Sequence':
        """Loads an sequence from a json string in Mocap Intel RealSense format and returns an Sequence object.

        Args:
            json_str (str): The json string.

        Returns:
            Sequence: a new Sequence instance from the given input.
        """
        # load, parse file from json and return class
        json_data = json.loads(json_str)
        positions = np.array(json_data["positions"])

        # reshape positions to 3d array
        positions = np.reshape(positions, (np.shape(positions)[0], int(np.shape(positions)[1] / 3), 3))

        # MKA X points left -> HMA X points right
        # MKA Y points down -> HMA Y points front
        # MKA Z points backwards -> HMA Z points up
        # Switch Y/Z
        y_positions_mka = positions[:, :, 1].copy()
        z_positions_mka = positions[:, :, 2].copy()
        positions[:, :, 1] = z_positions_mka
        positions[:, :, 2] = y_positions_mka
        # Flip X
        positions[:, :, 0] *= -1
        # Flip Y
        positions[:, :, 1] *= -1
        # Flip Z
        positions[:, :, 2] *= -1

        # NOTE: Uncomment if you want to standardize the used joint positions as in HMA
        # Change body part indices according to the target body part format
        # positions_mka = positions.copy()
        # positions[:, 0, :] = positions_mka[:, 26, :]  # "head": 0
        # positions[:, 1, :] = positions_mka[:, 3, :]  # "neck": 1
        # positions[:, 2, :] = positions_mka[:, 5, :]  # "shoulder_l": 2
        # positions[:, 3, :] = positions_mka[:, 12, :]  # "shoulder_r": 3
        # positions[:, 4, :] = positions_mka[:, 6, :]  # "elbow_l": 4
        # positions[:, 5, :] = positions_mka[:, 13, :]  # "elbow_r": 5
        # positions[:, 6, :] = positions_mka[:, 7, :]  # "wrist_l": 6
        # positions[:, 7, :] = positions_mka[:, 14, :]  # "wrist_r": 7
        # positions[:, 8, :] = positions_mka[:, 1, :]  # "torso": 8 -> SpineNavel
        # positions[:, 9, :] = positions_mka[:, 0, :]  # "pelvis": 9
        # positions[:, 10, :] = positions_mka[:, 18, :]  # "hip_l": 10
        # positions[:, 11, :] = positions_mka[:, 22, :]  # "hip_r": 11
        # positions[:, 12, :] = positions_mka[:, 19, :]  # "knee_l": 12
        # positions[:, 13, :] = positions_mka[:, 23, :]  # "knee_r": 13
        # positions[:, 14, :] = positions_mka[:, 20, :]  # "ankle_l": 14
        # positions[:, 15, :] = positions_mka[:, 24, :]  # "ankle_r": 15
        # positions = positions[:, :16]

        # Normalize Skeleton
        # Center Positions
        # positions = norm.center_positions(positions)
        # All Positions relative to root (pelvis)
        # positions = norm.relative_to_root(positions, root_idx=0)
        # positions = norm.orientation_first_pose_frontal_to_camera(positions, hip_l_idx=10, hip_r_idx=11)

        return cls(positions, name=name, desc=desc)

    @classmethod
    def from_hdm05_c3d_file(cls, path: str, name: str = 'Sequence', desc: str = None) -> 'Sequence':
        """Loads the Positions of the a .c3d file and returns an Sequence object.

        Args:
            path (str): Path to the c3d file

        Returns:
            Sequence: a new Sequence instance from the given input.
        """
        c3d_object = c3d(str(path))
        positions = c3d_object['data']['points']
        positions = positions.swapaxes(0, 2)[:, :, :3]
        return cls(positions, name=name, desc=desc)

    def merge(self, sequence) -> 'Sequence':
        """Returns the merged two sequences.

        Raises ValueError if either the body_parts, the poseformat or the body_parts do not match!
        """
        if self.body_parts != sequence.body_parts:
            raise ValueError('body_parts of both sequences do not match!')

        # Copy the given sequence to not change it implicitly
        sequence = sequence[:]
        # concatenate positions, timestamps and angles
        self.positions = np.concatenate((self.positions, sequence.positions), axis=0)

        return self

    def to_motionimg(
            self,
            output_size: (int, int) = (256, 256),
            minmax_pos_x: (int, int) = (-1000, 1000),
            minmax_pos_y: (int, int) = (-1000, 1000),
            minmax_pos_z: (int, int) = (-1000, 1000),
            show_img=False,
            show_skeleton=False,
    ) -> np.ndarray:
        # TODO: Calculate 'smart' minmax_pos values
        """ Returns a Motion Image, that represents this sequences' positions.

            Creates an Image from 3-D position data of motion sequences.
            Rows represent a body part (or some arbitrary position instance).
            Columns represent a frame of the sequence.

            Args:
                output_size (int, int): The size of the output image in pixels (height, width). Default=(200,200)
                minmax_pos_x (int, int): The minimum and maximum x-position values. Mapped to color range (0, 255).
                minmax_pos_y (int, int): The minimum and maximum y-position values. Mapped to color range (0, 255).
                minmax_pos_z (int, int): The minimum and maximum z-position values. Mapped to color range (0, 255).
        """
        # Create Image container
        img = np.zeros((len(self.positions[0, :]), len(self.positions), 3), dtype='uint8')
        # 1. Map (min_pos, max_pos) range to (0, 255) Color range.
        # 2. Swap Axes of and frames(0) body parts(1) so rows represent body parts and cols represent frames.
        img[:, :, 0] = np.interp(self.positions[:, :, 0], [minmax_pos_x[0], minmax_pos_x[1]], [0, 255]).swapaxes(0, 1)
        img[:, :, 1] = np.interp(self.positions[:, :, 1], [minmax_pos_y[0], minmax_pos_y[1]], [0, 255]).swapaxes(0, 1)
        img[:, :, 2] = np.interp(self.positions[:, :, 2], [minmax_pos_z[0], minmax_pos_z[1]], [0, 255]).swapaxes(0, 1)
        img = cv2.resize(img, output_size)

        if show_img:
            cv2.imshow(self.name, img)
            print(f"Showing motion image from [{self.name}]. Press any key to close the image and continue.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if show_skeleton:
            sv = SkeletonVisualizer(self)
            sv.show()
        return img

    def norm_center_positions(self):
        self.positions = norm.center_positions(self.positions)
        return

    def norm_relative_to_position_idx(self, position_idx):
        self.positions = norm.relative_to_root(self.positions, root_idx=position_idx)
        return

    def norm_relative_to_positions(self, positions):
        self.positions = norm.relative_to_positions(self.positions, root_positions=positions)
        return

    def norm_orientation_first_pose_frontal_to_camera(self, hip_l_idx, hip_r_idx):
        self.positions = norm.orientation_first_pose_frontal_to_camera(self.positions, hip_l_idx, hip_r_idx)
        return