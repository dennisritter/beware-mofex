"""Contains the code for the sequence model including the scenegraph and angle computation."""
import json
import copy
import networkx as nx
import numpy as np
from scipy.spatial.transform import Rotation
import mofex.transformations as transformations


# Ignore pylint 'Function redefined warning' as Sequence is imported for pyright
# pylint: disable=E0102
class Sequence:
    """Represents a motion sequence.

    Attributes:
        body_parts (dict): A dictionary mapping body part names to position indices in the "positions" attribute array.
        positions (list): The tracked body part positions for each frame.
        timestamps (list): The timestamps for each tracked frame.
        name (str): The name of this sequence.
        scene_graph (networkx.DiGraph): A Directed Graph defining the hierarchy between body parts that will be filled with related data.
    """
    def __init__(self, body_parts: dict, positions: np.ndarray, timestamps: np.ndarray, name: str = 'sequence', scene_graph: nx.DiGraph = None):
        self.name = name
        # Number, order and label of tracked body parts
        # Example: { "head": 0, "neck": 1, ... }
        self.body_parts = body_parts

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
        self.positions = self._get_pelvis_cs_positions(np.array(positions)[zero_frames_filter_list])

        # Timestamps for when the positions have been tracked
        # Example: [<someTimestamp1>, <someTimestamp2>, <someTimestamp3>, ...]
        self.timestamps = np.array(timestamps)[zero_frames_filter_list]

        # A directed graph that defines the hierarchy between human body parts
        self.scene_graph = nx.DiGraph([
            ("pelvis", "torso"),
            ("torso", "neck"),
            ("neck", "head"),
            ("neck", "shoulder_l"),
            ("shoulder_l", "elbow_l"),
            ("elbow_l", "wrist_l"),
            ("neck", "shoulder_r"),
            ("shoulder_r", "elbow_r"),
            ("elbow_r", "wrist_r"),
            ("pelvis", "hip_l"),
            ("hip_l", "knee_l"),
            ("knee_l", "ankle_l"),
            ("pelvis", "hip_r"),
            ("hip_r", "knee_r"),
            ("knee_r", "ankle_r"),
        ]) if scene_graph is None else scene_graph.copy()
        self._fill_scenegraph(self.scene_graph, self.positions)

    def __len__(self) -> int:
        return len(self.timestamps)

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

        # Slice All data lists stored in the scene_graphs nodes and edges
        scene_graph = copy.deepcopy(self.scene_graph)
        for node in scene_graph.nodes:
            for vector_list in scene_graph.nodes[node]['coordinate_system'].keys():
                if vector_list:
                    scene_graph.nodes[node]['coordinate_system'][vector_list] = scene_graph.nodes[node]['coordinate_system'][vector_list][start:stop:step]
            for angle_list in scene_graph.nodes[node]['angles'].keys():
                if angle_list:
                    scene_graph.nodes[node]['angles'][angle_list] = scene_graph.nodes[node]['angles'][angle_list][start:stop:step]
        for edge1, edge2 in scene_graph.edges:
            for data_list in scene_graph[edge1][edge2].keys():
                if data_list:
                    scene_graph[edge1][edge2][data_list] = scene_graph[edge1][edge2][data_list][start:stop:step]

        return Sequence(self.body_parts, self.positions[start:stop:step], self.timestamps[start:stop:step], self.name, scene_graph)

    def _get_pelvis_cs_positions(self, positions):
        """Transforms all points in positions parameter so they are relative to the pelvis. X-Axis = right, Y-Axis = front, Z-Axis = up. """
        transformed_positions = []
        for i, frame in enumerate(positions):
            transformed_positions.append([])
            pelvis_cs = transformations.get_pelvis_coordinate_system(positions[i][self.body_parts["pelvis"]], positions[i][self.body_parts["torso"]],
                                                                     positions[i][self.body_parts["hip_l"]], positions[i][self.body_parts["hip_r"]])
            transformation = transformations.get_cs_projection_transformation(
                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array([pelvis_cs[0][0], pelvis_cs[0][1][0], pelvis_cs[0][1][1], pelvis_cs[0][1][2]]))
            for _, pos in enumerate(frame):
                transformed_positions[i].append((transformation @ np.append(pos, 1))[:3])

        return np.array(transformed_positions)

    def _fill_scenegraph(self, scene_graph, positions):
        # Find Scene Graph Root Node
        root_node = None
        nodes = list(scene_graph.nodes)
        # Find root_node
        for node in nodes:
            predecessors = list(scene_graph.predecessors(node))
            if not predecessors:
                root_node = node
                break

        # Predefine node data attributes to store data for each frame of the sequence
        for node in scene_graph.nodes:
            scene_graph.nodes[node]['coordinate_system'] = {}
            scene_graph.nodes[node]['angles'] = {}

        # Predefine edge data lists to store data for each frame of the sequence
        for node1, node2 in scene_graph.edges:
            scene_graph[node1][node2]['transformation'] = []

        # Start recursive function with root node in our directed scene_graph
        self._calc_scene_graph_transformations_batch(scene_graph, root_node, root_node, positions)

    def _calc_scene_graph_transformations_batch(self, scene_graph, node, root_node, positions):
        n_frames = len(positions)
        successors = list(scene_graph.successors(node))

        # Root Node handling
        if node == root_node:
            # The node with no predecessors is the root node, so add the initial coordinate system vectors
            scene_graph.nodes[node]['coordinate_system']['origin'] = np.zeros((n_frames, 3))
            x_axes = np.empty([n_frames, 3])
            x_axes[:] = np.array([1, 0, 0])
            scene_graph.nodes[node]['coordinate_system']['x_axis'] = x_axes
            y_axes = np.empty([n_frames, 3])
            y_axes[:] = np.array([0, 1, 0])
            scene_graph.nodes[node]['coordinate_system']['y_axis'] = y_axes
            z_axes = np.empty([n_frames, 3])
            z_axes[:] = np.array([0, 0, 1])
            scene_graph.nodes[node]['coordinate_system']['z_axis'] = z_axes

            # Repeat function recursive for each child node of the root node
            for child_node in successors:
                self._calc_scene_graph_transformations_batch(scene_graph, child_node, root_node, positions)
            return

        node_pos = positions[:, self.body_parts[node]]
        predecessors = list(scene_graph.predecessors(node))

        parent_node = predecessors[0]
        parent_pos = positions[:, self.body_parts[parent_node]]
        parent_cs = scene_graph.nodes[parent_node]['coordinate_system']

        translation_mat4x4 = transformations.translation_matrix_4x4_batch(node_pos - parent_pos)
        # If No successors or more than one successor present, add translation only to (parent_node, node) edge.
        # TODO: How can we circumvent to check whether node == "torso" ?
        if len(successors) != 1 or node == "torso":
            scene_graph[parent_node][node]['transformation'] = translation_mat4x4
            scene_graph.nodes[node]['coordinate_system']["origin"] = transformations.mat_mul_batch(translation_mat4x4,
                                                                                                   transformations.v3_to_v4_batch(parent_cs['origin']))[:, :3]
            scene_graph.nodes[node]['coordinate_system']["x_axis"] = parent_cs['x_axis']
            scene_graph.nodes[node]['coordinate_system']["y_axis"] = parent_cs['y_axis']
            scene_graph.nodes[node]['coordinate_system']["z_axis"] = parent_cs['z_axis']
        elif len(successors) == 1:
            child_node = successors[0]
            child_pos = positions[:, self.body_parts[child_node]]

            # --Determine Joint Rotation--
            scene_graph.nodes[node]['angles'] = {}
            # Get direction vector from node to child to determine rotation of nodes' joint
            node_to_child_node = transformations.norm_batch(child_pos - node_pos)
            # Get parent coordinate system Z-axis as reference for nodes' joint rotation..
            # ..Determine 4x4 homogenious rotation matrix to derive joint angles later
            rot_parent_to_node = transformations.get_rotation_batch(parent_cs['z_axis'] * -1, node_to_child_node)

            # Get Euler Sequences to be able to determine medical joint angles
            # NOTE: Scipy from_dcm() function has been renamed to 'as_matrix()' in scipy=1.4.*
            #       lates version for win64 is still 1.3.* ; Consider updating scipy dependency when 1.4.* is available for win64.
            euler_angles_xyz = Rotation.from_dcm(rot_parent_to_node[:, :3, :3]).as_euler('XYZ', degrees=True)
            euler_angles_yxz = Rotation.from_dcm(rot_parent_to_node[:, :3, :3]).as_euler('YXZ', degrees=True)
            euler_angles_zxz = Rotation.from_dcm(rot_parent_to_node[:, :3, :3]).as_euler('ZXZ', degrees=True)

            # Store transformation from parent to current node in corresponding edge
            scene_graph[parent_node][node]['transformation'] = transformations.mat_mul_batch(translation_mat4x4, rot_parent_to_node)
            # Store Rotation Matrix
            scene_graph.nodes[node]['angles']['rotation_matrix'] = np.array(rot_parent_to_node)
            # Store Euler Sequences
            scene_graph.nodes[node]['angles']['euler_xyz'] = euler_angles_xyz
            scene_graph.nodes[node]['angles']['euler_yxz'] = euler_angles_yxz
            scene_graph.nodes[node]['angles']['euler_zxz'] = euler_angles_zxz
            # Store the nodes coordinate system
            scene_graph.nodes[node]['coordinate_system']['origin'] = transformations.mat_mul_batch(translation_mat4x4,
                                                                                                   transformations.v3_to_v4_batch(parent_cs['origin']))[:, :3]
            x_axes = transformations.norm_batch(transformations.mat_mul_batch(rot_parent_to_node[:, :3, :3], parent_cs['x_axis']))
            scene_graph.nodes[node]['coordinate_system']['x_axis'] = x_axes
            y_axes = transformations.norm_batch(transformations.mat_mul_batch(rot_parent_to_node[:, :3, :3], parent_cs['y_axis']))
            scene_graph.nodes[node]['coordinate_system']['y_axis'] = y_axes
            z_axes = transformations.norm_batch(transformations.mat_mul_batch(rot_parent_to_node[:, :3, :3], parent_cs['z_axis']))
            scene_graph.nodes[node]['coordinate_system']['z_axis'] = z_axes

        # Repeat procedure if successors present
        for child_node in successors:
            self._calc_scene_graph_transformations_batch(scene_graph, child_node, root_node, positions)

        return

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
        timestamps = np.array(json_data["timestamps"])

        # reshape positions to 3d array
        positions = np.reshape(positions, (np.shape(positions)[0], int(np.shape(positions)[1] / 3), 3))

        # Center Positions by subtracting the mean of each coordinate
        positions[:, :, 0] -= np.mean(positions[:, :, 0])
        positions[:, :, 1] -= np.mean(positions[:, :, 1])
        positions[:, :, 2] -= np.mean(positions[:, :, 2])

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

        # The target Body Part format
        body_parts = {
            "head": 0,
            "neck": 1,
            "shoulder_l": 2,
            "shoulder_r": 3,
            "elbow_l": 4,
            "elbow_r": 5,
            "wrist_l": 6,
            "wrist_r": 7,
            "torso": 8,
            "pelvis": 9,
            "hip_l": 10,
            "hip_r": 11,
            "knee_l": 12,
            "knee_r": 13,
            "ankle_l": 14,
            "ankle_r": 15,
        }

        # Change body part indices according to the target body part format
        # TODO: Adjust scene graph format to fit optimal kinect azure format
        positions_mka = positions.copy()
        positions[:, 0, :] = positions_mka[:, 26, :]  # "head": 0
        positions[:, 1, :] = positions_mka[:, 3, :]  # "neck": 1
        positions[:, 2, :] = positions_mka[:, 5, :]  # "shoulder_l": 2
        positions[:, 3, :] = positions_mka[:, 12, :]  # "shoulder_r": 3
        positions[:, 4, :] = positions_mka[:, 6, :]  # "elbow_l": 4
        positions[:, 5, :] = positions_mka[:, 13, :]  # "elbow_r": 5
        positions[:, 6, :] = positions_mka[:, 7, :]  # "wrist_l": 6
        positions[:, 7, :] = positions_mka[:, 14, :]  # "wrist_r": 7
        positions[:, 8, :] = positions_mka[:, 1, :]  # "torso": 8 -> SpineNavel
        positions[:, 9, :] = positions_mka[:, 0, :]  # "pelvis": 9
        positions[:, 10, :] = positions_mka[:, 18, :]  # "hip_l": 10
        positions[:, 11, :] = positions_mka[:, 22, :]  # "hip_r": 11
        positions[:, 12, :] = positions_mka[:, 19, :]  # "knee_l": 12
        positions[:, 13, :] = positions_mka[:, 23, :]  # "knee_r": 13
        positions[:, 14, :] = positions_mka[:, 20, :]  # "ankle_l": 14
        positions[:, 15, :] = positions_mka[:, 24, :]  # "ankle_r": 15

        positions = positions[:, :16]

        return cls(body_parts, positions, timestamps, name=name)

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

        self.timestamps = np.concatenate((self.timestamps, sequence.timestamps), axis=0)

        for node in sequence.scene_graph.nodes:
            merge_node_data = sequence.scene_graph.nodes[node]
            node_data = self.scene_graph.nodes[node]
            # Concatenate Coordinate System data
            node_data['coordinate_system']['origin'] = np.concatenate(
                (node_data['coordinate_system']['origin'], merge_node_data['coordinate_system']['origin']))
            node_data['coordinate_system']['x_axis'] = np.concatenate(
                (node_data['coordinate_system']['x_axis'], merge_node_data['coordinate_system']['x_axis']))
            node_data['coordinate_system']['y_axis'] = np.concatenate(
                (node_data['coordinate_system']['y_axis'], merge_node_data['coordinate_system']['y_axis']))
            node_data['coordinate_system']['z_axis'] = np.concatenate(
                (node_data['coordinate_system']['z_axis'], merge_node_data['coordinate_system']['z_axis']))
            # Concatenate angle data
            if 'rotation_matrix' in node_data['angles'].keys():
                node_data['angles']['rotation_matrix'] = np.concatenate((node_data['angles']['rotation_matrix'], merge_node_data['angles']['rotation_matrix']))
                node_data['angles']['euler_xyz'] = np.concatenate((node_data['angles']['euler_xyz'], merge_node_data['angles']['euler_xyz']))
                node_data['angles']['euler_yxz'] = np.concatenate((node_data['angles']['euler_yxz'], merge_node_data['angles']['euler_yxz']))
                node_data['angles']['euler_zxz'] = np.concatenate((node_data['angles']['euler_zxz'], merge_node_data['angles']['euler_zxz']))
        for edge1, edge2 in self.scene_graph.edges:
            merge_edge_data = sequence.scene_graph[edge1][edge2]
            edge_data = self.scene_graph[edge1][edge2]
            edge_data['transformation'] = np.concatenate((edge_data['transformation'], merge_edge_data['transformation']))

        return self

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
