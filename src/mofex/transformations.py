"""This module contains 3-D transformation function as well as geometric calculations."""
import math
import numpy as np
import sklearn.preprocessing as preprocessing


def get_angle(vec1, vec2):
    """Returns the angle between vectors vec1 and vec2 (normalized) in degrees

    Args:
        vec1 (np.ndarray): A 3-D vector whose direction defines an angle of zero.
        vec2 (np.ndarray): A 3-D vector whose direction defines the spanned angle to vec1.
    """
    return np.arccos(np.dot(norm(vec1), norm(vec2)))


def get_angle_batch(vec1, vec2):
    """Returns the angle between each vectors of vec1 and vec2 (normalized) in degrees respectively

    Args:
        vec1 (np.ndarray): A list of 3-D vectors whose direction define an angle of zero.
        vec2 (np.ndarray): A list of 3-D vectors whose direction define the spanned angle to vec1.
    """
    vec1 = norm_batch(vec1)
    vec2 = norm_batch(vec2)
    return np.arccos(dot_batch(vec1, vec2))


def get_rotation(vec1, vec2):
    """Returns a homogenious 4x4 transformation matrix without translation vector that describes the rotational transformation from vec1 to vec2

    Args:
        vec1 (np.ndarray): A 3-D vector whose direction defines the starting point of rotation.
        vec2 (np.ndarray): A 3-D vectors whose direction defines the end point of rotation.
    """
    vec1 = norm(vec1)
    vec2 = norm(vec2)
    theta = get_angle(vec1, vec2)
    rotation_axis = get_perpendicular_vector(vec1, vec2)
    rot_mat = rotation_matrix_4x4(rotation_axis, theta)
    return rot_mat


def get_rotation_batch(vec1, vec2):
    """Returns a homogenious 4x4 transformation matrix without translation vector that describes the rotational transformation from vec1 to vec2

    Args:
        vec1 (np.ndarray): A list of 3-D vectors whose direction define the starting point of rotation.
        vec2 (np.ndarray): A list of 3-D vectors whose direction define the end point of rotation.
    """
    vec1 = norm_batch(vec1)
    vec2 = norm_batch(vec2)
    theta = get_angle_batch(vec1, vec2)
    rotation_axis = get_perpendicular_vector_batch(vec1, vec2)
    rot_mat = rotation_matrix_4x4_batch(rotation_axis, theta)
    return rot_mat


def get_perpendicular_vector(vec1, vec2):
    """Returns a vector that is perpendicular to vec1 and vec2

    Args:
        vec1 (np.ndarray): Vector one, which is perpendicular to the returned vector.
        vec2 (np.ndarray): Vector two, which is perpendicular to the returned vector.
    """
    vec1 = norm(vec1)
    vec2 = norm(vec2)

    # If theta 180° (dot product = -1)
    vec1_dot_vec2 = np.dot(vec1, vec2)
    if vec1_dot_vec2 == -1 or vec1_dot_vec2 == 1:
        # Whenever vec1 and vec2 are parallel to each other, we can use an arbitrary vector that is NOT parallel to vec1 and vec2
        # So call this function recursively until a non-parallel vector has been found
        return get_perpendicular_vector(np.random.rand(3), vec2)
    else:
        return norm(np.cross(vec1, vec2))


def get_perpendicular_vector_batch(v_arr1, v_arr2):
    """Returns an array of vectors that are perpendicular to the vectors at corresponding indices.

    Args:
        vec1 (np.ndarray): The first array of vectors that are perpendicular to the respective vector of the returned array.
        vec2 (np.ndarray): The second array of vectors that are perpendicular to the respective vector of the returned array.
    """
    v_arr1 = norm_batch(v_arr1)
    v_arr2 = norm_batch(v_arr2)

    # Matrix multiplication to get dot products of all rows of A and columns of B
    # The diagonal values of the resulting product matrix represent the dot product of A[n] and B[n] in the original arrays
    dot_products = dot_batch(v_arr1, v_arr2)
    # Get indices of vector pairs that are parallel theta 0°/180° (dot product = 1 or -1)
    parallel_vector_indices = np.where((dot_products == -1) | (dot_products == 1))[0]
    for idx in parallel_vector_indices:
        # Replace all vectors of v_arr1 that are parallel to the respective vector in v_arr2 by random unparallel vectors
        v_arr1[idx] = random_unparallel_vector(v_arr2[idx])

    return norm_batch(np.cross(v_arr1, v_arr2))


def random_unparallel_vector(vec1):
    """Returns an unparallel vector to vec1 with same dimension. """
    vec2 = np.random.rand(vec1.shape[0])
    vec1_dot_vec2 = np.dot(vec1, vec2)
    if vec1_dot_vec2 == -1 or vec1_dot_vec2 == 1:
        return random_unparallel_vector(vec1)
    else:
        return vec2


def dot_batch(va1, va2) -> np.ndarray:
    """Returns the dot product for respective vectors in va1 and va2.

    Args:
        va1 (np.ndarray): A list of 3-D vectors.
        va2 (np.ndarray): Another list of 3-D vectors.
    """
    return (va1 @ va2.transpose()).diagonal()


def norm(vec) -> np.ndarray:
    """Normalises the given vector vec and returns it afterwards.

    Args:
        vec (np.ndarray): The vector to normalise.
    """
    vec_norm = np.linalg.norm(vec)
    if vec_norm == 0:
        return np.zeros(3)
    return vec / vec_norm


def norm_batch(v_arr):
    """Normalises the given vectors of v_arr and returns them in an array afterwards.

    Args:
        v_arr (np.ndarray): A numpy array of vectors to normalise.
    """
    return preprocessing.normalize(v_arr, norm='l2')


def mat_mul_batch(arr1, arr2):
    """Performs an element-wise Matrix multiplication for each element in the given arrays.

    The batch-wise matrix multiplication is equivalent to arr1 @ arr2 for each element in arr1 and arr2 respectively.
    The returned np.array contains the resulting 1-D or 2-D np.arrays depending on the dimensions of parameters arr1 and arr2.

    Args:
        arr1 (np.ndarray): An array of vectors or matrices (1-D or 2-D np.arrays)
        arr2 (np.ndarray): An array of vectors or matrices (1-D or 2-D np.arrays)

    """
    if len(arr1) != len(arr2):
        raise ValueError('The number of elements in the first dimension of parameters a and b should be equal.')
    else:
        if arr1.ndim == 3 and arr2.ndim == 2:
            return np.einsum('ijk, ik -> ij', arr1, arr2)
        elif arr1.ndim == 2 and arr2.ndim == 3:
            return np.einsum('ik, ijk -> ik', arr1, arr2)
        elif arr1.ndim == 3 and arr2.ndim == 3:
            return np.einsum('ijk, ikl -> ijl', arr1, arr2)
        else:
            raise ValueError('The parameters dimensions are not supported.\nSuppported dimensions: (2,3), (3,2), (3,3)')


def v3_to_v4(vec):
    """Returns a 3-D position vector with appended homogenious coordinate from the given 3-D vector.

    Args:
        vec (np.ndarray): A 3-D vector.
    """
    return np.append(vec, 1)


def v3_to_v4_batch(vec):
    """Returns a 3-D position vector with appended homogenious coordinate from each given 3-D vector in parameter vec.

    Args:
        vec (np.ndarray): A numpy array of 3-D vectors.
    """
    return np.hstack((vec, np.ones(len(vec)).reshape((len(vec), 1))))


def rotation_matrix_4x4(axis, theta) -> np.ndarray:
    # Source: https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
    """
    Returns the rotation matrix associated with counterclockwise rotation about
    the given axis by theta in radians as 4x4 Transformation Matrix

    Args:
        axis(np.array): The vector to rotate about.
        theta(float): The degrees to rotate about the given axis.

    Returns:
        (np.ndarray) 4x4 rotation matrix representing a rotation about the given axis
    """
    # pylint: disable=invalid-name
    axis = np.asarray(axis)
    axis = norm(axis)
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([
        [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac), 0],
        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab), 0],
        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc, 0],
        [0, 0, 0, 1]
    ])  # yapf: disable
    # pylint: enable=invalid-name


def rotation_matrix_4x4_batch(axis, theta) -> np.ndarray:
    # Source: https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
    """
    Returns an array of rotation matrices each associated with counterclockwise rotation about
    the respective axis by theta in degrees as 4x4 Transformation Matrix

    Args:
        axis(np.array): The vectors to rotate about.
        theta(float): The degrees to rotate about the respective axis.

    Returns:
        (np.ndarray) An array containing 4x4 rotation matrices representing rotations about the given axes
    """
    # pylint: disable=invalid-name
    matrices = np.empty([len(theta), 4, 4])
    axis = norm_batch(np.asarray(axis))
    for i, _ in enumerate(theta):
        a = math.cos(theta[i] / 2.0)
        b, c, d = -axis[i] * math.sin(theta[i] / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        matrices[i] = np.array([
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac), 0],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab), 0],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc, 0],
            [0, 0, 0, 1]
        ])  # yapf: disable
    return matrices
    # pylint: enable=invalid-name


def translation_matrix_4x4(vec) -> np.ndarray:
    """Returns a 4x4 Matrix representing a translation.

    Args:
        vec (np.ndarray): A vector defining the translation.

    Returns:
        (np.ndarray) 4x4 transformation matrix representing a translation as defined by argument v.
    """
    translation_mat = np.array([
        [1.0, 0, 0, 0],
        [0, 1.0, 0, 0],
        [0, 0, 1.0, 0],
        [0, 0, 0, 1.0]
    ])  # yapf: disable
    translation_mat[:3, 3] = vec
    return translation_mat


def translation_matrix_4x4_batch(v_arr) -> np.ndarray:
    """Returns an array of 4x4 Matrices, each representing a translation.

    Args:
        v_arr (np.ndarray): An array of vectors defining translations.

    Returns:
        (np.ndarray) Array of 4x4 transformation matrices, each representing a translation as defined by respective vectors in v_arr.
    """
    transformation_mat = np.empty([len(v_arr), 4, 4])
    ident_mat = [[1.0, 0, 0, 0],
                 [0, 1.0, 0, 0],
                 [0, 0, 1.0, 0],
                 [0, 0, 0, 1.0]]  # yapf: disable
    transformation_mat[:] = ident_mat
    transformation_mat[:, :3, 3] = v_arr[:, :]
    return transformation_mat


def get_local_coordinate_system_direction_vectors(origin, x_direction_bp_pos, y_direction_bp_pos):
    """Returns normalised direction vectors representing the axes directions of the constreucted coordinate system."""
    # New X-Axis from origin to x_direction
    vec_x = x_direction_bp_pos - origin
    if vec_x[0] < 0:
        vec_x = -vec_x
    # New Z-Axis is perpendicular to the origin-y_direction vector and vec_x
    vec_z = get_perpendicular_vector((y_direction_bp_pos - origin), vec_x)
    if vec_z[2] < 0:
        vec_z = -vec_z
    # New Y-Axis is perpendicular to new X-Axis and Z-Axis
    vec_y = get_perpendicular_vector(vec_x, vec_z)
    if vec_y[1] < 0:
        vec_y = -vec_y

    return np.array([norm(vec_x), norm(vec_y), norm(vec_z)])


# TODO: Decide where is the best place for this function. Maybe a new module makes sense so transformations.py holds general functions only
#       and another module/class implements more specific functions like get_pelvis_coordinate_system.
#           Transformation module (keep here)?
#           Sequence class?
#           New module?
def get_pelvis_coordinate_system(pelvis: np.ndarray, torso: np.ndarray, hip_l: np.ndarray, hip_r: np.ndarray):
    """Returns a pelvis coordinate system defined as a tuple containing an origin point and a list of three normalised direction vectors.

    Constructs direction vectors that define the axes directions of the pelvis coordinate system.
    X-Axis-Direction:   Normalised vector whose direction points from hip_l to hip_r. Afterwards, it is translated so that it starts at the pelvis.
    Y-Axis-Direction:   Normalised vector whose direction is determined so that it is perpendicular to the hip_l-hip_r vector and points to the torso.
                        Afterwards, it is translated so that it starts at the pelvis.
    Z-Axis-Direction:   The normalised cross product vector between X-Axis and Y-Axis that starts at the pelvis and results in a right handed Coordinate System.

    Args:
        pelvis (np.ndarray): The X, Y and Z coordinates of the pelvis body part.
        torso (np.ndarray): The X, Y and Z coordinates of the torso body part.
        hip_r (np.ndarray): The X, Y and Z coordinates of the hip_l body part.
        hip_l (np.ndarray): The X, Y and Z coordinates of the hip_r body part.
    """

    # Direction of hip_l -> hip_r is the direction of the X-Axis
    hip_l_hip_r = hip_r - hip_l

    # Orthogonal Projection to determine Y-Axis direction
    vec_a = torso - hip_l
    vec_b = hip_r - hip_l

    scalar = np.dot(vec_a, vec_b) / np.dot(vec_b, vec_b)
    a_on_b = (scalar * vec_b) + hip_l
    vec = torso - a_on_b

    origin = pelvis
    vec_x = norm(hip_l_hip_r)
    vec_z = norm(vec)
    vec_y = get_perpendicular_vector(vec_z, vec_x)

    return [(origin, [vec_x, vec_y, vec_z])]


def get_cs_projection_transformation(from_cs: np.ndarray, target_cs: np.ndarray):
    """Returns a 4x4 transformation to project positions from the from_cs coordinate system to the to_cs coordinate system.

    Args:
        from_cs (np.ndarray): The current coordinate system
            example: [[0,0,0], [1,0,0], [0,1,0], [0,0,1]]
        target_cs (np.ndarray): The target coordinate system
    """
    from_cs_origin, from_cs_x, from_cs_y, _ = from_cs
    target_cs_origin, target_cs_x, target_cs_y, _ = target_cs

    # Get Translation
    transformation_mat = translation_matrix_4x4(from_cs_origin - target_cs_origin)
    # Construct rotation matrix for X-Alignment to rotate about x_rot_axis for the angle theta
    x_rot_axis = get_perpendicular_vector(target_cs_x, from_cs_x)
    theta_x = get_angle(target_cs_x, from_cs_x)
    rot_mat_x = rotation_matrix_4x4(x_rot_axis, theta_x)

    # Use target x-axis direction vector as rotation axis as it must be perpendicular to the y-axis
    y_rot_axis = target_cs_x
    target_cs_y_rx = (rot_mat_x @ np.append(target_cs_y, 1))[:3]
    theta_y = get_angle(target_cs_y_rx, from_cs_y)
    rot_mat_y = rotation_matrix_4x4(norm(y_rot_axis), theta_y)

    # Determine complete transformation matrix
    transformation_mat = rot_mat_x @ rot_mat_y @ transformation_mat
    return transformation_mat


def align_coordinates_to(origin_bp_idx: int, x_direction_bp_idx: int, z_direction_bp_idx: int, positions: np.ndarray):
    """
    Aligns the coordinate system to the given origin point.
    The X-Axis will be in direction of x_direction-origin.
    The Y-Axis will be in direction of y_direction-origin, without crossing the y_direction point but perpendicular to the new X-Axis.
    The Z-Axis will be perpendicular to the XY-Plane.

    Args:
        origin_bp_idx (int): The body part index whose position represents the origin of the coordinate system.
        x_direction_bp_idx (int): The body part index whose position denotes the direction of the x-axis.
        y_direction_bp_idx (int): The body part index whose position denotes the direction of the y-axis.
        positions (np.ndarray): The tracked positions of all body parts for one frame of a motion sequence.
    """

    # Positions of given orientation joints in GCS
    origin = positions[origin_bp_idx]
    x_direction_bp_pos = positions[x_direction_bp_idx]
    z_direction_bp_pos = positions[z_direction_bp_idx]

    # New X-Axis from origin to x_direction
    vec_x = x_direction_bp_pos - origin
    if vec_x[0] < 0:
        vec_x = -vec_x
    # New Z-Axis is perpendicular to the origin-y_direction vector and vec_x
    vec_y = get_perpendicular_vector((z_direction_bp_pos - origin), vec_x)
    if vec_y[1] < 0:
        vec_y = -vec_y

    # New Y-Axis is perpendicular to new X-Axis and Z-Axis
    vec_z = get_perpendicular_vector(vec_x, vec_y)
    if vec_z[2] < 0:
        vec_z = -vec_z

    # Construct translation Matrix to move given origin to zero-position
    translation_mat = translation_matrix_4x4(np.array([0, 0, 0]) - origin)
    # Construct rotation matrix for X-Alignment to rotate about x_rot_axis for the angle theta
    x_rot_axis = get_perpendicular_vector(vec_x, np.array([1, 0, 0]))
    theta_x = get_angle(vec_x, np.array([1, 0, 0]))
    rot_mat_x = rotation_matrix_4x4(x_rot_axis, theta_x)
    # Use new X-Axis axis for y rotation and rot_matotate Y-direction vector to get rotation angle for Y-Alignment
    z_rot_axis = vec_x
    vec_z_rx = np.matmul(rot_mat_x, np.append(vec_z, 1))[:3]
    theta_z = get_angle(vec_z_rx, np.array([0, 1, 0]))
    rot_mat_z = rotation_matrix_4x4(norm(z_rot_axis), theta_z)
    # Transform all positions
    transformed_positions = []
    transformation_mat = np.matmul(translation_mat, rot_mat_x, rot_mat_z)
    for pos in positions:
        pos = np.matmul(transformation_mat, np.append(pos, 1))[:3]
        transformed_positions.append(pos)

    return transformed_positions
