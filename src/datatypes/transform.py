"""
Transformation matrix utilities for 3D graphics.

This module provides functions to convert between different transformation
representations (Pose, Scaling, axis-angle) and 4x4 homogeneous transformation matrices.

All matrices follow OpenGL conventions (column-major when passed to OpenGL).
Transformation order: T × R × S (Translation × Rotation × Scale)
"""

import numpy as np
import math
from typing import Optional


def axis_angle_to_rotation_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """
    Convert axis-angle representation to a 3x3 rotation matrix.
    Uses Rodrigues' rotation formula.

    Args:
        axis_angle: 3x1 or 3x0 numpy array where the direction is the rotation axis
                    and the magnitude is the rotation angle in radians

    Returns:
        3x3 rotation matrix

    Example:
        >>> # 90 degree rotation around Z-axis
        >>> rot = np.array([[0], [0], [np.pi/2]])
        >>> R = axis_angle_to_rotation_matrix(rot)
    """
    # Flatten to 1D array for easier handling
    r = axis_angle.flatten()

    # Calculate angle (magnitude of rotation vector)
    angle = np.linalg.norm(r)

    # If angle is zero, return identity matrix
    if angle < 1e-10:
        return np.eye(3)

    # Normalize to get axis
    axis = r / angle

    # Rodrigues' formula: R = I + sin(θ)K + (1-cos(θ))K²
    # where K is the skew-symmetric cross-product matrix
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])

    # Compute rotation matrix
    R = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)

    return R


def translation_to_matrix(translation: np.ndarray) -> np.ndarray:
    """
    Convert a translation vector to a 4x4 homogeneous transformation matrix.

    Args:
        translation: 3x1 numpy array [x, y, z]

    Returns:
        4x4 translation matrix with translation in the last column

    Example:
        >>> trans = np.array([[1], [2], [3]])
        >>> T = translation_to_matrix(trans)
        >>> # T = [[1, 0, 0, 1],
        >>> #      [0, 1, 0, 2],
        >>> #      [0, 0, 1, 3],
        >>> #      [0, 0, 0, 1]]
    """
    T = np.eye(4)
    T[0:3, 3:4] = translation
    return T


def rotation_to_matrix(rotation: np.ndarray) -> np.ndarray:
    """
    Convert axis-angle rotation to a 4x4 homogeneous transformation matrix.

    Args:
        rotation: 3x1 numpy array (axis-angle representation)

    Returns:
        4x4 transformation matrix with 3x3 rotation in upper-left

    Example:
        >>> rot = np.array([[0], [0], [np.pi]])  # 180° around Z
        >>> R = rotation_to_matrix(rot)
    """
    R_3x3 = axis_angle_to_rotation_matrix(rotation)
    R_4x4 = np.eye(4)
    R_4x4[0:3, 0:3] = R_3x3
    return R_4x4


def scaling_to_matrix(scaling) -> np.ndarray:
    """
    Convert a Scaling object to a 4x4 homogeneous scale matrix.

    Args:
        scaling: Scaling object with x, y, z properties, or None for identity

    Returns:
        4x4 diagonal scale matrix

    Example:
        >>> from src.datatypes.scaling import Scaling
        >>> s = Scaling(x=2.0, y=3.0, z=1.0)
        >>> S = scaling_to_matrix(s)
        >>> # S = [[2, 0, 0, 0],
        >>> #      [0, 3, 0, 0],
        >>> #      [0, 0, 1, 0],
        >>> #      [0, 0, 0, 1]]
    """
    S = np.eye(4)

    if scaling is not None:
        scale_vec = scaling.get_scaling()
        S[0, 0] = scale_vec[0, 0]  # x
        S[1, 1] = scale_vec[1, 0]  # y
        S[2, 2] = scale_vec[2, 0]  # z

    return S


def compose_transform(translation: np.ndarray,
                      rotation: np.ndarray,
                      scaling=None) -> np.ndarray:
    """
    Compose a complete 4x4 transformation matrix from translation, rotation, and scaling.

    Applies transformations in the order: T × R × S (Translation × Rotation × Scale)
    This order ensures:
    - Scaling happens in object space (first)
    - Rotation happens around origin (second)
    - Translation moves to world position (last)

    Args:
        translation: 3x1 numpy array [x, y, z]
        rotation: 3x1 numpy array (axis-angle representation)
        scaling: Scaling object or None for identity scaling

    Returns:
        4x4 transformation matrix

    Example:
        >>> from src.datatypes.pose import Pose
        >>> from src.datatypes.scaling import Scaling
        >>> pose = Pose(
        ...     translation=np.array([[1], [2], [3]]),
        ...     rotation=np.array([[0], [0], [np.pi/2]])
        ... )
        >>> scaling = Scaling(x=2.0, y=1.0, z=1.0)
        >>> M = compose_transform(pose.translation, pose.rotation, scaling)
    """
    # Build individual matrices
    T = translation_to_matrix(translation)
    R = rotation_to_matrix(rotation)
    S = scaling_to_matrix(scaling)

    # Compose in order: T × R × S
    return T @ R @ S


def pose_to_matrix(pose, scaling=None) -> np.ndarray:
    """
    Convert a Pose object (and optional Scaling) to a 4x4 transformation matrix.

    Args:
        pose: Pose object with translation and rotation
        scaling: Optional Scaling object

    Returns:
        4x4 transformation matrix

    Example:
        >>> from src.datatypes.pose import Pose
        >>> pose = Pose(
        ...     translation=np.array([[0], [0], [5]]),
        ...     rotation=np.array([[0], [np.pi/4], [0]])
        ... )
        >>> M = pose_to_matrix(pose)
    """
    return compose_transform(
        pose.get_translation(),
        pose.get_rotation(),
        scaling
    )


def decompose_matrix(matrix: np.ndarray) -> tuple:
    """
    Decompose a 4x4 transformation matrix into translation, rotation, and scale.

    Note: This assumes the matrix was composed using T × R × S order.

    Args:
        matrix: 4x4 transformation matrix

    Returns:
        Tuple of (translation, rotation_matrix, scale) where:
        - translation: 3x1 numpy array
        - rotation_matrix: 3x3 numpy array
        - scale: 3x1 numpy array [sx, sy, sz]

    Warning:
        Converting rotation matrix back to axis-angle is not implemented yet.
        This function returns the rotation matrix directly.
    """
    # Extract translation (last column)
    translation = matrix[0:3, 3:4]

    # Extract the 3x3 upper-left submatrix (rotation + scale)
    upper_left = matrix[0:3, 0:3]

    # Extract scale (length of each column vector)
    scale_x = np.linalg.norm(upper_left[:, 0])
    scale_y = np.linalg.norm(upper_left[:, 1])
    scale_z = np.linalg.norm(upper_left[:, 2])
    scale = np.array([[scale_x], [scale_y], [scale_z]])

    # Extract rotation (normalize columns to remove scale)
    rotation_matrix = np.zeros((3, 3))
    rotation_matrix[:, 0] = upper_left[:, 0] / scale_x
    rotation_matrix[:, 1] = upper_left[:, 1] / scale_y
    rotation_matrix[:, 2] = upper_left[:, 2] / scale_z

    return translation, rotation_matrix, scale
