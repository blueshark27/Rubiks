"""
Quaternion implementation for 3D rotations.

Quaternions provide several advantages over axis-angle and Euler angles:
- No gimbal lock
- Smooth interpolation (SLERP)
- Efficient composition
- Numerically stable

A quaternion is represented as: q = w + xi + yj + zk
where w is the scalar part and (x, y, z) is the vector part.
"""

import numpy as np
import math
from typing import Tuple, Optional
from pydantic import BaseModel, field_validator
from numpydantic import NDArray, Shape


class Quaternion(BaseModel):
    """
    Quaternion class for representing 3D rotations.

    Stored as [w, x, y, z] where:
    - w: scalar/real part
    - (x, y, z): vector/imaginary part

    Unit quaternions (||q|| = 1) represent rotations.
    """

    data: NDArray[Shape["4, 1"], float]

    class Config:
        arbitrary_types_allowed = True

    @field_validator('data')
    def validate_shape(cls, v):
        """Ensure quaternion is 4x1"""
        if v.shape != (4, 1):
            raise ValueError(f"Quaternion must be 4x1, got {v.shape}")
        return v

    def __init__(self, w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0, **kwargs):
        """
        Create a quaternion.

        Args:
            w: Scalar part (default: 1.0 for identity)
            x: i component
            y: j component
            z: k component
        """
        if 'data' not in kwargs:
            data = np.array([[w], [x], [y], [z]], dtype=float)
            super().__init__(data=data)
        else:
            super().__init__(**kwargs)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Quaternion':
        """Create quaternion from numpy array [w, x, y, z]"""
        if arr.shape == (4,):
            arr = arr.reshape(4, 1)
        return cls(data=arr)

    @classmethod
    def identity(cls) -> 'Quaternion':
        """Create identity quaternion (no rotation)"""
        return cls(w=1.0, x=0.0, y=0.0, z=0.0)

    @classmethod
    def from_axis_angle(cls, axis: np.ndarray, angle: float) -> 'Quaternion':
        """
        Create quaternion from axis-angle representation.

        Args:
            axis: 3x1 or 3-element array (will be normalized)
            angle: Rotation angle in radians

        Returns:
            Quaternion representing the rotation
        """
        axis = axis.flatten()
        axis_norm = np.linalg.norm(axis)

        if axis_norm < 1e-10:
            # Zero rotation
            return cls.identity()

        axis = axis / axis_norm
        half_angle = angle / 2.0
        sin_half = math.sin(half_angle)

        return cls(
            w=math.cos(half_angle),
            x=axis[0] * sin_half,
            y=axis[1] * sin_half,
            z=axis[2] * sin_half
        )

    @classmethod
    def from_axis_angle_vector(cls, axis_angle: np.ndarray) -> 'Quaternion':
        """
        Create quaternion from axis-angle vector (axis * angle).

        Args:
            axis_angle: 3x1 array where direction is axis and magnitude is angle

        Returns:
            Quaternion representing the rotation
        """
        axis_angle = axis_angle.flatten()
        angle = np.linalg.norm(axis_angle)

        if angle < 1e-10:
            return cls.identity()

        axis = axis_angle / angle
        return cls.from_axis_angle(axis, angle)

    @classmethod
    def from_euler(cls, roll: float, pitch: float, yaw: float) -> 'Quaternion':
        """
        Create quaternion from Euler angles (XYZ convention).

        Args:
            roll: Rotation around X-axis (radians)
            pitch: Rotation around Y-axis (radians)
            yaw: Rotation around Z-axis (radians)

        Returns:
            Quaternion representing the rotation
        """
        cr = math.cos(roll / 2.0)
        sr = math.sin(roll / 2.0)
        cp = math.cos(pitch / 2.0)
        sp = math.sin(pitch / 2.0)
        cy = math.cos(yaw / 2.0)
        sy = math.sin(yaw / 2.0)

        return cls(
            w=cr * cp * cy + sr * sp * sy,
            x=sr * cp * cy - cr * sp * sy,
            y=cr * sp * cy + sr * cp * sy,
            z=cr * cp * sy - sr * sp * cy
        )

    def to_axis_angle(self) -> Tuple[np.ndarray, float]:
        """
        Convert quaternion to axis-angle representation.

        Returns:
            Tuple of (axis, angle) where axis is 3x1 and angle is in radians
        """
        w, x, y, z = self.w, self.x, self.y, self.z

        # Normalize quaternion
        norm = math.sqrt(w*w + x*x + y*y + z*z)
        if norm < 1e-10:
            return np.array([[0], [0], [1]]), 0.0

        w, x, y, z = w/norm, x/norm, y/norm, z/norm

        # Calculate angle
        angle = 2.0 * math.acos(np.clip(w, -1.0, 1.0))

        # Calculate axis
        sin_half = math.sin(angle / 2.0)
        if abs(sin_half) < 1e-10:
            # No rotation or 360° rotation
            axis = np.array([[0], [0], [1]])
        else:
            axis = np.array([[x], [y], [z]]) / sin_half

        return axis, angle

    def to_axis_angle_vector(self) -> np.ndarray:
        """
        Convert quaternion to axis-angle vector (axis * angle).

        Returns:
            3x1 array where direction is axis and magnitude is angle
        """
        axis, angle = self.to_axis_angle()
        return axis * angle

    def to_rotation_matrix(self) -> np.ndarray:
        """
        Convert quaternion to 3x3 rotation matrix.

        Returns:
            3x3 rotation matrix
        """
        w, x, y, z = self.w, self.x, self.y, self.z

        # Normalize quaternion
        norm = math.sqrt(w*w + x*x + y*y + z*z)
        if norm < 1e-10:
            return np.eye(3)

        w, x, y, z = w/norm, x/norm, y/norm, z/norm

        # Convert to rotation matrix
        return np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
            [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
            [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])

    def to_euler(self) -> Tuple[float, float, float]:
        """
        Convert quaternion to Euler angles (XYZ convention).

        Returns:
            Tuple of (roll, pitch, yaw) in radians
        """
        w, x, y, z = self.w, self.x, self.y, self.z

        # Roll (X-axis)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (Y-axis)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90° if out of range
        else:
            pitch = math.asin(sinp)

        # Yaw (Z-axis)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    @property
    def w(self) -> float:
        """Scalar part"""
        return float(self.data[0, 0])

    @property
    def x(self) -> float:
        """i component"""
        return float(self.data[1, 0])

    @property
    def y(self) -> float:
        """j component"""
        return float(self.data[2, 0])

    @property
    def z(self) -> float:
        """k component"""
        return float(self.data[3, 0])

    def normalize(self) -> 'Quaternion':
        """
        Return normalized quaternion (unit quaternion).

        Returns:
            Normalized quaternion
        """
        norm = math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        if norm < 1e-10:
            return Quaternion.identity()
        return Quaternion.from_array(self.data / norm)

    def conjugate(self) -> 'Quaternion':
        """
        Return conjugate quaternion (inverse for unit quaternions).

        Returns:
            Conjugate quaternion q* = w - xi - yj - zk
        """
        return Quaternion(w=self.w, x=-self.x, y=-self.y, z=-self.z)

    def inverse(self) -> 'Quaternion':
        """
        Return inverse quaternion.

        Returns:
            Inverse quaternion q^-1
        """
        norm_sq = self.w**2 + self.x**2 + self.y**2 + self.z**2
        if norm_sq < 1e-10:
            return Quaternion.identity()

        conj = self.conjugate()
        return Quaternion.from_array(conj.data / norm_sq)

    def __mul__(self, other: 'Quaternion') -> 'Quaternion':
        """
        Multiply two quaternions (Hamilton product).

        Args:
            other: Another quaternion

        Returns:
            Product quaternion self * other
        """
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z

        return Quaternion(
            w=w1*w2 - x1*x2 - y1*y2 - z1*z2,
            x=w1*x2 + x1*w2 + y1*z2 - z1*y2,
            y=w1*y2 - x1*z2 + y1*w2 + z1*x2,
            z=w1*z2 + x1*y2 - y1*x2 + z1*w2
        )

    def rotate_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Rotate a 3D vector using this quaternion.

        Args:
            vector: 3x1 or 3-element array

        Returns:
            Rotated 3x1 vector
        """
        vector = vector.flatten()

        # Create quaternion from vector
        v_quat = Quaternion(w=0.0, x=vector[0], y=vector[1], z=vector[2])

        # Rotate: v' = q * v * q*
        result = self * v_quat * self.conjugate()

        return np.array([[result.x], [result.y], [result.z]])

    def dot(self, other: 'Quaternion') -> float:
        """
        Dot product with another quaternion.

        Args:
            other: Another quaternion

        Returns:
            Dot product
        """
        return self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z

    def __repr__(self) -> str:
        return f"Quaternion(w={self.w:.4f}, x={self.x:.4f}, y={self.y:.4f}, z={self.z:.4f})"


def slerp(q1: Quaternion, q2: Quaternion, t: float) -> Quaternion:
    """
    Spherical Linear Interpolation between two quaternions.

    SLERP provides smooth interpolation between rotations, maintaining
    constant angular velocity.

    Args:
        q1: Start quaternion
        q2: End quaternion
        t: Interpolation parameter [0, 1]

    Returns:
        Interpolated quaternion
    """
    # Clamp t to [0, 1]
    t = np.clip(t, 0.0, 1.0)

    # Compute dot product
    dot = q1.dot(q2)

    # If dot < 0, negate q2 to take shorter path
    if dot < 0.0:
        q2 = Quaternion(w=-q2.w, x=-q2.x, y=-q2.y, z=-q2.z)
        dot = -dot

    # If quaternions are very close, use linear interpolation
    if dot > 0.9995:
        result = Quaternion.from_array(
            q1.data * (1.0 - t) + q2.data * t
        )
        return result.normalize()

    # Calculate angle between quaternions
    theta_0 = math.acos(np.clip(dot, -1.0, 1.0))
    theta = theta_0 * t

    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)

    s1 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s2 = sin_theta / sin_theta_0

    result = Quaternion.from_array(q1.data * s1 + q2.data * s2)
    return result.normalize()


def lerp(q1: Quaternion, q2: Quaternion, t: float) -> Quaternion:
    """
    Linear interpolation between two quaternions.

    Faster than SLERP but doesn't maintain constant angular velocity.
    Good for small angular differences.

    Args:
        q1: Start quaternion
        q2: End quaternion
        t: Interpolation parameter [0, 1]

    Returns:
        Interpolated quaternion (normalized)
    """
    t = np.clip(t, 0.0, 1.0)

    # If dot product is negative, negate q2 for shorter path
    if q1.dot(q2) < 0:
        q2 = Quaternion(w=-q2.w, x=-q2.x, y=-q2.y, z=-q2.z)

    result = Quaternion.from_array(q1.data * (1.0 - t) + q2.data * t)
    return result.normalize()
