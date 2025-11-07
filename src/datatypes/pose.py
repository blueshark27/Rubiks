import numpy as np
from pydantic import BaseModel, field_validator
from numpydantic import NDArray
from typing import Annotated, Optional

class Pose(BaseModel):
    translation: Annotated[NDArray, 3]
    rotation: Annotated[NDArray, 3]

    @field_validator('translation', 'rotation')
    @classmethod
    def check_3d_arrays(cls, v):
        if not isinstance(v, np.ndarray) or v.shape != (3, 1):
            raise ValueError("Each element muss be a three dimensional numpy array!")
        return v

    def get_translation(self) -> NDArray:
        return self.translation

    def get_translation_as_homogeneous(self) -> NDArray:
        translation = self.get_translation()
        return np.array([[translation[0, 0]], [translation[1, 0]], [translation[2, 0]], [1.0]])

    def get_rotation(self) -> NDArray:
        return self.rotation

    def set_translation(self, translation: NDArray) -> None:
        if translation.shape != (3, 1):
            raise ValueError("Translation must be a three dimensional numpy array!")
        self.translation = translation

    def set_rotation(self, rotation: NDArray) -> None:
        if rotation.shape != (3, 1):
            raise ValueError("Rotation must be a three dimensional numpy array!")
        self.rotation = rotation

    def t(self) -> NDArray:
        return self.get_translation()

    def r(self) -> NDArray:
        return self.get_rotation()

    def to_pose_quat(self) -> 'PoseQuat':
        """
        Convert this axis-angle pose to a quaternion-based pose.

        Returns:
            PoseQuat with equivalent rotation
        """
        from src.datatypes.quaternion import Quaternion
        quat = Quaternion.from_axis_angle_vector(self.rotation)
        return PoseQuat(translation=self.translation, quaternion=quat)


class PoseQuat(BaseModel):
    """
    Pose representation using quaternions for rotation.

    This is preferred over axis-angle Pose for:
    - Animation and interpolation (SLERP)
    - Avoiding gimbal lock
    - Smoother rotations
    - Easier composition of rotations

    Attributes:
        translation: 3x1 position vector
        quaternion: Quaternion representing rotation
    """
    translation: Annotated[NDArray, 3]
    quaternion: 'Quaternion'

    class Config:
        arbitrary_types_allowed = True

    @field_validator('translation')
    @classmethod
    def check_translation(cls, v):
        if not isinstance(v, np.ndarray) or v.shape != (3, 1):
            raise ValueError("Translation must be a 3x1 numpy array!")
        return v

    @classmethod
    def from_translation_quaternion(cls, translation: np.ndarray, quaternion: 'Quaternion') -> 'PoseQuat':
        """
        Create pose from translation and quaternion.

        Args:
            translation: 3x1 translation vector
            quaternion: Quaternion for rotation

        Returns:
            PoseQuat instance
        """
        return cls(translation=translation, quaternion=quaternion)

    @classmethod
    def from_translation_axis_angle(cls, translation: np.ndarray, axis: np.ndarray, angle: float) -> 'PoseQuat':
        """
        Create pose from translation and axis-angle rotation.

        Args:
            translation: 3x1 translation vector
            axis: 3x1 or 3-element rotation axis
            angle: Rotation angle in radians

        Returns:
            PoseQuat instance
        """
        from src.datatypes.quaternion import Quaternion
        quat = Quaternion.from_axis_angle(axis, angle)
        return cls(translation=translation, quaternion=quat)

    @classmethod
    def from_translation_euler(cls, translation: np.ndarray, roll: float, pitch: float, yaw: float) -> 'PoseQuat':
        """
        Create pose from translation and Euler angles.

        Args:
            translation: 3x1 translation vector
            roll: Rotation around X-axis (radians)
            pitch: Rotation around Y-axis (radians)
            yaw: Rotation around Z-axis (radians)

        Returns:
            PoseQuat instance
        """
        from src.datatypes.quaternion import Quaternion
        quat = Quaternion.from_euler(roll, pitch, yaw)
        return cls(translation=translation, quaternion=quat)

    @classmethod
    def identity(cls) -> 'PoseQuat':
        """
        Create identity pose (no translation or rotation).

        Returns:
            PoseQuat at origin with no rotation
        """
        from src.datatypes.quaternion import Quaternion
        return cls(
            translation=np.zeros((3, 1)),
            quaternion=Quaternion.identity()
        )

    def get_translation(self) -> NDArray:
        """Get translation vector"""
        return self.translation

    def get_translation_as_homogeneous(self) -> NDArray:
        """Get translation as homogeneous 4x1 vector"""
        translation = self.get_translation()
        return np.array([[translation[0, 0]], [translation[1, 0]], [translation[2, 0]], [1.0]])

    def get_quaternion(self) -> 'Quaternion':
        """Get rotation quaternion"""
        return self.quaternion

    def set_translation(self, translation: NDArray) -> None:
        """Set translation vector"""
        if translation.shape != (3, 1):
            raise ValueError("Translation must be a 3x1 numpy array!")
        self.translation = translation

    def set_quaternion(self, quaternion: 'Quaternion') -> None:
        """Set rotation quaternion"""
        self.quaternion = quaternion

    def t(self) -> NDArray:
        """Shorthand for get_translation()"""
        return self.get_translation()

    def q(self) -> 'Quaternion':
        """Shorthand for get_quaternion()"""
        return self.get_quaternion()

    def to_pose(self) -> Pose:
        """
        Convert this quaternion pose to an axis-angle pose.

        Returns:
            Pose with equivalent rotation as axis-angle
        """
        axis_angle = self.quaternion.to_axis_angle_vector()
        return Pose(translation=self.translation, rotation=axis_angle)

    def to_matrix(self) -> np.ndarray:
        """
        Convert pose to 4x4 transformation matrix.

        Returns:
            4x4 homogeneous transformation matrix
        """
        # Get rotation matrix from quaternion
        R = self.quaternion.to_rotation_matrix()

        # Build 4x4 transformation matrix
        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3:4] = self.translation

        return T

    def interpolate(self, other: 'PoseQuat', t: float, method: str = 'slerp') -> 'PoseQuat':
        """
        Interpolate between this pose and another.

        Args:
            other: Target pose
            t: Interpolation parameter [0, 1] where 0 = self, 1 = other
            method: Interpolation method ('slerp' or 'lerp')

        Returns:
            Interpolated pose
        """
        from src.datatypes.quaternion import slerp, lerp

        # Linear interpolation of translation
        interp_translation = self.translation * (1.0 - t) + other.translation * t

        # Quaternion interpolation
        if method == 'slerp':
            interp_quat = slerp(self.quaternion, other.quaternion, t)
        elif method == 'lerp':
            interp_quat = lerp(self.quaternion, other.quaternion, t)
        else:
            raise ValueError(f"Unknown interpolation method: {method}. Use 'slerp' or 'lerp'.")

        return PoseQuat(translation=interp_translation, quaternion=interp_quat)

    def __repr__(self) -> str:
        t = self.translation.flatten()
        return f"PoseQuat(t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}], q={self.quaternion})"