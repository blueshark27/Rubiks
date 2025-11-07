from OpenGL.GL import *
import numpy as np
import math

from src.lights.base_light import BaseLight


class OpenGLLightWrapper:
    def __init__(self, light: BaseLight, index: int = 0):
        self.light = light
        self.index = index

    @staticmethod
    def _rotation_vector_to_matrix(rotation_vec: np.ndarray) -> np.ndarray:
        """
        Convert a rotation vector (axis-angle representation) to a rotation matrix.
        Uses Rodrigues' rotation formula.

        Args:
            rotation_vec: 3x1 numpy array representing rotation (axis * angle)

        Returns:
            3x3 rotation matrix
        """
        # Flatten to 1D array for easier handling
        r = rotation_vec.flatten()

        # Calculate angle (magnitude of rotation vector)
        angle = np.linalg.norm(r)

        # If angle is zero, return identity matrix
        if angle < 1e-10:
            return np.eye(3)

        # Normalize to get axis
        axis = r / angle

        # Rodrigues' formula components
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])

        # Rotation matrix: R = I + sin(θ)K + (1-cos(θ))K²
        R = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)

        return R

    def _get_spotlight_direction(self) -> list:
        """
        Calculate spotlight direction based on pose rotation.
        Base direction is [0, -1, 0] (pointing downward).

        Returns:
            List of 3 floats representing the direction
        """
        base_direction = np.array([0.0, -1.0, 0.0])
        rotation_vec = self.light.get_pose().get_rotation()

        # Convert rotation vector to rotation matrix
        R = self._rotation_vector_to_matrix(rotation_vec)

        # Apply rotation to base direction
        rotated_direction = R @ base_direction

        return rotated_direction.tolist()

    def setup_lighting(self):
        glEnable(eval(f"GL_LIGHT{self.index}"))
        for l in self.light.get_options():
            if type(self.light.get_options()[l]) is list:
                glLightfv(eval(f"GL_LIGHT{self.index}"), eval(f"GL_{l.upper()}"), self.light.get_options()[l])
            else:
                glLightf(eval(f"GL_LIGHT{self.index}"), eval(f"GL_{l.upper()}"), self.light.get_options()[l])

        pose = self.light.get_pose()
        glLightfv(eval(f"GL_LIGHT{self.index}"), GL_POSITION, self.light.pose.get_translation_as_homogeneous())
        if self.light.get_light_type().name == "SPOT":
            direction = self._get_spotlight_direction()
            glLightfv(eval(f"GL_LIGHT{self.index}"), GL_SPOT_DIRECTION, direction)
