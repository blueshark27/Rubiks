"""
Camera abstraction for view and projection matrices.

Provides high-level camera creation and manipulation without direct OpenGL calls.
"""
import numpy as np
from typing import Optional, List
from OpenGL.GL import *
from OpenGL.GLU import *


class Camera:
    """
    Camera class for managing view and projection transformations.

    Supports both perspective and orthographic projections.
    """

    def __init__(self, projection_type: str = "perspective"):
        """
        Initialize camera.

        Args:
            projection_type: Either "perspective" or "orthographic"
        """
        self.projection_type = projection_type

        # View parameters
        self.position = np.array([0.0, 0.0, 5.0])
        self.look_at_point = np.array([0.0, 0.0, 0.0])
        self.up_vector = np.array([0.0, 1.0, 0.0])

        # Perspective parameters
        self.fov = 45.0
        self.aspect_ratio = 4.0/3.0
        self.near = 0.1
        self.far = 100.0

        # Orthographic parameters
        self.left = -10.0
        self.right = 10.0
        self.bottom = -10.0
        self.top = 10.0
        self.ortho_near = -10.0
        self.ortho_far = 10.0

    @classmethod
    def perspective(cls,
                    fov: float = 45.0,
                    aspect_ratio: float = 4.0/3.0,
                    near: float = 0.1,
                    far: float = 100.0,
                    position: Optional[List[float]] = None,
                    look_at: Optional[List[float]] = None,
                    up: Optional[List[float]] = None) -> 'Camera':
        """
        Create a perspective camera.

        Args:
            fov: Field of view in degrees
            aspect_ratio: Width/height ratio
            near: Near clipping plane
            far: Far clipping plane
            position: Camera position [x, y, z]
            look_at: Point to look at [x, y, z]
            up: Up vector [x, y, z]

        Returns:
            Camera configured for perspective projection
        """
        camera = cls(projection_type="perspective")
        camera.fov = fov
        camera.aspect_ratio = aspect_ratio
        camera.near = near
        camera.far = far

        if position is not None:
            camera.position = np.array(position)
        if look_at is not None:
            camera.look_at_point = np.array(look_at)
        if up is not None:
            camera.up_vector = np.array(up)

        return camera

    @classmethod
    def orthographic(cls,
                     left: float = -10.0,
                     right: float = 10.0,
                     bottom: float = -10.0,
                     top: float = 10.0,
                     near: float = -10.0,
                     far: float = 10.0,
                     position: Optional[List[float]] = None,
                     look_at: Optional[List[float]] = None,
                     up: Optional[List[float]] = None) -> 'Camera':
        """
        Create an orthographic camera.

        Args:
            left: Left clipping plane
            right: Right clipping plane
            bottom: Bottom clipping plane
            top: Top clipping plane
            near: Near clipping plane
            far: Far clipping plane
            position: Camera position [x, y, z]
            look_at: Point to look at [x, y, z]
            up: Up vector [x, y, z]

        Returns:
            Camera configured for orthographic projection
        """
        camera = cls(projection_type="orthographic")
        camera.left = left
        camera.right = right
        camera.bottom = bottom
        camera.top = top
        camera.ortho_near = near
        camera.ortho_far = far

        if position is not None:
            camera.position = np.array(position)
        if look_at is not None:
            camera.look_at_point = np.array(look_at)
        if up is not None:
            camera.up_vector = np.array(up)

        return camera

    def set_position(self, position: List[float]):
        """Set camera position."""
        self.position = np.array(position)

    def set_look_at(self, look_at: List[float]):
        """Set point camera is looking at."""
        self.look_at_point = np.array(look_at)

    def set_up_vector(self, up: List[float]):
        """Set camera up vector."""
        self.up_vector = np.array(up)

    def set_aspect_ratio(self, width: int, height: int):
        """Update aspect ratio based on window dimensions."""
        self.aspect_ratio = width / height if height > 0 else 1.0

    def apply_projection(self):
        """Apply projection matrix to OpenGL state."""
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        if self.projection_type == "perspective":
            gluPerspective(self.fov, self.aspect_ratio, self.near, self.far)
        elif self.projection_type == "orthographic":
            glOrtho(self.left, self.right, self.bottom, self.top,
                   self.ortho_near, self.ortho_far)
        else:
            raise ValueError(f"Unknown projection type: {self.projection_type}")

    def apply_view(self):
        """Apply view matrix to OpenGL state."""
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(
            self.position[0], self.position[1], self.position[2],
            self.look_at_point[0], self.look_at_point[1], self.look_at_point[2],
            self.up_vector[0], self.up_vector[1], self.up_vector[2]
        )

    def apply(self):
        """Apply both projection and view transformations."""
        self.apply_projection()
        self.apply_view()
