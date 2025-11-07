from OpenGL.GL import *
import numpy as np
from typing import Optional, List

from src.primitives.base_scene_object import BaseSceneObject


class OpenGLPrimitivesWrapper:
    def __init__(self, object: BaseSceneObject, color: Optional[List[float]] = None):
        """
        Initialize OpenGL wrapper for a primitive object.

        Args:
            object: The BaseSceneObject to wrap
            color: Optional color override [r, g, b, a] (for backward compatibility)
                   If provided, overrides the object's material color
        """
        self.object = object
        self.color_override = color

    def draw(self):
        # Get world transformation matrix
        world_matrix = self.object.get_world_transform()

        # Push current matrix onto stack
        glPushMatrix()

        # Apply world transformation
        # OpenGL expects column-major, NumPy is row-major, so transpose
        glMultMatrixf(world_matrix.T.astype(np.float32).flatten())

        # Apply material properties
        self._apply_material()

        # Now draw all primitives
        p = self.object.get_mesh_primitives()
        v = self.object.get_vertices()
        n = self.object.get_normals()
        assert len(p) == len(v), f"len(p): {len(p)}, len(n): {len(v)}"
        assert len(p) == len(n), f"len(p): {len(p)}, len(n): {len(n)}"

        for i, mesh in enumerate(self.object.get_mesh_primitives()):
            if mesh.name in ["TRIANGLES", "TRIANGLE_STRIP", "TRIANGLE_FAN"]:
                assert len(v[i]) == len(n[i]), f"len(v): {len(v[i])}, len(n): {len(n[i])}"
                glBegin(eval(f"GL_{mesh.name}"))
                for vertex, normal in zip(self.object.get_vertices()[i], self.object.get_normals()[i]):
                    glNormal3f(*normal)
                    glVertex3f(*vertex)
                glEnd()
            elif mesh.name in ["QUADS", "QUAD_STRIP"]:
                assert len(v[i]) == 2*len(n[i]), f"len(v): {len(v[i])}, (2x) len(n): {2*len(n[i])}"
                glBegin(eval(f"GL_{mesh.name}"))
                for vertex in self.object.get_vertices()[i]:
                    glVertex3f(*vertex)
                glEnd()
            else:
                raise ValueError(f"Unsupported mesh primitive type: {mesh.name}")

        # Pop matrix to restore previous state
        glPopMatrix()

    def _apply_material(self):
        """Apply material properties to OpenGL state"""
        material = self.object.get_material()

        # If color override is provided, use it for diffuse and ambient
        if self.color_override is not None:
            ambient = [c * 0.2 for c in self.color_override]
            diffuse = self.color_override
            specular = material.get_specular()
            shininess = material.get_shininess()
        else:
            # Use material properties from the object
            ambient = material.get_ambient()
            diffuse = material.get_diffuse()
            specular = material.get_specular()
            shininess = material.get_shininess()

        # Apply material properties
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, ambient)
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specular)
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shininess)