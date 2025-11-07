from OpenGL.GL import *
import numpy as np

from src.primitives.base_scene_object import BaseSceneObject

class OpenGLPrimitivesWrapper:
    def __init__(self, object: BaseSceneObject):
        self.object = object

    def draw(self):
        # Get world transformation matrix
        world_matrix = self.object.get_world_transform()

        # Push current matrix onto stack
        glPushMatrix()

        # Apply world transformation
        # OpenGL expects column-major, NumPy is row-major, so transpose
        glMultMatrixf(world_matrix.T.astype(np.float32).flatten())

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