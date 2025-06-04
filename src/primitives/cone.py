import math
from typing import List, Optional

from src.datatypes.pose import Pose
from src.datatypes.scaling import Scaling
from src.primitives.base_scene_object import MeshPrimitive, SceneObject


class Cone(SceneObject):
    def __init__(self,
                 pose: Pose,
                 scaling: Scaling = Scaling(1.0, 1.0, 1.0),
                 radius: float = 1.0,
                 height: float = 1.0,
                 num_segments: int = 20,
                 name: str = "Cone",
                 parent: Optional[SceneObject] = None):
        super().__init__(scaling=scaling, name=name, pose=pose)
        self.radius = radius
        self.height = height
        self.num_segments = num_segments
        self.parent = parent
        self.children = []
        self.mesh_primitives = list()
        self.vertices = list()
        self.normals = list()
        self.__create_vertices()

    def __create_vertices(self):
        self.mesh_primitives.append(MeshPrimitive.TRIANGLE_FAN)
        self.mesh_primitives.append(MeshPrimitive.TRIANGLE_FAN)
        v_base = list()
        v_side = list()
        n_base = list()
        n_side = list()
        v_base.append([0.0, 0.0, 0.0])
        v_side.append([0.0, 0.0, self.height])
        n_base.append([0.0, 0.0, 1.0])
        n_side.append([0.0, 0.0, 1.0])
        for i in range(self.num_segments + 1):
            angle = 2.0 * math.pi * i / self.num_segments
            x = self.radius * math.cos(angle)
            y = self.radius * math.sin(angle)
            v_base.append([x, y, 0.0])
            v_side.append([x, y, 0.0])

            # Normals for smooth shading
            nx = x
            ny = y
            nz = self.radius / self.height  # Derivative of cone surface
            length = math.sqrt(nx**2 + ny**2 + nz**2)
            n_base.append([nx / length, ny / length, nz / length])
            n_side.append([nx / length, ny / length, nz / length])

        self.vertices.append(v_base)
        self.vertices.append(v_side)
        self.normals.append(v_base)
        self.normals.append(v_side)

    def get_type(self) -> str:
        pass

    def get_parent(self):
        pass

    def set_parent(self, parent):
        pass

    def get_children(self) -> List:
        pass

    def add_child(self, child):
        pass

    def remove_child(self, child):
        pass

    def get_vertices(self) -> List[List[float]]:
        return self.vertices

    def get_normals(self) -> List[List[float]]:
        return self.normals

    def get_mesh_primitives(self) -> List[MeshPrimitive]:
        return self.mesh_primitives
