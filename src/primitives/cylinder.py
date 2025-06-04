import math
from typing import List, Optional

from src.datatypes.pose import Pose
from src.primitives.base_scene_object import MeshPrimitive, SceneObject


class Cylinder(SceneObject):
    def __init__(self,
                 pose: Pose,
                 radius: float = 1.0,
                 height: float = 1.0,
                 num_segments: int = 20,
                 name: str = "Cylinder",
                 parent: Optional[SceneObject] = None):
        super().__init__(name=name, pose=pose)
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

        half_height = self.height / 2.0
        angle_step = 2 * math.pi / self.num_segments

        # Mantel (seitliche Fläche)
        self.mesh_primitives.append(MeshPrimitive.QUAD_STRIP)
        self.normals.append(list())
        self.vertices.append(list())
        for i in range(self.num_segments + 1):
            angle = i * angle_step
            x = self.radius * math.cos(angle)
            y = self.radius * math.sin(angle)
            nx, ny = math.cos(angle), math.sin(angle)  # Normale = Radiusrichtung
            self.normals[-1].append([nx, ny, 0.0])
            self.vertices[-1].append([x, y, -half_height])
            self.vertices[-1].append([x, y, half_height])

        # Deckel oben
        self.mesh_primitives.append(MeshPrimitive.TRIANGLE_FAN)
        self.normals.append(list())
        self.vertices.append(list())
        self.normals[-1].append([0.0, 0.0, 1.0])
        self.vertices[-1].append([0.0, 0.0, half_height])
        for i in range(self.num_segments + 1):
            angle = i * angle_step
            x = self.radius * math.cos(angle)
            y = self.radius * math.sin(angle)
            self.normals[-1].append([x, y, 1.0])
            self.vertices[-1].append([x, y, half_height])

        # Boden unten
        self.mesh_primitives.append(MeshPrimitive.TRIANGLE_FAN)
        self.normals.append(list())
        self.vertices.append(list())
        self.normals[-1].append([0.0, 0.0, -1.0])
        self.vertices[-1].append([0.0, 0.0, -half_height])
        for i in range(self.num_segments + 1):
            angle = -i * angle_step  # Im Uhrzeigersinn für Normale nach unten
            x = self.radius * math.cos(angle)
            y = self.radius * math.sin(angle)
            self.normals[-1].append([x, y, -1.0])
            self.vertices[-1].append([x, y, -half_height])

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
