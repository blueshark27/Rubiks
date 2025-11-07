import math
from typing import List, Optional

from src.datatypes.pose import Pose
from src.datatypes.scaling import Scaling
from src.primitives.base_scene_object import MeshPrimitive, BaseSceneObject

# This is a Python implementation of an icosphere, which is a type of sphere made up of triangles.
# https://www.songho.ca/opengl/gl_sphere.html#icosphere

class Sphere(BaseSceneObject):
    def __init__(self,
                 pose: Pose,
                 radius: float = 1.0,
                 subdivision: int = 3,
                 scaling: Optional[Scaling] = None,
                 name: str = "Sphere",
                 parent: Optional['Sphere'] = None):
        if scaling is None:
            scaling = Scaling(x=1.0, y=1.0, z=1.0)
        super().__init__(name=name, pose=pose, scaling=scaling, parent=parent)
        self.radius = radius
        self.subdivision = subdivision
        self.mesh_primitives = list()
        self.__vertices = list()
        self.__triangles = list()
        self.vertices = list()
        self.triangles = list()
        self.normals = list()
        self.__create_icosahedron()
        self.__subdivide()

    @staticmethod
    def __normalize(v: List[float]) -> List[float]:
        length = math.sqrt(sum([coord ** 2 for coord in v]))
        return [coord / length for coord in v]

    @staticmethod
    def __midpoint(v1: List[float], v2:  List[float]) -> List[float]:
        return Sphere.__normalize([(v1[i] + v2[i]) / 2.0 for i in range(3)])

    def __create_icosahedron(self):
        self.mesh_primitives.append(MeshPrimitive.TRIANGLES)

        t = (1.0 + math.sqrt(5.0)) / 2.0

        base_vertices = [
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
        ]
        self.__vertices = [Sphere.__normalize(v) for v in base_vertices]

        base_faces = [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ]
        self.__triangles.extend(base_faces)

    def __subdivide(self):
        for _ in range(self.subdivision):
            new_triangles = []
            midpoint_cache = {}

            def cached_midpoint(i1, i2):
                key = tuple(sorted((i1, i2)))
                if key in midpoint_cache:
                    return midpoint_cache[key]
                v1, v2 = self.__vertices[i1], self.__vertices[i2]
                mid = Sphere.__midpoint(v1, v2)
                self.__vertices.append(mid)
                idx = len(self.__vertices) - 1
                midpoint_cache[key] = idx
                return idx

            for tri in self.__triangles:
                a, b, c = tri
                ab = cached_midpoint(a, b)
                bc = cached_midpoint(b, c)
                ca = cached_midpoint(c, a)

                new_triangles += [
                    [a, ab, ca],
                    [b, bc, ab],
                    [c, ca, bc],
                    [ab, bc, ca]
                ]
            self.__triangles = new_triangles

        self.normals.append(list())
        self.vertices.append(list())
        for tri in self.__triangles:
            for idx in tri:
                normal = self.__vertices[idx]
                vertex = [coord * self.radius for coord in normal]
                self.normals[0].append(normal)
                self.vertices[0].append(vertex)


    def get_type(self) -> str:
        pass

    def get_vertices(self) -> List[List[float]]:
        return self.vertices

    def get_normals(self) -> List[List[float]]:
        return self.normals

    def get_mesh_primitives(self) -> List[MeshPrimitive]:
        return self.mesh_primitives