from abc import abstractmethod
from typing import List
from enum import Enum

from src.datatypes.scaling import Scaling
from src.scene_object import SceneObject


class MeshPrimitive(Enum):
    TRIANGLES = 1
    TRIANGLE_STRIP = 2
    TRIANGLE_FAN = 3
    QUAD = 4
    QUAD_STRIP = 5
    POLYGON = 6
    POLYGON_STRIP = 7
    POLYGON_FAN = 8
    POLYLINE = 9
    LINE_LOOP = 10


class BaseSceneObject(SceneObject):

    def __init__(self, scaling: Scaling, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaling = scaling

    def get_scaling(self) -> Scaling:
        return self.scaling

    def set_scaling(self, scaling: Scaling):
        self.scaling = scaling

    @abstractmethod
    def get_vertices(self) -> List[List[float]]:
        pass

    @abstractmethod
    def get_normals(self) -> List[List[float]]:
        pass

    @abstractmethod
    def get_mesh_primitives(self) -> List[MeshPrimitive]:
        pass


    # @abstractmethod
    # def get_bounding_box(self):
    #     pass
    #
    # @abstractmethod
    # def get_bounding_box_pose(self):
    #     pass
    #
    # @abstractmethod
    # def get_bounding_box_dimensions(self):
    #     pass
    #
    # @abstractmethod
    # def get_bounding_box_orientation(self):
    #     pass
    #
    # @abstractmethod
    # def get_bounding_box_center(self):
    #     pass
    #
    # @abstractmethod
    # def get_bounding_box_corners(self):
    #     pass
    #
    # @abstractmethod
    # def get_bounding_box_axes(self):
    #     pass
    #
    # @abstractmethod
    # def get_bounding_box_extents(self):
    #     pass
    #
    # @abstractmethod
    # def get_bounding_box_volume(self):
    #     pass
    #
    # @abstractmethod
    # def get_bounding_box_surface_area(self):
    #     pass
    #
    # @abstractmethod
    # def get_bounding_box_min(self):
    #     pass
    #
    # @abstractmethod
    # def get_bounding_box_max(self):
    #     pass
    #
    # @abstractmethod
    # def get_bounding_box_center_of_mass(self):
    #     pass
    #
    # @abstractmethod
    # def get_bounding_box_inertia_tensor(self):
    #     pass
    #
    # @abstractmethod
    # def get_bounding_box_inertia_tensor_origin(self):
    #     pass
    #
    # @abstractmethod
    # def get_bounding_box_inertia_tensor_orientation(self):
    #     pass
    #
    # @abstractmethod
    # def get_bounding_box_inertia_tensor_principal_axes(self):
    #     pass
    #
    # @abstractmethod
    # def get_bounding_box_inertia_tensor_principal_values(self):
    #     pass
    #
    # @abstractmethod
    # def get_bounding_box_inertia_tensor_principal_axes_orientation(self):
    #     pass
