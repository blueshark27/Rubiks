from abc import abstractmethod
from typing import List, Optional
from enum import Enum
import numpy as np

from src.datatypes.scaling import Scaling
from src.datatypes.material import Material
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

    def __init__(self, scaling: Scaling, material: Optional[Material] = None,
                 color: Optional[List[float]] = None, *args, **kwargs):
        """
        Initialize base scene object with scaling and material.

        Args:
            scaling: Scaling information
            material: Material for the object (optional)
            color: Convenience parameter - creates a Material from this color (optional)
                   If both material and color are provided, material takes precedence
            *args, **kwargs: Arguments passed to parent SceneObject
        """
        super().__init__(*args, **kwargs)
        self.scaling = scaling

        # Handle material/color initialization
        if material is not None:
            self.material = material
        elif color is not None:
            self.material = Material.from_color(color)
        else:
            self.material = Material.default()

    def get_scaling(self) -> Scaling:
        return self.scaling

    def set_scaling(self, scaling: Scaling):
        self.scaling = scaling
        self.mark_transform_dirty()

    def get_material(self) -> Material:
        """Get the material for this object"""
        return self.material

    def set_material(self, material: Material):
        """
        Set the material for this object.

        Args:
            material: New material to apply
        """
        self.material = material

    def set_color(self, color: List[float]):
        """
        Convenience method to set the object's color.

        This updates the material's diffuse and ambient colors.

        Args:
            color: RGBA color [r, g, b, a] where each component is in [0, 1]
        """
        self.material.set_color(color)

    def get_local_transform(self) -> np.ndarray:
        """
        Get the local transformation matrix (pose + scaling).
        Overrides SceneObject to include scaling.
        Uses caching to avoid unnecessary recalculation.

        Returns:
            4x4 local transformation matrix
        """
        if self._local_transform_dirty:
            from src.datatypes import transform
            self._local_transform_cache = transform.pose_to_matrix(
                self.pose,
                self.scaling
            )
            self._local_transform_dirty = False

        return self._local_transform_cache

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
