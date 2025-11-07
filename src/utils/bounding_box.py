"""
Bounding box computation for scene objects.

This module provides axis-aligned bounding box (AABB) support for:
- Computing bounding boxes for primitive shapes
- Transforming bounding boxes to world space
- Computing hierarchical bounding boxes
- Collision detection and intersection tests
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class BoundingBox:
    """
    Axis-Aligned Bounding Box (AABB).

    Represented by minimum and maximum corners in 3D space.

    Attributes:
        min: Minimum corner (3x1 array)
        max: Maximum corner (3x1 array)
    """
    min: np.ndarray  # 3x1
    max: np.ndarray  # 3x1

    def __post_init__(self):
        """Validate bounding box"""
        if self.min.shape != (3, 1):
            raise ValueError(f"min must be 3x1, got {self.min.shape}")
        if self.max.shape != (3, 1):
            raise ValueError(f"max must be 3x1, got {self.max.shape}")

        # Ensure min <= max for each dimension (skip for empty boxes with inf values)
        if not (np.any(np.isinf(self.min)) or np.any(np.isinf(self.max))):
            if np.any(self.min > self.max):
                raise ValueError(f"min must be <= max. Got min={self.min.flatten()}, max={self.max.flatten()}")

    @classmethod
    def from_points(cls, points: np.ndarray) -> 'BoundingBox':
        """
        Create bounding box from a set of points.

        Args:
            points: Nx3 or 3xN array of points

        Returns:
            BoundingBox enclosing all points
        """
        if points.shape[0] == 3:
            # 3xN format
            min_corner = np.min(points, axis=1).reshape(3, 1)
            max_corner = np.max(points, axis=1).reshape(3, 1)
        else:
            # Nx3 format
            min_corner = np.min(points, axis=0).reshape(3, 1)
            max_corner = np.max(points, axis=0).reshape(3, 1)

        return cls(min=min_corner, max=max_corner)

    @classmethod
    def from_center_size(cls, center: np.ndarray, size: np.ndarray) -> 'BoundingBox':
        """
        Create bounding box from center and size.

        Args:
            center: 3x1 center point
            size: 3x1 size (width, height, depth)

        Returns:
            BoundingBox centered at center with given size
        """
        half_size = size / 2.0
        return cls(min=center - half_size, max=center + half_size)

    @classmethod
    def empty(cls) -> 'BoundingBox':
        """
        Create an empty bounding box.

        Returns:
            BoundingBox with inverted min/max (useful for merging)
        """
        return cls(
            min=np.array([[np.inf], [np.inf], [np.inf]]),
            max=np.array([[-np.inf], [-np.inf], [-np.inf]])
        )

    def get_center(self) -> np.ndarray:
        """Get center point of bounding box"""
        return (self.min + self.max) / 2.0

    def get_size(self) -> np.ndarray:
        """Get size (extents) of bounding box"""
        return self.max - self.min

    def get_volume(self) -> float:
        """Get volume of bounding box"""
        size = self.get_size()
        return float(size[0, 0] * size[1, 0] * size[2, 0])

    def get_surface_area(self) -> float:
        """Get surface area of bounding box"""
        size = self.get_size().flatten()
        return 2.0 * (size[0] * size[1] + size[1] * size[2] + size[2] * size[0])

    def get_corners(self) -> np.ndarray:
        """
        Get all 8 corners of the bounding box.

        Returns:
            3x8 array where each column is a corner
        """
        x_min, y_min, z_min = self.min[0, 0], self.min[1, 0], self.min[2, 0]
        x_max, y_max, z_max = self.max[0, 0], self.max[1, 0], self.max[2, 0]

        corners = np.array([
            [x_min, x_max, x_max, x_min, x_min, x_max, x_max, x_min],
            [y_min, y_min, y_max, y_max, y_min, y_min, y_max, y_max],
            [z_min, z_min, z_min, z_min, z_max, z_max, z_max, z_max]
        ])

        return corners

    def contains_point(self, point: np.ndarray) -> bool:
        """
        Check if a point is inside the bounding box.

        Args:
            point: 3x1 point to test

        Returns:
            True if point is inside or on the boundary
        """
        return bool(
            np.all(point >= self.min) and
            np.all(point <= self.max)
        )

    def intersects(self, other: 'BoundingBox') -> bool:
        """
        Check if this bounding box intersects another.

        Args:
            other: Other bounding box to test

        Returns:
            True if bounding boxes overlap
        """
        return bool(
            np.all(self.min <= other.max) and
            np.all(self.max >= other.min)
        )

    def merge(self, other: 'BoundingBox') -> 'BoundingBox':
        """
        Merge this bounding box with another.

        Args:
            other: Other bounding box to merge

        Returns:
            New bounding box containing both boxes
        """
        new_min = np.minimum(self.min, other.min)
        new_max = np.maximum(self.max, other.max)
        return BoundingBox(min=new_min, max=new_max)

    def transform(self, matrix: np.ndarray) -> 'BoundingBox':
        """
        Transform bounding box by a 4x4 transformation matrix.

        Args:
            matrix: 4x4 transformation matrix

        Returns:
            New bounding box in transformed space
        """
        # Get all 8 corners
        corners = self.get_corners()  # 3x8

        # Convert to homogeneous coordinates
        corners_homo = np.vstack([corners, np.ones((1, 8))])  # 4x8

        # Transform all corners
        transformed = matrix @ corners_homo  # 4x8

        # Extract 3D points
        transformed_3d = transformed[0:3, :]  # 3x8

        # Create new bounding box from transformed points
        return BoundingBox.from_points(transformed_3d)

    def expand(self, amount: float) -> 'BoundingBox':
        """
        Expand bounding box by a fixed amount in all directions.

        Args:
            amount: Amount to expand (can be negative to shrink)

        Returns:
            New expanded bounding box
        """
        expansion = np.array([[amount], [amount], [amount]])
        return BoundingBox(min=self.min - expansion, max=self.max + expansion)

    def __repr__(self) -> str:
        min_str = f"[{self.min[0,0]:.2f}, {self.min[1,0]:.2f}, {self.min[2,0]:.2f}]"
        max_str = f"[{self.max[0,0]:.2f}, {self.max[1,0]:.2f}, {self.max[2,0]:.2f}]"
        return f"BoundingBox(min={min_str}, max={max_str})"


def compute_sphere_bounds(center: np.ndarray, radius: float) -> BoundingBox:
    """
    Compute bounding box for a sphere.

    Args:
        center: 3x1 center of sphere
        radius: Radius of sphere

    Returns:
        BoundingBox tightly enclosing the sphere
    """
    r = np.array([[radius], [radius], [radius]])
    return BoundingBox(min=center - r, max=center + r)


def compute_hierarchy_bounds(scene_object, include_children: bool = True) -> Optional[BoundingBox]:
    """
    Compute bounding box for a scene object and optionally its children.

    This function computes the world-space bounding box by:
    1. Computing local bounding box for the object
    2. Transforming to world space
    3. Merging with children's bounding boxes if requested

    Args:
        scene_object: SceneObject to compute bounds for
        include_children: If True, include all descendants in the bounds

    Returns:
        BoundingBox in world space, or None if object has no bounds
    """
    # Import here to avoid circular dependency
    from src.primitives.sphere import Sphere
    from src.primitives.cylinder import Cylinder
    from src.primitives.cone import Cone

    # Compute local bounding box based on object type
    local_bounds = None

    if isinstance(scene_object, Sphere):
        # Sphere: bounding box around origin with radius
        local_bounds = compute_sphere_bounds(
            center=np.zeros((3, 1)),
            radius=scene_object.radius
        )

    elif isinstance(scene_object, (Cylinder, Cone)):
        # Cylinder/Cone: bounding box based on height and radius
        height = scene_object.height
        radius = scene_object.radius

        local_bounds = BoundingBox(
            min=np.array([[-radius], [-radius], [0]]),
            max=np.array([[radius], [radius], [height]])
        )

    # If we have local bounds, transform to world space
    if local_bounds is not None:
        world_transform = scene_object.get_world_transform()
        world_bounds = local_bounds.transform(world_transform)
    else:
        world_bounds = None

    # Include children if requested
    if include_children:
        children = scene_object.get_children()
        for child in children:
            child_bounds = compute_hierarchy_bounds(child, include_children=True)
            if child_bounds is not None:
                if world_bounds is None:
                    world_bounds = child_bounds
                else:
                    world_bounds = world_bounds.merge(child_bounds)

    return world_bounds


def find_objects_in_box(scene_object, query_box: BoundingBox,
                       include_children: bool = True) -> List:
    """
    Find all objects whose bounding boxes intersect a query box.

    Args:
        scene_object: Root object to search from
        query_box: Query bounding box
        include_children: If True, search recursively through children

    Returns:
        List of objects whose bounds intersect the query box
    """
    results = []

    # Check this object
    obj_bounds = compute_hierarchy_bounds(scene_object, include_children=False)
    if obj_bounds is not None and obj_bounds.intersects(query_box):
        results.append(scene_object)

    # Check children if requested
    if include_children:
        children = scene_object.get_children()
        for child in children:
            results.extend(find_objects_in_box(child, query_box, include_children=True))

    return results


def check_collision(obj1, obj2) -> bool:
    """
    Check if two scene objects collide (bounding box intersection test).

    Args:
        obj1: First scene object
        obj2: Second scene object

    Returns:
        True if bounding boxes intersect
    """
    bounds1 = compute_hierarchy_bounds(obj1, include_children=True)
    bounds2 = compute_hierarchy_bounds(obj2, include_children=True)

    if bounds1 is None or bounds2 is None:
        return False

    return bounds1.intersects(bounds2)
