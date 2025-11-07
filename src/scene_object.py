from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

from src.datatypes.pose import Pose


class SceneObject(ABC):

    def __init__(self, name: str, pose: Pose, parent: Optional['SceneObject'] = None):
        self.name = name
        self.pose = pose
        self.parent = None
        self.children = []

        # Transform caching for performance
        self._local_transform_dirty = True
        self._world_transform_dirty = True
        self._local_transform_cache = None
        self._world_transform_cache = None

        # Set parent if provided (uses set_parent to maintain consistency)
        if parent is not None:
            self.set_parent(parent)

    def get_pose(self) -> Pose:
        return self.pose

    def set_pose(self, pose: Pose):
        self.pose = pose
        self.mark_transform_dirty()

    def get_name(self) -> str:
        return self.name

    def _would_create_cycle(self, potential_parent: Optional['SceneObject']) -> bool:
        """
        Check if setting potential_parent would create a circular reference.

        Args:
            potential_parent: The object to check

        Returns:
            True if setting this parent would create a cycle, False otherwise
        """
        if potential_parent is None:
            return False

        # Walk up the parent chain
        current = potential_parent
        while current is not None:
            if current is self:
                return True
            current = current.get_parent()

        return False

    def get_parent(self) -> Optional['SceneObject']:
        """
        Get the parent object.

        Returns:
            Parent SceneObject or None if no parent
        """
        return self.parent

    def set_parent(self, parent: Optional['SceneObject']):
        """
        Set the parent object and update parent-child relationships.

        Args:
            parent: The new parent object, or None to remove parent

        Raises:
            ValueError: If setting this parent would create a circular reference
        """
        # Check for circular references
        if self._would_create_cycle(parent):
            raise ValueError(f"Setting parent would create a cycle: {self.name}")

        # Remove from old parent's children list
        if self.parent is not None:
            if self in self.parent.children:
                self.parent.children.remove(self)

        # Set new parent
        self.parent = parent

        # Add to new parent's children list
        if parent is not None:
            if self not in parent.children:
                parent.children.append(self)

        # Mark transform as dirty
        self.mark_transform_dirty()

    def get_children(self) -> List['SceneObject']:
        """
        Get the list of child objects.

        Returns:
            List of child SceneObjects
        """
        return self.children

    def add_child(self, child: 'SceneObject'):
        """
        Add a child object. This sets the child's parent to self.

        Args:
            child: The child object to add

        Raises:
            ValueError: If adding this child would create a circular reference
        """
        # This will handle circular reference checking and updating both sides
        child.set_parent(self)

    def remove_child(self, child: 'SceneObject'):
        """
        Remove a child object. This sets the child's parent to None.

        Args:
            child: The child object to remove
        """
        if child in self.children:
            self.children.remove(child)
            child.parent = None
            child.mark_transform_dirty()

    def mark_transform_dirty(self):
        """
        Mark this object's transform cache as dirty, requiring recalculation.
        Recursively marks all children as dirty as well.
        """
        self._local_transform_dirty = True
        self._world_transform_dirty = True

        # Recursively mark all children
        for child in self.children:
            child.mark_transform_dirty()

    def get_local_transform(self) -> np.ndarray:
        """
        Get the local transformation matrix (pose only, no scaling).
        Uses caching to avoid unnecessary recalculation.

        Returns:
            4x4 local transformation matrix
        """
        if self._local_transform_dirty:
            from src.datatypes import transform
            self._local_transform_cache = transform.pose_to_matrix(self.pose)
            self._local_transform_dirty = False

        return self._local_transform_cache

    def get_world_transform(self) -> np.ndarray:
        """
        Get the world transformation matrix by traversing the parent hierarchy.
        Uses caching to avoid unnecessary recalculation.

        Returns:
            4x4 world transformation matrix
        """
        if self._world_transform_dirty:
            local = self.get_local_transform()

            if self.parent is None:
                # No parent, world transform = local transform
                self._world_transform_cache = local.copy()
            else:
                # Multiply parent's world transform with our local transform
                parent_world = self.parent.get_world_transform()
                self._world_transform_cache = parent_world @ local

            self._world_transform_dirty = False

        return self._world_transform_cache
