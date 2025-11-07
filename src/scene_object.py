from abc import ABC, abstractmethod
from typing import List, Optional, Dict
import numpy as np
import warnings

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

        # Performance monitoring
        self._cache_hits = 0
        self._cache_misses = 0
        self._depth_warning_threshold = 15  # Warn if hierarchy exceeds this depth

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
            self._cache_misses += 1
            local = self.get_local_transform()

            if self.parent is None:
                # No parent, world transform = local transform
                self._world_transform_cache = local.copy()
            else:
                # Multiply parent's world transform with our local transform
                parent_world = self.parent.get_world_transform()
                self._world_transform_cache = parent_world @ local

                # Check hierarchy depth and warn if too deep
                depth = self.get_depth()
                if depth > self._depth_warning_threshold:
                    warnings.warn(
                        f"Deep hierarchy detected: '{self.name}' is at depth {depth}. "
                        f"Consider restructuring for better performance.",
                        UserWarning,
                        stacklevel=2
                    )

            self._world_transform_dirty = False
        else:
            self._cache_hits += 1

        return self._world_transform_cache

    # ===== Optimization & Monitoring Methods =====

    def get_depth(self) -> int:
        """
        Get the depth of this object in the hierarchy.

        Returns:
            Depth (0 for root, 1 for children of root, etc.)
        """
        depth = 0
        current = self.parent
        while current is not None:
            depth += 1
            current = current.get_parent()
        return depth

    def get_hierarchy_depth(self) -> int:
        """
        Get the maximum depth of the hierarchy below this object.

        Returns:
            Maximum depth of subtree (0 if no children)
        """
        if not self.children:
            return 0

        max_child_depth = 0
        for child in self.children:
            child_depth = child.get_hierarchy_depth()
            max_child_depth = max(max_child_depth, child_depth)

        return 1 + max_child_depth

    def get_descendant_count(self) -> int:
        """
        Count total number of descendants (children, grandchildren, etc.).

        Returns:
            Total number of descendants
        """
        count = len(self.children)
        for child in self.children:
            count += child.get_descendant_count()
        return count

    def get_cache_statistics(self) -> Dict[str, int]:
        """
        Get cache performance statistics for this object.

        Returns:
            Dictionary with cache hits, misses, and hit rate
        """
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0.0

        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'total': total,
            'hit_rate_percent': hit_rate
        }

    def reset_cache_statistics(self):
        """Reset cache performance counters."""
        self._cache_hits = 0
        self._cache_misses = 0

    def get_hierarchy_statistics(self) -> Dict[str, int]:
        """
        Get comprehensive statistics about this hierarchy.

        Returns:
            Dictionary with depth, descendant count, and subtree depth
        """
        return {
            'depth': self.get_depth(),
            'descendants': self.get_descendant_count(),
            'subtree_depth': self.get_hierarchy_depth(),
            'children': len(self.children)
        }

    def print_hierarchy(self, indent: int = 0, show_stats: bool = False):
        """
        Print a visual representation of the hierarchy tree.

        Args:
            indent: Current indentation level (used for recursion)
            show_stats: If True, show cache statistics for each node
        """
        prefix = "  " * indent
        stats_str = ""

        if show_stats:
            cache_stats = self.get_cache_statistics()
            stats_str = f" [cache: {cache_stats['hit_rate_percent']:.1f}% hits, depth: {self.get_depth()}]"

        print(f"{prefix}{self.name}{stats_str}")

        for child in self.children:
            child.print_hierarchy(indent + 1, show_stats)

    def validate_hierarchy(self) -> List[str]:
        """
        Validate the hierarchy and return a list of warnings/issues.

        Returns:
            List of warning messages (empty if no issues)
        """
        warnings_list = []

        # Check depth
        depth = self.get_depth()
        if depth > self._depth_warning_threshold:
            warnings_list.append(
                f"Object '{self.name}' is at depth {depth} "
                f"(threshold: {self._depth_warning_threshold})"
            )

        # Check for too many children (can impact iteration performance)
        if len(self.children) > 100:
            warnings_list.append(
                f"Object '{self.name}' has {len(self.children)} children "
                f"(consider grouping for better organization)"
            )

        # Recursively check children
        for child in self.children:
            warnings_list.extend(child.validate_hierarchy())

        return warnings_list

    def set_depth_warning_threshold(self, threshold: int):
        """
        Set the depth threshold for performance warnings.

        Args:
            threshold: Maximum depth before warning is issued
        """
        self._depth_warning_threshold = threshold
