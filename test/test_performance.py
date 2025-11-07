"""
Performance tests for hierarchical transformations.

Tests the performance of transform calculations with:
- Different hierarchy depths
- Different hierarchy widths (number of children)
- Cache effectiveness
"""

import unittest
import numpy as np
import time

from src.primitives.sphere import Sphere
from src.datatypes.pose import Pose


class TestPerformance(unittest.TestCase):

    def setUp(self):
        """Create test poses"""
        self.origin_pose = Pose(
            translation=np.array([[0], [0], [0]]),
            rotation=np.array([[0], [0], [0]])
        )
        self.offset_pose = Pose(
            translation=np.array([[1], [0], [0]]),
            rotation=np.array([[0], [0], [0.1]])
        )

    def create_linear_hierarchy(self, depth):
        """Create a linear hierarchy (chain) of given depth"""
        objects = []
        for i in range(depth):
            obj = Sphere(pose=self.offset_pose, name=f"Object{i}")
            objects.append(obj)
            if i > 0:
                objects[i-1].add_child(obj)
        return objects

    def create_wide_hierarchy(self, width):
        """Create a flat hierarchy with one parent and many children"""
        parent = Sphere(pose=self.origin_pose, name="Parent")
        children = []
        for i in range(width):
            child = Sphere(pose=self.offset_pose, name=f"Child{i}")
            parent.add_child(child)
            children.append(child)
        return parent, children

    def create_balanced_tree(self, depth, branching_factor):
        """Create a balanced tree hierarchy"""
        def create_node(level, parent_name=""):
            name = f"{parent_name}L{level}"
            node = Sphere(pose=self.offset_pose, name=name)

            if level < depth:
                for i in range(branching_factor):
                    child = create_node(level + 1, f"{name}_C{i}")
                    node.add_child(child)

            return node

        return create_node(0)

    def test_linear_hierarchy_depth_10(self):
        """Test performance with depth=10 linear hierarchy"""
        objects = self.create_linear_hierarchy(10)

        start = time.perf_counter()
        for _ in range(100):
            objects[-1].get_world_transform()  # Get transform of deepest object
        end = time.perf_counter()

        elapsed = end - start
        per_call = elapsed / 100 * 1000  # ms per call

        print(f"\nLinear Hierarchy (depth=10): {per_call:.4f} ms per call")
        self.assertLess(per_call, 10, "Transform calculation took too long")

    def test_linear_hierarchy_depth_50(self):
        """Test performance with depth=50 linear hierarchy"""
        objects = self.create_linear_hierarchy(50)

        start = time.perf_counter()
        for _ in range(100):
            objects[-1].get_world_transform()
        end = time.perf_counter()

        elapsed = end - start
        per_call = elapsed / 100 * 1000

        print(f"Linear Hierarchy (depth=50): {per_call:.4f} ms per call")
        self.assertLess(per_call, 50, "Deep hierarchy took too long")

    def test_wide_hierarchy_100_children(self):
        """Test performance with 100 children"""
        parent, children = self.create_wide_hierarchy(100)

        start = time.perf_counter()
        for _ in range(100):
            for child in children:
                child.get_world_transform()
        end = time.perf_counter()

        elapsed = end - start
        per_child = elapsed / (100 * 100) * 1000

        print(f"Wide Hierarchy (100 children): {per_child:.4f} ms per child")
        self.assertLess(per_child, 1, "Wide hierarchy took too long")

    def test_cache_effectiveness(self):
        """Test that caching improves performance"""
        objects = self.create_linear_hierarchy(20)
        leaf = objects[-1]

        # First call (cache miss)
        start = time.perf_counter()
        for _ in range(100):
            leaf._world_transform_dirty = True  # Force recalculation
            leaf.get_world_transform()
        end = time.perf_counter()
        uncached_time = end - start

        # Second call (cache hit)
        start = time.perf_counter()
        for _ in range(100):
            leaf.get_world_transform()  # Should use cache
        end = time.perf_counter()
        cached_time = end - start

        speedup = uncached_time / cached_time if cached_time > 0 else float('inf')

        print(f"Cache speedup: {speedup:.1f}x faster")
        print(f"  Uncached: {uncached_time*1000:.4f} ms")
        print(f"  Cached:   {cached_time*1000:.4f} ms")

        self.assertGreater(speedup, 10, "Cache should provide significant speedup")

    def test_update_propagation_performance(self):
        """Test performance of dirty flag propagation"""
        # Create a tree with multiple levels and children
        root = self.create_balanced_tree(depth=4, branching_factor=3)

        # Count total objects
        def count_nodes(node):
            count = 1
            for child in node.get_children():
                count += count_nodes(child)
            return count

        total_objects = count_nodes(root)

        # Time how long it takes to update root and propagate
        start = time.perf_counter()
        for _ in range(100):
            new_pose = Pose(
                translation=np.array([[1], [2], [3]]),
                rotation=np.array([[0.1], [0.2], [0.3]])
            )
            root.set_pose(new_pose)
        end = time.perf_counter()

        elapsed = end - start
        per_update = elapsed / 100 * 1000

        print(f"Update propagation ({total_objects} objects): {per_update:.4f} ms per update")
        self.assertLess(per_update, 10, "Update propagation too slow")

    def test_balanced_tree_performance(self):
        """Test performance with balanced tree (depth=4, branching=3)"""
        root = self.create_balanced_tree(depth=4, branching_factor=3)

        # Get all leaf nodes
        def get_leaves(node):
            if not node.get_children():
                return [node]
            leaves = []
            for child in node.get_children():
                leaves.extend(get_leaves(child))
            return leaves

        leaves = get_leaves(root)

        start = time.perf_counter()
        for leaf in leaves:
            leaf.get_world_transform()
        end = time.perf_counter()

        elapsed = end - start
        per_leaf = elapsed / len(leaves) * 1000

        print(f"Balanced tree ({len(leaves)} leaves): {per_leaf:.4f} ms per leaf")
        self.assertLess(per_leaf, 1, "Balanced tree performance degraded")

    def test_create_hierarchy_performance(self):
        """Test performance of creating hierarchies"""
        start = time.perf_counter()
        for _ in range(100):
            objects = []
            for i in range(10):
                obj = Sphere(pose=self.offset_pose, name=f"Obj{i}")
                objects.append(obj)
                if i > 0:
                    objects[i-1].add_child(obj)
        end = time.perf_counter()

        elapsed = end - start
        per_creation = elapsed / 100 * 1000

        print(f"Hierarchy creation (depth=10): {per_creation:.4f} ms per hierarchy")
        self.assertLess(per_creation, 50, "Hierarchy creation too slow")

    def test_reparenting_performance(self):
        """Test performance of reparenting operations"""
        parent1 = Sphere(pose=self.origin_pose, name="Parent1")
        parent2 = Sphere(pose=self.origin_pose, name="Parent2")
        child = Sphere(pose=self.offset_pose, name="Child")

        start = time.perf_counter()
        for i in range(1000):
            if i % 2 == 0:
                child.set_parent(parent1)
            else:
                child.set_parent(parent2)
        end = time.perf_counter()

        elapsed = end - start
        per_reparent = elapsed / 1000 * 1000

        print(f"Reparenting: {per_reparent:.4f} ms per operation")
        self.assertLess(per_reparent, 1, "Reparenting too slow")


if __name__ == '__main__':
    # Run with verbose output to see timing information
    unittest.main(verbosity=2)
