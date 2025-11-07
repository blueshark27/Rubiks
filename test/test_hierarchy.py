"""
Unit tests for parent-child hierarchy operations.

Tests parent-child relationships, circular reference detection,
transform propagation, and reparenting scenarios.
"""

import unittest
import numpy as np

from src.primitives.sphere import Sphere
from src.datatypes.pose import Pose
from src.datatypes.scaling import Scaling


class TestHierarchy(unittest.TestCase):

    def setUp(self):
        """Create test objects before each test"""
        # Create some basic poses
        self.origin_pose = Pose(
            translation=np.array([[0], [0], [0]]),
            rotation=np.array([[0], [0], [0]])
        )
        self.offset_pose = Pose(
            translation=np.array([[5], [0], [0]]),
            rotation=np.array([[0], [0], [0]])
        )

    def test_initial_state(self):
        """Test that newly created objects have no parent or children"""
        obj = Sphere(pose=self.origin_pose, name="TestSphere")
        self.assertIsNone(obj.get_parent())
        self.assertEqual(len(obj.get_children()), 0)
        self.assertEqual(obj.get_children(), [])

    def test_add_child(self):
        """Test adding a child to a parent"""
        parent = Sphere(pose=self.origin_pose, name="Parent")
        child = Sphere(pose=self.offset_pose, name="Child")

        parent.add_child(child)

        # Verify relationships
        self.assertEqual(child.get_parent(), parent)
        self.assertIn(child, parent.get_children())
        self.assertEqual(len(parent.get_children()), 1)

    def test_set_parent(self):
        """Test setting a parent via set_parent"""
        parent = Sphere(pose=self.origin_pose, name="Parent")
        child = Sphere(pose=self.offset_pose, name="Child")

        child.set_parent(parent)

        # Verify relationships
        self.assertEqual(child.get_parent(), parent)
        self.assertIn(child, parent.get_children())
        self.assertEqual(len(parent.get_children()), 1)

    def test_multiple_children(self):
        """Test adding multiple children to one parent"""
        parent = Sphere(pose=self.origin_pose, name="Parent")
        child1 = Sphere(pose=self.offset_pose, name="Child1")
        child2 = Sphere(pose=self.offset_pose, name="Child2")
        child3 = Sphere(pose=self.offset_pose, name="Child3")

        parent.add_child(child1)
        parent.add_child(child2)
        parent.add_child(child3)

        # Verify all children are registered
        self.assertEqual(len(parent.get_children()), 3)
        self.assertIn(child1, parent.get_children())
        self.assertIn(child2, parent.get_children())
        self.assertIn(child3, parent.get_children())

        # Verify all have correct parent
        self.assertEqual(child1.get_parent(), parent)
        self.assertEqual(child2.get_parent(), parent)
        self.assertEqual(child3.get_parent(), parent)

    def test_remove_child(self):
        """Test removing a child from parent"""
        parent = Sphere(pose=self.origin_pose, name="Parent")
        child = Sphere(pose=self.offset_pose, name="Child")

        parent.add_child(child)
        self.assertEqual(len(parent.get_children()), 1)

        parent.remove_child(child)

        # Verify child is removed
        self.assertEqual(len(parent.get_children()), 0)
        self.assertNotIn(child, parent.get_children())
        self.assertIsNone(child.get_parent())

    def test_orphaning(self):
        """Test setting parent to None (orphaning)"""
        parent = Sphere(pose=self.origin_pose, name="Parent")
        child = Sphere(pose=self.offset_pose, name="Child")

        parent.add_child(child)
        self.assertEqual(child.get_parent(), parent)

        # Orphan the child
        child.set_parent(None)

        # Verify child is orphaned
        self.assertIsNone(child.get_parent())
        self.assertNotIn(child, parent.get_children())

    def test_reparenting(self):
        """Test changing a child's parent"""
        parent1 = Sphere(pose=self.origin_pose, name="Parent1")
        parent2 = Sphere(pose=self.offset_pose, name="Parent2")
        child = Sphere(pose=self.offset_pose, name="Child")

        # Set initial parent
        child.set_parent(parent1)
        self.assertEqual(child.get_parent(), parent1)
        self.assertIn(child, parent1.get_children())

        # Change parent
        child.set_parent(parent2)

        # Verify new relationships
        self.assertEqual(child.get_parent(), parent2)
        self.assertIn(child, parent2.get_children())
        self.assertEqual(len(parent2.get_children()), 1)

        # Verify old parent no longer has this child
        self.assertNotIn(child, parent1.get_children())
        self.assertEqual(len(parent1.get_children()), 0)

    def test_circular_reference_direct(self):
        """Test that direct circular references are prevented (A -> A)"""
        obj = Sphere(pose=self.origin_pose, name="SelfRef")

        # Attempt to set self as parent
        with self.assertRaises(ValueError) as context:
            obj.set_parent(obj)

        self.assertIn("cycle", str(context.exception).lower())

    def test_circular_reference_two_level(self):
        """Test that two-level circular references are prevented (A -> B -> A)"""
        obj_a = Sphere(pose=self.origin_pose, name="A")
        obj_b = Sphere(pose=self.offset_pose, name="B")

        obj_a.add_child(obj_b)  # A -> B

        # Attempt to make A a child of B (would create cycle)
        with self.assertRaises(ValueError) as context:
            obj_b.add_child(obj_a)

        self.assertIn("cycle", str(context.exception).lower())

    def test_circular_reference_three_level(self):
        """Test that three-level circular references are prevented (A -> B -> C -> A)"""
        obj_a = Sphere(pose=self.origin_pose, name="A")
        obj_b = Sphere(pose=self.offset_pose, name="B")
        obj_c = Sphere(pose=self.offset_pose, name="C")

        obj_a.add_child(obj_b)  # A -> B
        obj_b.add_child(obj_c)  # B -> C

        # Attempt to make A a child of C (would create cycle)
        with self.assertRaises(ValueError) as context:
            obj_c.add_child(obj_a)

        self.assertIn("cycle", str(context.exception).lower())

    def test_deep_hierarchy(self):
        """Test creating a deep hierarchy (grandparent -> parent -> child -> grandchild)"""
        grandparent = Sphere(pose=self.origin_pose, name="Grandparent")
        parent = Sphere(pose=self.offset_pose, name="Parent")
        child = Sphere(pose=self.offset_pose, name="Child")
        grandchild = Sphere(pose=self.offset_pose, name="Grandchild")

        grandparent.add_child(parent)
        parent.add_child(child)
        child.add_child(grandchild)

        # Verify hierarchy structure
        self.assertIsNone(grandparent.get_parent())
        self.assertEqual(parent.get_parent(), grandparent)
        self.assertEqual(child.get_parent(), parent)
        self.assertEqual(grandchild.get_parent(), child)

        # Verify grandchild can access grandparent through chain
        # grandchild -> child -> parent -> grandparent
        self.assertEqual(grandchild.get_parent().get_parent().get_parent(), grandparent)

    def test_transform_propagation_simple(self):
        """Test that transform dirty flags propagate to children"""
        parent = Sphere(pose=self.origin_pose, name="Parent")
        child = Sphere(pose=self.offset_pose, name="Child")
        parent.add_child(child)

        # Get transforms to clear dirty flags
        parent.get_world_transform()
        child.get_world_transform()

        # Verify caches are clean
        self.assertFalse(parent._world_transform_dirty)
        self.assertFalse(child._world_transform_dirty)

        # Update parent pose
        new_pose = Pose(
            translation=np.array([[10], [10], [10]]),
            rotation=np.array([[0], [0], [0]])
        )
        parent.set_pose(new_pose)

        # Verify both parent and child are marked dirty
        self.assertTrue(parent._world_transform_dirty)
        self.assertTrue(child._world_transform_dirty)

    def test_transform_propagation_deep(self):
        """Test that transform dirty flags propagate through deep hierarchy"""
        root = Sphere(pose=self.origin_pose, name="Root")
        level1 = Sphere(pose=self.offset_pose, name="Level1")
        level2 = Sphere(pose=self.offset_pose, name="Level2")
        level3 = Sphere(pose=self.offset_pose, name="Level3")

        root.add_child(level1)
        level1.add_child(level2)
        level2.add_child(level3)

        # Clear dirty flags
        root.get_world_transform()
        level1.get_world_transform()
        level2.get_world_transform()
        level3.get_world_transform()

        # Update root
        new_pose = Pose(
            translation=np.array([[5], [5], [5]]),
            rotation=np.array([[0], [0], [np.pi/4]])
        )
        root.set_pose(new_pose)

        # Verify all levels are marked dirty
        self.assertTrue(root._world_transform_dirty)
        self.assertTrue(level1._world_transform_dirty)
        self.assertTrue(level2._world_transform_dirty)
        self.assertTrue(level3._world_transform_dirty)

    def test_world_transform_inheritance(self):
        """Test that child's world transform includes parent's transform"""
        # Parent at origin, no rotation
        parent = Sphere(
            pose=Pose(
                translation=np.array([[0], [0], [0]]),
                rotation=np.array([[0], [0], [0]])
            ),
            name="Parent"
        )

        # Child offset by (5, 0, 0) from parent
        child = Sphere(
            pose=Pose(
                translation=np.array([[5], [0], [0]]),
                rotation=np.array([[0], [0], [0]])
            ),
            name="Child"
        )

        parent.add_child(child)

        # Get world transforms
        parent_world = parent.get_world_transform()
        child_world = child.get_world_transform()

        # Child's world position should be (5, 0, 0) since parent is at origin
        # Extract translation from world matrix (last column)
        child_world_pos = child_world[0:3, 3:4]
        expected_pos = np.array([[5], [0], [0]])

        np.testing.assert_array_almost_equal(child_world_pos, expected_pos)

    def test_world_transform_with_parent_translation(self):
        """Test world transform when parent is translated"""
        # Parent at (10, 0, 0)
        parent = Sphere(
            pose=Pose(
                translation=np.array([[10], [0], [0]]),
                rotation=np.array([[0], [0], [0]])
            ),
            name="Parent"
        )

        # Child offset by (5, 0, 0) from parent
        child = Sphere(
            pose=Pose(
                translation=np.array([[5], [0], [0]]),
                rotation=np.array([[0], [0], [0]])
            ),
            name="Child"
        )

        parent.add_child(child)

        # Child's world position should be (15, 0, 0) = parent + local offset
        child_world = child.get_world_transform()
        child_world_pos = child_world[0:3, 3:4]
        expected_pos = np.array([[15], [0], [0]])

        np.testing.assert_array_almost_equal(child_world_pos, expected_pos)

    def test_world_transform_with_rotation(self):
        """Test world transform when parent is rotated"""
        # Parent rotated 90° around Z-axis
        parent = Sphere(
            pose=Pose(
                translation=np.array([[0], [0], [0]]),
                rotation=np.array([[0], [0], [np.pi/2]])  # 90° around Z
            ),
            name="Parent"
        )

        # Child offset by (1, 0, 0) from parent
        child = Sphere(
            pose=Pose(
                translation=np.array([[1], [0], [0]]),
                rotation=np.array([[0], [0], [0]])
            ),
            name="Child"
        )

        parent.add_child(child)

        # After 90° Z rotation, point (1, 0, 0) becomes (0, 1, 0)
        child_world = child.get_world_transform()
        child_world_pos = child_world[0:3, 3:4]
        expected_pos = np.array([[0], [1], [0]])

        np.testing.assert_array_almost_equal(child_world_pos, expected_pos, decimal=5)

    def test_scaling_inheritance(self):
        """Test that child inherits parent's scaling in world transform"""
        # Parent with 2x scaling
        parent = Sphere(
            pose=Pose(
                translation=np.array([[0], [0], [0]]),
                rotation=np.array([[0], [0], [0]])
            ),
            scaling=Scaling(x=2.0, y=2.0, z=2.0),
            name="Parent"
        )

        # Child offset by (1, 0, 0)
        child = Sphere(
            pose=Pose(
                translation=np.array([[1], [0], [0]]),
                rotation=np.array([[0], [0], [0]])
            ),
            name="Child"
        )

        parent.add_child(child)

        # Child's world position should be (2, 0, 0) due to parent's 2x scale
        child_world = child.get_world_transform()
        child_world_pos = child_world[0:3, 3:4]
        expected_pos = np.array([[2], [0], [0]])

        np.testing.assert_array_almost_equal(child_world_pos, expected_pos)

    def test_constructor_with_parent(self):
        """Test creating object with parent specified in constructor"""
        parent = Sphere(pose=self.origin_pose, name="Parent")
        child = Sphere(pose=self.offset_pose, name="Child", parent=parent)

        # Verify relationship is established
        self.assertEqual(child.get_parent(), parent)
        self.assertIn(child, parent.get_children())

    def test_no_duplicate_children(self):
        """Test that adding same child twice doesn't create duplicates"""
        parent = Sphere(pose=self.origin_pose, name="Parent")
        child = Sphere(pose=self.offset_pose, name="Child")

        parent.add_child(child)
        parent.add_child(child)  # Add again

        # Should still only have one child
        self.assertEqual(len(parent.get_children()), 1)
        self.assertEqual(parent.get_children().count(child), 1)


if __name__ == '__main__':
    unittest.main()
