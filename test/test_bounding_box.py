"""
Unit tests for bounding box computation and operations.
"""

import unittest
import numpy as np

from src.utils.bounding_box import (
    BoundingBox, compute_sphere_bounds, compute_hierarchy_bounds,
    find_objects_in_box, check_collision
)
from src.primitives.sphere import Sphere
from src.datatypes.pose import Pose


class TestBoundingBox(unittest.TestCase):

    def test_bounding_box_creation(self):
        """Test creating a bounding box"""
        bbox = BoundingBox(
            min=np.array([[0], [0], [0]]),
            max=np.array([[1], [1], [1]])
        )

        np.testing.assert_array_equal(bbox.min, np.array([[0], [0], [0]]))
        np.testing.assert_array_equal(bbox.max, np.array([[1], [1], [1]]))

    def test_invalid_bounding_box_raises_error(self):
        """Test that invalid bounding box raises ValueError"""
        with self.assertRaises(ValueError):
            # min > max
            BoundingBox(
                min=np.array([[1], [1], [1]]),
                max=np.array([[0], [0], [0]])
            )

    def test_from_points(self):
        """Test creating bounding box from points"""
        points = np.array([
            [0, 0, 0],
            [1, 2, 3],
            [-1, 1, -1],
            [2, -1, 1]
        ])

        bbox = BoundingBox.from_points(points)

        np.testing.assert_array_equal(bbox.min, np.array([[-1], [-1], [-1]]))
        np.testing.assert_array_equal(bbox.max, np.array([[2], [2], [3]]))

    def test_from_center_size(self):
        """Test creating bounding box from center and size"""
        center = np.array([[5], [5], [5]])
        size = np.array([[2], [4], [6]])

        bbox = BoundingBox.from_center_size(center, size)

        expected_min = np.array([[4], [3], [2]])
        expected_max = np.array([[6], [7], [8]])

        np.testing.assert_array_almost_equal(bbox.min, expected_min)
        np.testing.assert_array_almost_equal(bbox.max, expected_max)

    def test_empty_bounding_box(self):
        """Test creating empty bounding box"""
        bbox = BoundingBox.empty()

        # Empty bbox has inverted min/max for merging
        self.assertTrue(np.all(bbox.min > bbox.max))

    def test_get_center(self):
        """Test getting center of bounding box"""
        bbox = BoundingBox(
            min=np.array([[0], [0], [0]]),
            max=np.array([[4], [6], [8]])
        )

        center = bbox.get_center()
        expected = np.array([[2], [3], [4]])

        np.testing.assert_array_almost_equal(center, expected)

    def test_get_size(self):
        """Test getting size of bounding box"""
        bbox = BoundingBox(
            min=np.array([[1], [2], [3]]),
            max=np.array([[4], [7], [9]])
        )

        size = bbox.get_size()
        expected = np.array([[3], [5], [6]])

        np.testing.assert_array_almost_equal(size, expected)

    def test_get_volume(self):
        """Test computing volume"""
        bbox = BoundingBox(
            min=np.array([[0], [0], [0]]),
            max=np.array([[2], [3], [4]])
        )

        volume = bbox.get_volume()
        self.assertAlmostEqual(volume, 24.0)

    def test_get_surface_area(self):
        """Test computing surface area"""
        bbox = BoundingBox(
            min=np.array([[0], [0], [0]]),
            max=np.array([[2], [2], [2]])
        )

        # Cube with side 2: surface area = 6 * 4 = 24
        area = bbox.get_surface_area()
        self.assertAlmostEqual(area, 24.0)

    def test_get_corners(self):
        """Test getting all 8 corners"""
        bbox = BoundingBox(
            min=np.array([[0], [0], [0]]),
            max=np.array([[1], [1], [1]])
        )

        corners = bbox.get_corners()

        # Should have 8 corners
        self.assertEqual(corners.shape, (3, 8))

        # Check that all corners are at min or max for each coordinate
        for i in range(8):
            corner = corners[:, i]
            for coord in range(3):
                self.assertIn(corner[coord], [0.0, 1.0])

    def test_contains_point_inside(self):
        """Test point containment - point inside"""
        bbox = BoundingBox(
            min=np.array([[0], [0], [0]]),
            max=np.array([[1], [1], [1]])
        )

        point = np.array([[0.5], [0.5], [0.5]])
        self.assertTrue(bbox.contains_point(point))

    def test_contains_point_outside(self):
        """Test point containment - point outside"""
        bbox = BoundingBox(
            min=np.array([[0], [0], [0]]),
            max=np.array([[1], [1], [1]])
        )

        point = np.array([[2], [0.5], [0.5]])
        self.assertFalse(bbox.contains_point(point))

    def test_contains_point_on_boundary(self):
        """Test point containment - point on boundary"""
        bbox = BoundingBox(
            min=np.array([[0], [0], [0]]),
            max=np.array([[1], [1], [1]])
        )

        point = np.array([[0], [0.5], [0.5]])
        self.assertTrue(bbox.contains_point(point))

    def test_intersects_overlapping(self):
        """Test intersection - overlapping boxes"""
        bbox1 = BoundingBox(
            min=np.array([[0], [0], [0]]),
            max=np.array([[2], [2], [2]])
        )
        bbox2 = BoundingBox(
            min=np.array([[1], [1], [1]]),
            max=np.array([[3], [3], [3]])
        )

        self.assertTrue(bbox1.intersects(bbox2))
        self.assertTrue(bbox2.intersects(bbox1))

    def test_intersects_separated(self):
        """Test intersection - separated boxes"""
        bbox1 = BoundingBox(
            min=np.array([[0], [0], [0]]),
            max=np.array([[1], [1], [1]])
        )
        bbox2 = BoundingBox(
            min=np.array([[2], [2], [2]]),
            max=np.array([[3], [3], [3]])
        )

        self.assertFalse(bbox1.intersects(bbox2))
        self.assertFalse(bbox2.intersects(bbox1))

    def test_intersects_touching(self):
        """Test intersection - touching boxes"""
        bbox1 = BoundingBox(
            min=np.array([[0], [0], [0]]),
            max=np.array([[1], [1], [1]])
        )
        bbox2 = BoundingBox(
            min=np.array([[1], [0], [0]]),
            max=np.array([[2], [1], [1]])
        )

        self.assertTrue(bbox1.intersects(bbox2))

    def test_merge(self):
        """Test merging two bounding boxes"""
        bbox1 = BoundingBox(
            min=np.array([[0], [0], [0]]),
            max=np.array([[1], [1], [1]])
        )
        bbox2 = BoundingBox(
            min=np.array([[2], [2], [2]]),
            max=np.array([[3], [3], [3]])
        )

        merged = bbox1.merge(bbox2)

        np.testing.assert_array_equal(merged.min, np.array([[0], [0], [0]]))
        np.testing.assert_array_equal(merged.max, np.array([[3], [3], [3]]))

    def test_transform_identity(self):
        """Test transforming by identity matrix"""
        bbox = BoundingBox(
            min=np.array([[0], [0], [0]]),
            max=np.array([[1], [1], [1]])
        )

        transformed = bbox.transform(np.eye(4))

        np.testing.assert_array_almost_equal(transformed.min, bbox.min)
        np.testing.assert_array_almost_equal(transformed.max, bbox.max)

    def test_transform_translation(self):
        """Test transforming by translation"""
        bbox = BoundingBox(
            min=np.array([[0], [0], [0]]),
            max=np.array([[1], [1], [1]])
        )

        # Translation by (5, 5, 5)
        T = np.eye(4)
        T[0:3, 3] = [5, 5, 5]

        transformed = bbox.transform(T)

        expected_min = np.array([[5], [5], [5]])
        expected_max = np.array([[6], [6], [6]])

        np.testing.assert_array_almost_equal(transformed.min, expected_min)
        np.testing.assert_array_almost_equal(transformed.max, expected_max)

    def test_transform_scaling(self):
        """Test transforming by scaling"""
        bbox = BoundingBox(
            min=np.array([[-1], [-1], [-1]]),
            max=np.array([[1], [1], [1]])
        )

        # Scale by 2
        S = np.eye(4)
        S[0, 0] = 2.0
        S[1, 1] = 2.0
        S[2, 2] = 2.0

        transformed = bbox.transform(S)

        expected_min = np.array([[-2], [-2], [-2]])
        expected_max = np.array([[2], [2], [2]])

        np.testing.assert_array_almost_equal(transformed.min, expected_min)
        np.testing.assert_array_almost_equal(transformed.max, expected_max)

    def test_expand(self):
        """Test expanding bounding box"""
        bbox = BoundingBox(
            min=np.array([[0], [0], [0]]),
            max=np.array([[1], [1], [1]])
        )

        expanded = bbox.expand(0.5)

        expected_min = np.array([[-0.5], [-0.5], [-0.5]])
        expected_max = np.array([[1.5], [1.5], [1.5]])

        np.testing.assert_array_almost_equal(expanded.min, expected_min)
        np.testing.assert_array_almost_equal(expanded.max, expected_max)

    def test_repr(self):
        """Test string representation"""
        bbox = BoundingBox(
            min=np.array([[0], [0], [0]]),
            max=np.array([[1], [1], [1]])
        )

        repr_str = repr(bbox)
        self.assertIn("BoundingBox", repr_str)
        self.assertIn("0.00", repr_str)
        self.assertIn("1.00", repr_str)


class TestBoundingBoxUtilities(unittest.TestCase):

    def test_compute_sphere_bounds(self):
        """Test computing bounds for a sphere"""
        center = np.array([[5], [5], [5]])
        radius = 2.0

        bbox = compute_sphere_bounds(center, radius)

        expected_min = np.array([[3], [3], [3]])
        expected_max = np.array([[7], [7], [7]])

        np.testing.assert_array_almost_equal(bbox.min, expected_min)
        np.testing.assert_array_almost_equal(bbox.max, expected_max)

    def test_compute_hierarchy_bounds_single_sphere(self):
        """Test computing bounds for a single sphere"""
        pose = Pose(
            translation=np.array([[0], [0], [0]]),
            rotation=np.array([[0], [0], [0]])
        )
        sphere = Sphere(pose=pose, radius=1.0, name="test")

        bbox = compute_hierarchy_bounds(sphere, include_children=False)

        # Should be centered at origin with radius 1
        self.assertIsNotNone(bbox)
        np.testing.assert_array_almost_equal(bbox.min, np.array([[-1], [-1], [-1]]))
        np.testing.assert_array_almost_equal(bbox.max, np.array([[1], [1], [1]]))

    def test_compute_hierarchy_bounds_translated_sphere(self):
        """Test computing bounds for a translated sphere"""
        pose = Pose(
            translation=np.array([[5], [0], [0]]),
            rotation=np.array([[0], [0], [0]])
        )
        sphere = Sphere(pose=pose, radius=1.0, name="test")

        bbox = compute_hierarchy_bounds(sphere, include_children=False)

        # Should be centered at (5, 0, 0) with radius 1
        self.assertIsNotNone(bbox)
        np.testing.assert_array_almost_equal(bbox.min, np.array([[4], [-1], [-1]]))
        np.testing.assert_array_almost_equal(bbox.max, np.array([[6], [1], [1]]))

    def test_compute_hierarchy_bounds_with_children(self):
        """Test computing bounds for hierarchy with children"""
        parent_pose = Pose(
            translation=np.array([[0], [0], [0]]),
            rotation=np.array([[0], [0], [0]])
        )
        parent = Sphere(pose=parent_pose, radius=1.0, name="parent")

        child_pose = Pose(
            translation=np.array([[5], [0], [0]]),
            rotation=np.array([[0], [0], [0]])
        )
        child = Sphere(pose=child_pose, radius=1.0, name="child", parent=parent)

        bbox = compute_hierarchy_bounds(parent, include_children=True)

        # Should enclose both spheres
        self.assertIsNotNone(bbox)

        # Parent is at origin with radius 1: [-1, 1] in all dims
        # Child is at (5, 0, 0) relative to parent with radius 1: [4, 6] in x, [-1, 1] in y,z
        # Combined should be roughly [-1, 6] in x, [-1, 1] in y,z
        self.assertLessEqual(bbox.min[0, 0], -0.9)
        self.assertGreaterEqual(bbox.max[0, 0], 5.9)

    def test_find_objects_in_box(self):
        """Test finding objects in a query box"""
        # Create three spheres at different positions
        sphere1 = Sphere(
            pose=Pose(
                translation=np.array([[0], [0], [0]]),
                rotation=np.array([[0], [0], [0]])
            ),
            radius=1.0,
            name="sphere1"
        )

        sphere2 = Sphere(
            pose=Pose(
                translation=np.array([[5], [0], [0]]),
                rotation=np.array([[0], [0], [0]])
            ),
            radius=1.0,
            name="sphere2",
            parent=sphere1
        )

        sphere3 = Sphere(
            pose=Pose(
                translation=np.array([[10], [0], [0]]),
                rotation=np.array([[0], [0], [0]])
            ),
            radius=1.0,
            name="sphere3",
            parent=sphere1
        )

        # Query box near origin
        query = BoundingBox(
            min=np.array([[-2], [-2], [-2]]),
            max=np.array([[2], [2], [2]])
        )

        results = find_objects_in_box(sphere1, query, include_children=True)

        # Should find sphere1 (at origin) but not sphere2 or sphere3
        self.assertIn(sphere1, results)

    def test_check_collision_intersecting(self):
        """Test collision detection - intersecting spheres"""
        sphere1 = Sphere(
            pose=Pose(
                translation=np.array([[0], [0], [0]]),
                rotation=np.array([[0], [0], [0]])
            ),
            radius=1.0,
            name="sphere1"
        )

        sphere2 = Sphere(
            pose=Pose(
                translation=np.array([[1], [0], [0]]),
                rotation=np.array([[0], [0], [0]])
            ),
            radius=1.0,
            name="sphere2"
        )

        self.assertTrue(check_collision(sphere1, sphere2))

    def test_check_collision_separated(self):
        """Test collision detection - separated spheres"""
        sphere1 = Sphere(
            pose=Pose(
                translation=np.array([[0], [0], [0]]),
                rotation=np.array([[0], [0], [0]])
            ),
            radius=1.0,
            name="sphere1"
        )

        sphere2 = Sphere(
            pose=Pose(
                translation=np.array([[10], [0], [0]]),
                rotation=np.array([[0], [0], [0]])
            ),
            radius=1.0,
            name="sphere2"
        )

        self.assertFalse(check_collision(sphere1, sphere2))


if __name__ == '__main__':
    unittest.main()
