import numpy as np
from src.datatypes.pose import Pose
import unittest


class TestPose(unittest.TestCase):
    def test_pose_initialization(self):
        # Test initialization with valid input
        pose = Pose(translation=np.array([[1], [2], [3]]), rotation=np.array([[0], [0], [1]]))
        self.assertTrue(isinstance(pose, Pose))

    def test_pose_initialization_invalid(self):
        # Test initialization with invalid input
        with self.assertRaises(ValueError):
            Pose(translation=np.array([[1], [2]]), rotation=np.array([[0], [0], [1]]))
        with self.assertRaises(ValueError):
            Pose(translation=np.array([[1], [2], [3]]), rotation=np.array([[0], [0]]))
        with self.assertRaises(ValueError):
            Pose(translation=np.array([[1], [2], [3]]), rotation=np.array([[0, 0, 0]]))
        with self.assertRaises(ValueError):
            Pose(translation=np.array([[1], [2], [3]]), rotation="invalid_rotation")
        with self.assertRaises(ValueError):
            Pose(translation="invalid_translation", rotation=np.array([[0], [0], [1]]))
        with self.assertRaises(ValueError):
            Pose(translation=None, rotation=np.array([[0], [0], [1]]))

    def test_get_translation(self):
        # Test getting translation
        pose = Pose(translation=np.array([[1], [2], [3]]), rotation=np.array([[0], [0], [1]]))
        translation = pose.get_translation()
        expected_translation = np.array([[1], [2], [3]])
        np.testing.assert_array_equal(translation, expected_translation)

    def test_get_translation_as_homogeneous(self):
        # Test getting translation as homogeneous coordinates
        pose = Pose(translation=np.array([[1], [2], [3]]), rotation=np.array([[0], [0], [1]]))
        homogeneous_translation = pose.get_translation_as_homogeneous()
        expected_homogeneous_translation = np.array([[1], [2], [3], [1]])
        np.testing.assert_array_equal(homogeneous_translation, expected_homogeneous_translation)

    def test_get_rotation(self):
        # Test getting rotation
        pose = Pose(translation=np.array([[1], [2], [3]]), rotation=np.array([[0], [0], [1]]))
        rotation = pose.get_rotation()
        expected_rotation = np.array([[0], [0], [1]])
        np.testing.assert_array_equal(rotation, expected_rotation)

    def test_set_translation(self):
        # Test setting translation
        pose = Pose(translation=np.array([[1], [2], [3]]), rotation=np.array([[0], [0], [1]]))
        new_translation = np.array([[3], [1], [2]])
        pose.set_translation(new_translation)
        np.testing.assert_array_equal(pose.get_translation(), new_translation)

    def test_set_rotation(self):
        # Test setting rotation
        pose = Pose(translation=np.array([[1], [2], [3]]), rotation=np.array([[0], [0], [1]]))
        new_rotation = np.array([[1], [0], [0]])
        pose.set_rotation(new_rotation)
        np.testing.assert_array_equal(pose.get_rotation(), new_rotation)


if __name__ == "__main__":
    unittest.main()
