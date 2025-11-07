import unittest
import numpy as np
import math

from src.datatypes.pose import Pose
from src.datatypes.scaling import Scaling
from src.datatypes import transform


class TestTransform(unittest.TestCase):

    def test_axis_angle_to_rotation_matrix_identity(self):
        """Test that zero rotation gives identity matrix"""
        axis_angle = np.array([[0], [0], [0]])
        R = transform.axis_angle_to_rotation_matrix(axis_angle)
        np.testing.assert_array_almost_equal(R, np.eye(3))

    def test_axis_angle_to_rotation_matrix_90deg_z(self):
        """Test 90 degree rotation around Z-axis"""
        # 90 degrees around Z-axis
        axis_angle = np.array([[0], [0], [np.pi / 2]])
        R = transform.axis_angle_to_rotation_matrix(axis_angle)

        # Expected rotation matrix for 90° around Z:
        # [cos(90)  -sin(90)  0]   [ 0  -1  0]
        # [sin(90)   cos(90)  0] = [ 1   0  0]
        # [   0         0     1]   [ 0   0  1]
        expected = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        np.testing.assert_array_almost_equal(R, expected)

    def test_axis_angle_to_rotation_matrix_180deg_x(self):
        """Test 180 degree rotation around X-axis"""
        axis_angle = np.array([[np.pi], [0], [0]])
        R = transform.axis_angle_to_rotation_matrix(axis_angle)

        # Expected rotation matrix for 180° around X:
        # [1    0        0   ]
        # [0  cos(180) -sin(180)] = [1  0   0]
        # [0  sin(180)  cos(180)]   [0 -1   0]
        #                            [0  0  -1]
        expected = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
        np.testing.assert_array_almost_equal(R, expected)

    def test_translation_to_matrix(self):
        """Test translation vector to matrix conversion"""
        translation = np.array([[1], [2], [3]])
        T = transform.translation_to_matrix(translation)

        expected = np.array([
            [1, 0, 0, 1],
            [0, 1, 0, 2],
            [0, 0, 1, 3],
            [0, 0, 0, 1]
        ])
        np.testing.assert_array_almost_equal(T, expected)

    def test_rotation_to_matrix(self):
        """Test rotation to 4x4 matrix conversion"""
        rotation = np.array([[0], [0], [np.pi / 2]])
        R = transform.rotation_to_matrix(rotation)

        # Should be 4x4 with 3x3 rotation in upper-left
        self.assertEqual(R.shape, (4, 4))
        # Last row should be [0, 0, 0, 1]
        np.testing.assert_array_almost_equal(R[3, :], [0, 0, 0, 1])
        # Last column (except last element) should be [0, 0, 0]
        np.testing.assert_array_almost_equal(R[0:3, 3], [0, 0, 0])

    def test_scaling_to_matrix(self):
        """Test scaling to matrix conversion"""
        scaling = Scaling(x=2.0, y=3.0, z=0.5)
        S = transform.scaling_to_matrix(scaling)

        expected = np.array([
            [2, 0, 0, 0],
            [0, 3, 0, 0],
            [0, 0, 0.5, 0],
            [0, 0, 0, 1]
        ])
        np.testing.assert_array_almost_equal(S, expected)

    def test_scaling_to_matrix_none(self):
        """Test that None scaling gives identity"""
        S = transform.scaling_to_matrix(None)
        np.testing.assert_array_almost_equal(S, np.eye(4))

    def test_compose_transform_order(self):
        """Test that compose_transform uses correct order (T × R × S)"""
        # Simple test: translate by (1,0,0), rotate 90° around Z, scale by 2 in X
        translation = np.array([[1], [0], [0]])
        rotation = np.array([[0], [0], [np.pi / 2]])
        scaling = Scaling(x=2.0, y=1.0, z=1.0)

        M = transform.compose_transform(translation, rotation, scaling)

        # Apply to point (1, 0, 0)
        # Scale: (2, 0, 0)
        # Rotate 90° around Z: (0, 2, 0)
        # Translate: (1, 2, 0)
        point = np.array([[1], [0], [0], [1]])  # homogeneous
        result = M @ point

        expected = np.array([[1], [2], [0], [1]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_compose_transform_no_scaling(self):
        """Test compose_transform without scaling"""
        translation = np.array([[5], [3], [1]])
        rotation = np.array([[0], [0], [0]])  # No rotation

        M = transform.compose_transform(translation, rotation, None)

        # Should just translate
        point = np.array([[0], [0], [0], [1]])
        result = M @ point

        expected = np.array([[5], [3], [1], [1]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_pose_to_matrix(self):
        """Test pose to matrix conversion"""
        pose = Pose(
            translation=np.array([[2], [3], [4]]),
            rotation=np.array([[0], [0], [0]])
        )
        M = transform.pose_to_matrix(pose)

        # Should be 4x4 transformation matrix
        self.assertEqual(M.shape, (4, 4))

        # Translation should be in last column
        np.testing.assert_array_almost_equal(M[0:3, 3:4], pose.translation)

    def test_pose_to_matrix_with_scaling(self):
        """Test pose to matrix with scaling"""
        pose = Pose(
            translation=np.array([[1], [2], [3]]),
            rotation=np.array([[0], [0], [0]])
        )
        scaling = Scaling(x=2.0, y=2.0, z=2.0)

        M = transform.pose_to_matrix(pose, scaling)

        # Apply to point
        point = np.array([[1], [0], [0], [1]])
        result = M @ point

        # Scale by 2: (2, 0, 0)
        # No rotation: (2, 0, 0)
        # Translate: (3, 2, 3)
        expected = np.array([[3], [2], [3], [1]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_decompose_matrix(self):
        """Test matrix decomposition"""
        translation = np.array([[1], [2], [3]])
        rotation = np.array([[0], [0], [0]])  # Identity rotation
        scaling = Scaling(x=2.0, y=3.0, z=1.0)

        # Create matrix
        M = transform.compose_transform(translation, rotation, scaling)

        # Decompose
        t, r, s = transform.decompose_matrix(M)

        # Check translation
        np.testing.assert_array_almost_equal(t, translation)

        # Check scale
        expected_scale = np.array([[2.0], [3.0], [1.0]])
        np.testing.assert_array_almost_equal(s, expected_scale)

        # Check rotation is identity (within tolerance)
        np.testing.assert_array_almost_equal(r, np.eye(3))

    def test_full_transform_chain(self):
        """Test complete transformation chain with known values"""
        # Create a transformation: translate (10, 0, 0), rotate 90° Z, scale 2x in Y
        translation = np.array([[10], [0], [0]])
        rotation = np.array([[0], [0], [np.pi / 2]])
        scaling = Scaling(x=1.0, y=2.0, z=1.0)

        M = transform.compose_transform(translation, rotation, scaling)

        # Apply to point (0, 1, 0)
        point = np.array([[0], [1], [0], [1]])
        result = M @ point

        # Scale: (0, 2, 0)
        # Rotate 90° Z: (-2, 0, 0)
        # Translate: (8, 0, 0)
        expected = np.array([[8], [0], [0], [1]])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)


if __name__ == '__main__':
    unittest.main()
