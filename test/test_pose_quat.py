"""
Unit tests for quaternion-based pose representation.
"""

import unittest
import numpy as np
import math

from src.datatypes.pose import PoseQuat, Pose
from src.datatypes.quaternion import Quaternion


class TestPoseQuat(unittest.TestCase):

    def test_identity_pose(self):
        """Test identity pose creation"""
        pose = PoseQuat.identity()

        # Check translation is zero
        np.testing.assert_array_almost_equal(pose.translation, np.zeros((3, 1)))

        # Check quaternion is identity
        self.assertAlmostEqual(pose.quaternion.w, 1.0)
        self.assertAlmostEqual(pose.quaternion.x, 0.0)
        self.assertAlmostEqual(pose.quaternion.y, 0.0)
        self.assertAlmostEqual(pose.quaternion.z, 0.0)

    def test_from_translation_quaternion(self):
        """Test creating pose from translation and quaternion"""
        translation = np.array([[1.0], [2.0], [3.0]])
        quat = Quaternion.from_axis_angle(np.array([[0], [0], [1]]), np.pi/2)

        pose = PoseQuat.from_translation_quaternion(translation, quat)

        np.testing.assert_array_almost_equal(pose.translation, translation)
        self.assertAlmostEqual(pose.quaternion.w, quat.w)
        self.assertAlmostEqual(pose.quaternion.z, quat.z)

    def test_from_translation_axis_angle(self):
        """Test creating pose from translation and axis-angle"""
        translation = np.array([[1.0], [2.0], [3.0]])
        axis = np.array([[0], [0], [1]])
        angle = np.pi / 2

        pose = PoseQuat.from_translation_axis_angle(translation, axis, angle)

        np.testing.assert_array_almost_equal(pose.translation, translation)

        # Check quaternion represents 90° around Z
        expected_w = math.cos(np.pi / 4)
        expected_z = math.sin(np.pi / 4)
        self.assertAlmostEqual(pose.quaternion.w, expected_w, places=5)
        self.assertAlmostEqual(pose.quaternion.z, expected_z, places=5)

    def test_from_translation_euler(self):
        """Test creating pose from translation and Euler angles"""
        translation = np.array([[1.0], [2.0], [3.0]])
        roll, pitch, yaw = 0.1, 0.2, 0.3

        pose = PoseQuat.from_translation_euler(translation, roll, pitch, yaw)

        np.testing.assert_array_almost_equal(pose.translation, translation)

        # Verify by converting back to Euler
        roll_out, pitch_out, yaw_out = pose.quaternion.to_euler()
        self.assertAlmostEqual(roll_out, roll, places=5)
        self.assertAlmostEqual(pitch_out, pitch, places=5)
        self.assertAlmostEqual(yaw_out, yaw, places=5)

    def test_get_set_translation(self):
        """Test getting and setting translation"""
        pose = PoseQuat.identity()

        new_translation = np.array([[5.0], [6.0], [7.0]])
        pose.set_translation(new_translation)

        np.testing.assert_array_almost_equal(pose.get_translation(), new_translation)
        np.testing.assert_array_almost_equal(pose.t(), new_translation)

    def test_get_set_quaternion(self):
        """Test getting and setting quaternion"""
        pose = PoseQuat.identity()

        new_quat = Quaternion.from_axis_angle(np.array([[1], [0], [0]]), np.pi/4)
        pose.set_quaternion(new_quat)

        retrieved_quat = pose.get_quaternion()
        self.assertAlmostEqual(retrieved_quat.w, new_quat.w)
        self.assertAlmostEqual(retrieved_quat.x, new_quat.x)

        # Test shorthand
        self.assertAlmostEqual(pose.q().w, new_quat.w)

    def test_get_translation_as_homogeneous(self):
        """Test homogeneous translation vector"""
        translation = np.array([[1.0], [2.0], [3.0]])
        pose = PoseQuat.from_translation_quaternion(translation, Quaternion.identity())

        homogeneous = pose.get_translation_as_homogeneous()

        expected = np.array([[1.0], [2.0], [3.0], [1.0]])
        np.testing.assert_array_almost_equal(homogeneous, expected)

    def test_to_matrix(self):
        """Test conversion to transformation matrix"""
        translation = np.array([[1.0], [2.0], [3.0]])
        quat = Quaternion.from_axis_angle(np.array([[0], [0], [1]]), np.pi/2)

        pose = PoseQuat.from_translation_quaternion(translation, quat)
        matrix = pose.to_matrix()

        # Check it's 4x4
        self.assertEqual(matrix.shape, (4, 4))

        # Check translation part
        np.testing.assert_array_almost_equal(matrix[0:3, 3:4], translation)

        # Check rotation part (90° around Z)
        expected_rotation = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        np.testing.assert_array_almost_equal(matrix[0:3, 0:3], expected_rotation, decimal=5)

        # Check bottom row
        np.testing.assert_array_almost_equal(matrix[3, :], [0, 0, 0, 1])

    def test_conversion_to_pose(self):
        """Test conversion to axis-angle Pose"""
        translation = np.array([[1.0], [2.0], [3.0]])
        axis = np.array([[0], [0], [1]])
        angle = np.pi / 2

        pose_quat = PoseQuat.from_translation_axis_angle(translation, axis, angle)
        pose_axis_angle = pose_quat.to_pose()

        # Check translation preserved
        np.testing.assert_array_almost_equal(pose_axis_angle.translation, translation)

        # Check rotation converts back correctly
        axis_angle_vec = pose_axis_angle.rotation
        recovered_angle = np.linalg.norm(axis_angle_vec)
        recovered_axis = axis_angle_vec / recovered_angle if recovered_angle > 1e-10 else axis

        self.assertAlmostEqual(recovered_angle, angle, places=5)
        np.testing.assert_array_almost_equal(recovered_axis, axis, decimal=5)

    def test_conversion_from_pose(self):
        """Test conversion from axis-angle Pose"""
        translation = np.array([[1.0], [2.0], [3.0]])
        rotation = np.array([[0], [0], [np.pi/2]])  # 90° around Z

        pose_axis_angle = Pose(translation=translation, rotation=rotation)
        pose_quat = pose_axis_angle.to_pose_quat()

        # Check translation preserved
        np.testing.assert_array_almost_equal(pose_quat.translation, translation)

        # Check rotation converts correctly
        expected_w = math.cos(np.pi / 4)
        expected_z = math.sin(np.pi / 4)
        self.assertAlmostEqual(pose_quat.quaternion.w, expected_w, places=5)
        self.assertAlmostEqual(pose_quat.quaternion.z, expected_z, places=5)

    def test_interpolate_slerp_endpoints(self):
        """Test SLERP interpolation at endpoints"""
        pose1 = PoseQuat.from_translation_axis_angle(
            np.array([[0], [0], [0]]),
            np.array([[0], [0], [1]]),
            0.0
        )
        pose2 = PoseQuat.from_translation_axis_angle(
            np.array([[10], [0], [0]]),
            np.array([[0], [0], [1]]),
            np.pi/2
        )

        # At t=0, should get pose1
        result_0 = pose1.interpolate(pose2, 0.0, method='slerp')
        np.testing.assert_array_almost_equal(result_0.translation, pose1.translation)
        self.assertAlmostEqual(result_0.quaternion.w, pose1.quaternion.w, places=5)

        # At t=1, should get pose2
        result_1 = pose1.interpolate(pose2, 1.0, method='slerp')
        np.testing.assert_array_almost_equal(result_1.translation, pose2.translation)
        self.assertAlmostEqual(result_1.quaternion.w, pose2.quaternion.w, places=5)

    def test_interpolate_slerp_middle(self):
        """Test SLERP interpolation at middle"""
        pose1 = PoseQuat.from_translation_axis_angle(
            np.array([[0], [0], [0]]),
            np.array([[0], [0], [1]]),
            0.0  # 0° rotation
        )
        pose2 = PoseQuat.from_translation_axis_angle(
            np.array([[10], [0], [0]]),
            np.array([[0], [0], [1]]),
            np.pi/2  # 90° rotation
        )

        # At t=0.5, should get middle pose
        result = pose1.interpolate(pose2, 0.5, method='slerp')

        # Translation should be midpoint
        expected_translation = np.array([[5], [0], [0]])
        np.testing.assert_array_almost_equal(result.translation, expected_translation)

        # Rotation should be 45° (middle of 0° to 90°)
        expected_quat = Quaternion.from_axis_angle(np.array([[0], [0], [1]]), np.pi/4)
        self.assertAlmostEqual(result.quaternion.w, expected_quat.w, places=4)
        self.assertAlmostEqual(result.quaternion.z, expected_quat.z, places=4)

    def test_interpolate_lerp(self):
        """Test LERP interpolation"""
        pose1 = PoseQuat.identity()
        pose2 = PoseQuat.from_translation_axis_angle(
            np.array([[10], [0], [0]]),
            np.array([[0], [0], [1]]),
            np.pi/2
        )

        result = pose1.interpolate(pose2, 0.5, method='lerp')

        # Check translation interpolated
        expected_translation = np.array([[5], [0], [0]])
        np.testing.assert_array_almost_equal(result.translation, expected_translation)

        # Quaternion should be interpolated (different from SLERP result)
        # Just check it's valid
        quat_norm = math.sqrt(result.quaternion.w**2 + result.quaternion.x**2 +
                              result.quaternion.y**2 + result.quaternion.z**2)
        self.assertAlmostEqual(quat_norm, 1.0, places=5)

    def test_interpolate_invalid_method(self):
        """Test interpolation with invalid method raises error"""
        pose1 = PoseQuat.identity()
        pose2 = PoseQuat.from_translation_axis_angle(
            np.array([[10], [0], [0]]),
            np.array([[0], [0], [1]]),
            np.pi/2
        )

        with self.assertRaises(ValueError):
            pose1.interpolate(pose2, 0.5, method='invalid')

    def test_repr(self):
        """Test string representation"""
        translation = np.array([[1.5], [2.5], [3.5]])
        quat = Quaternion.identity()
        pose = PoseQuat.from_translation_quaternion(translation, quat)

        repr_str = repr(pose)

        # Should contain translation values
        self.assertIn("1.500", repr_str)
        self.assertIn("2.500", repr_str)
        self.assertIn("3.500", repr_str)
        self.assertIn("PoseQuat", repr_str)

    def test_round_trip_conversion(self):
        """Test converting Pose -> PoseQuat -> Pose preserves values"""
        translation = np.array([[1.0], [2.0], [3.0]])
        rotation = np.array([[0.1], [0.2], [0.3]])

        pose1 = Pose(translation=translation, rotation=rotation)
        pose_quat = pose1.to_pose_quat()
        pose2 = pose_quat.to_pose()

        # Check translation preserved
        np.testing.assert_array_almost_equal(pose2.translation, translation, decimal=5)

        # Check rotation preserved
        np.testing.assert_array_almost_equal(pose2.rotation, rotation, decimal=5)


if __name__ == '__main__':
    unittest.main()
