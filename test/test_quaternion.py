"""
Unit tests for quaternion operations.
"""

import unittest
import numpy as np
import math

from src.datatypes.quaternion import Quaternion, slerp, lerp


class TestQuaternion(unittest.TestCase):

    def test_identity_quaternion(self):
        """Test identity quaternion creation"""
        q = Quaternion.identity()
        self.assertAlmostEqual(q.w, 1.0)
        self.assertAlmostEqual(q.x, 0.0)
        self.assertAlmostEqual(q.y, 0.0)
        self.assertAlmostEqual(q.z, 0.0)

    def test_quaternion_creation(self):
        """Test basic quaternion creation"""
        q = Quaternion(w=0.5, x=0.5, y=0.5, z=0.5)
        self.assertAlmostEqual(q.w, 0.5)
        self.assertAlmostEqual(q.x, 0.5)
        self.assertAlmostEqual(q.y, 0.5)
        self.assertAlmostEqual(q.z, 0.5)

    def test_from_axis_angle_90deg_z(self):
        """Test quaternion from 90° rotation around Z-axis"""
        axis = np.array([[0], [0], [1]])
        angle = np.pi / 2  # 90 degrees

        q = Quaternion.from_axis_angle(axis, angle)

        # For 90° around Z: q = cos(45°) + sin(45°)k
        expected_w = math.cos(np.pi / 4)
        expected_z = math.sin(np.pi / 4)

        self.assertAlmostEqual(q.w, expected_w, places=5)
        self.assertAlmostEqual(q.x, 0.0, places=5)
        self.assertAlmostEqual(q.y, 0.0, places=5)
        self.assertAlmostEqual(q.z, expected_z, places=5)

    def test_from_axis_angle_vector(self):
        """Test quaternion from axis-angle vector"""
        # 90° around Z-axis
        axis_angle = np.array([[0], [0], [np.pi/2]])
        q = Quaternion.from_axis_angle_vector(axis_angle)

        expected_w = math.cos(np.pi / 4)
        expected_z = math.sin(np.pi / 4)

        self.assertAlmostEqual(q.w, expected_w, places=5)
        self.assertAlmostEqual(q.z, expected_z, places=5)

    def test_to_axis_angle(self):
        """Test conversion to axis-angle"""
        # Create quaternion for 90° around Z
        axis_in = np.array([[0], [0], [1]])
        angle_in = np.pi / 2

        q = Quaternion.from_axis_angle(axis_in, angle_in)
        axis_out, angle_out = q.to_axis_angle()

        # Check angle
        self.assertAlmostEqual(angle_out, angle_in, places=5)

        # Check axis (should be [0, 0, 1])
        np.testing.assert_array_almost_equal(axis_out, axis_in, decimal=5)

    def test_to_rotation_matrix_identity(self):
        """Test identity quaternion gives identity matrix"""
        q = Quaternion.identity()
        R = q.to_rotation_matrix()
        np.testing.assert_array_almost_equal(R, np.eye(3))

    def test_to_rotation_matrix_90deg_z(self):
        """Test rotation matrix for 90° around Z"""
        q = Quaternion.from_axis_angle(np.array([[0], [0], [1]]), np.pi/2)
        R = q.to_rotation_matrix()

        # Expected rotation matrix for 90° around Z
        expected = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])

        np.testing.assert_array_almost_equal(R, expected, decimal=5)

    def test_normalize(self):
        """Test quaternion normalization"""
        q = Quaternion(w=1.0, x=1.0, y=1.0, z=1.0)
        q_norm = q.normalize()

        # Check magnitude is 1
        magnitude = math.sqrt(q_norm.w**2 + q_norm.x**2 + q_norm.y**2 + q_norm.z**2)
        self.assertAlmostEqual(magnitude, 1.0, places=10)

    def test_conjugate(self):
        """Test quaternion conjugate"""
        q = Quaternion(w=0.5, x=0.5, y=0.5, z=0.5)
        q_conj = q.conjugate()

        self.assertAlmostEqual(q_conj.w, 0.5)
        self.assertAlmostEqual(q_conj.x, -0.5)
        self.assertAlmostEqual(q_conj.y, -0.5)
        self.assertAlmostEqual(q_conj.z, -0.5)

    def test_inverse(self):
        """Test quaternion inverse"""
        q = Quaternion.from_axis_angle(np.array([[0], [0], [1]]), np.pi/4)
        q_inv = q.inverse()

        # q * q^-1 should give identity
        result = q * q_inv

        self.assertAlmostEqual(result.w, 1.0, places=5)
        self.assertAlmostEqual(result.x, 0.0, places=5)
        self.assertAlmostEqual(result.y, 0.0, places=5)
        self.assertAlmostEqual(result.z, 0.0, places=5)

    def test_multiplication(self):
        """Test quaternion multiplication"""
        # Two 90° rotations around Z should give 180°
        q1 = Quaternion.from_axis_angle(np.array([[0], [0], [1]]), np.pi/2)
        q2 = Quaternion.from_axis_angle(np.array([[0], [0], [1]]), np.pi/2)

        result = q1 * q2

        # Should equal 180° rotation around Z
        expected = Quaternion.from_axis_angle(np.array([[0], [0], [1]]), np.pi)

        self.assertAlmostEqual(result.w, expected.w, places=5)
        self.assertAlmostEqual(result.x, expected.x, places=5)
        self.assertAlmostEqual(result.y, expected.y, places=5)
        self.assertAlmostEqual(result.z, expected.z, places=5)

    def test_rotate_vector(self):
        """Test vector rotation"""
        # 90° rotation around Z-axis
        q = Quaternion.from_axis_angle(np.array([[0], [0], [1]]), np.pi/2)

        # Rotate vector (1, 0, 0) by 90° around Z
        vector = np.array([[1], [0], [0]])
        rotated = q.rotate_vector(vector)

        # Should give (0, 1, 0)
        expected = np.array([[0], [1], [0]])

        np.testing.assert_array_almost_equal(rotated, expected, decimal=5)

    def test_dot_product(self):
        """Test quaternion dot product"""
        q1 = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
        q2 = Quaternion(w=0.0, x=1.0, y=0.0, z=0.0)

        dot = q1.dot(q2)
        self.assertAlmostEqual(dot, 0.0)

        q3 = Quaternion(w=0.5, x=0.5, y=0.5, z=0.5)
        dot2 = q3.dot(q3)
        self.assertAlmostEqual(dot2, 1.0)

    def test_slerp_start(self):
        """Test SLERP at t=0 returns start quaternion"""
        q1 = Quaternion.from_axis_angle(np.array([[0], [0], [1]]), 0.0)
        q2 = Quaternion.from_axis_angle(np.array([[0], [0], [1]]), np.pi/2)

        result = slerp(q1, q2, 0.0)

        self.assertAlmostEqual(result.w, q1.w, places=5)
        self.assertAlmostEqual(result.x, q1.x, places=5)
        self.assertAlmostEqual(result.y, q1.y, places=5)
        self.assertAlmostEqual(result.z, q1.z, places=5)

    def test_slerp_end(self):
        """Test SLERP at t=1 returns end quaternion"""
        q1 = Quaternion.from_axis_angle(np.array([[0], [0], [1]]), 0.0)
        q2 = Quaternion.from_axis_angle(np.array([[0], [0], [1]]), np.pi/2)

        result = slerp(q1, q2, 1.0)

        self.assertAlmostEqual(result.w, q2.w, places=5)
        self.assertAlmostEqual(result.x, q2.x, places=5)
        self.assertAlmostEqual(result.y, q2.y, places=5)
        self.assertAlmostEqual(result.z, q2.z, places=5)

    def test_slerp_middle(self):
        """Test SLERP at t=0.5 gives middle rotation"""
        # 0° to 90° around Z, middle should be 45°
        q1 = Quaternion.from_axis_angle(np.array([[0], [0], [1]]), 0.0)
        q2 = Quaternion.from_axis_angle(np.array([[0], [0], [1]]), np.pi/2)

        result = slerp(q1, q2, 0.5)
        expected = Quaternion.from_axis_angle(np.array([[0], [0], [1]]), np.pi/4)

        self.assertAlmostEqual(result.w, expected.w, places=4)
        self.assertAlmostEqual(result.z, expected.z, places=4)

    def test_lerp(self):
        """Test linear interpolation"""
        q1 = Quaternion.identity()
        q2 = Quaternion.from_axis_angle(np.array([[0], [0], [1]]), np.pi/2)

        # Test at various t values
        result_0 = lerp(q1, q2, 0.0)
        result_1 = lerp(q1, q2, 1.0)

        # At t=0, should be close to q1
        self.assertAlmostEqual(result_0.w, q1.w, places=4)

        # At t=1, should be close to q2
        self.assertAlmostEqual(result_1.w, q2.w, places=4)
        self.assertAlmostEqual(result_1.z, q2.z, places=4)

    def test_from_euler_identity(self):
        """Test Euler angles (0,0,0) gives identity"""
        q = Quaternion.from_euler(0.0, 0.0, 0.0)
        self.assertAlmostEqual(q.w, 1.0, places=5)
        self.assertAlmostEqual(q.x, 0.0, places=5)
        self.assertAlmostEqual(q.y, 0.0, places=5)
        self.assertAlmostEqual(q.z, 0.0, places=5)

    def test_euler_round_trip(self):
        """Test converting to/from Euler angles"""
        roll_in = 0.3
        pitch_in = 0.5
        yaw_in = 0.7

        q = Quaternion.from_euler(roll_in, pitch_in, yaw_in)
        roll_out, pitch_out, yaw_out = q.to_euler()

        self.assertAlmostEqual(roll_out, roll_in, places=5)
        self.assertAlmostEqual(pitch_out, pitch_in, places=5)
        self.assertAlmostEqual(yaw_out, yaw_in, places=5)


if __name__ == '__main__':
    unittest.main()
