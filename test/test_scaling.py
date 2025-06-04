import numpy as np
from src.datatypes.scaling import Scaling
import unittest


class TestScaling(unittest.TestCase):
    def test_scaling_initialization(self):
        # Test initialization with valid input
        scaling = Scaling(x=1.0, y=2.0, z=3.0)
        self.assertTrue(isinstance(scaling, Scaling))

    def test_scaling_initialization_invalid(self):
        # Test initialization with invalid input
        with self.assertRaises(ValueError):
            Scaling(x=1.0, y=2.0, z="invalid_z")
        with self.assertRaises(ValueError):
            Scaling(x=1.0, y="invalid_y", z=3.0)
        with self.assertRaises(ValueError):
            Scaling(x="invalid_x", y=2.0, z=3.0)

    def test_get_scaling(self):
        scaling = Scaling(x=1.0, y=2.0, z=3.0)
        expected_scaling = np.array([[1.0], [2.0], [3.0]])
        np.testing.assert_array_equal(scaling.get_scaling(), expected_scaling)

    def test_set_scaling(self):
        scaling = Scaling(x=1.0, y=2.0, z=3.0)
        scaling.set_scaling(np.array([[4.0], [5.0], [6.0]]))
        self.assertEqual(scaling.x, 4.0)
        self.assertEqual(scaling.y, 5.0)
        self.assertEqual(scaling.z, 6.0)