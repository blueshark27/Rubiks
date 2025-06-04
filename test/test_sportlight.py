import numpy as np
import unittest
from src.datatypes.pose import Pose
from src.lights.base_light import LightPrimitive
from src.lights.spotlight import Spotlight

class TestSpotlight(unittest.TestCase):
    def test_spotlight_init(self):
        spotlight = Spotlight(light_type=LightPrimitive.SPOT,
                              diffuse=[0.9, 0.8, 0.7, 1.0],
                              spot_cutoff=45.0,
                              spot_exponent=2.0,
                              pose=Pose(pose_internal=(np.array([[0], [0], [0]]), np.array([[0], [0], [1]]))),
                              name='Spotlight0')

        self.assertEqual(spotlight.get_light_type(), LightPrimitive.SPOT)
        self.assertEqual(spotlight.diffuse, [0.9, 0.8, 0.7, 1.0])
        self.assertEqual(spotlight.spot_cutoff, 45.0)
        self.assertEqual(spotlight.spot_exponent, 2.0)
        self.assertEqual(spotlight.pose.get_translation().tolist(), [[0], [0], [0]])
        self.assertEqual(spotlight.pose.get_rotation().tolist(), [[0], [0], [1]])