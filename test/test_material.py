import unittest
import numpy as np
from src.datatypes.material import Material, MaterialPresets
from src.datatypes.pose import Pose
from src.datatypes.scaling import Scaling
from src.primitives.sphere import Sphere


class TestMaterial(unittest.TestCase):
    """Test Material class"""

    def test_material_initialization(self):
        """Test basic material initialization"""
        material = Material(
            ambient=[0.2, 0.0, 0.0, 1.0],
            diffuse=[1.0, 0.0, 0.0, 1.0],
            specular=[0.5, 0.5, 0.5, 1.0],
            shininess=32.0
        )
        self.assertIsInstance(material, Material)
        self.assertEqual(material.ambient, [0.2, 0.0, 0.0, 1.0])
        self.assertEqual(material.diffuse, [1.0, 0.0, 0.0, 1.0])
        self.assertEqual(material.specular, [0.5, 0.5, 0.5, 1.0])
        self.assertEqual(material.shininess, 32.0)

    def test_material_validation_color_components(self):
        """Test that color validation rejects invalid number of components"""
        with self.assertRaises(ValueError):
            Material(
                ambient=[0.2, 0.0, 0.0],  # Only 3 components
                diffuse=[1.0, 0.0, 0.0, 1.0],
                specular=[0.5, 0.5, 0.5, 1.0],
                shininess=32.0
            )

        with self.assertRaises(ValueError):
            Material(
                ambient=[0.2, 0.0, 0.0, 1.0],
                diffuse=[1.0, 0.0, 0.0, 1.0, 0.5],  # 5 components
                specular=[0.5, 0.5, 0.5, 1.0],
                shininess=32.0
            )

    def test_material_validation_color_range(self):
        """Test that color validation rejects out-of-range values"""
        with self.assertRaises(ValueError):
            Material(
                ambient=[0.2, 0.0, 1.5, 1.0],  # 1.5 is out of range
                diffuse=[1.0, 0.0, 0.0, 1.0],
                specular=[0.5, 0.5, 0.5, 1.0],
                shininess=32.0
            )

        with self.assertRaises(ValueError):
            Material(
                ambient=[0.2, 0.0, 0.0, 1.0],
                diffuse=[-0.1, 0.0, 0.0, 1.0],  # -0.1 is out of range
                specular=[0.5, 0.5, 0.5, 1.0],
                shininess=32.0
            )

    def test_material_validation_shininess(self):
        """Test that shininess validation rejects negative values"""
        with self.assertRaises(ValueError):
            Material(
                ambient=[0.2, 0.0, 0.0, 1.0],
                diffuse=[1.0, 0.0, 0.0, 1.0],
                specular=[0.5, 0.5, 0.5, 1.0],
                shininess=-5.0  # Negative shininess
            )

    def test_material_from_color(self):
        """Test Material.from_color factory method"""
        color = [1.0, 0.0, 0.0, 1.0]
        material = Material.from_color(color)

        # Check that diffuse matches input color
        self.assertEqual(material.diffuse, color)

        # Check that ambient is 20% of diffuse
        expected_ambient = [c * 0.2 for c in color]
        self.assertEqual(material.ambient, expected_ambient)

        # Check specular is default gray
        self.assertEqual(material.specular, [0.5, 0.5, 0.5, 1.0])

        # Check shininess is default
        self.assertEqual(material.shininess, 32.0)

    def test_material_from_color_custom_shininess(self):
        """Test Material.from_color with custom shininess"""
        color = [0.0, 1.0, 0.0, 1.0]
        shininess = 64.0
        material = Material.from_color(color, shininess=shininess)

        self.assertEqual(material.diffuse, color)
        self.assertEqual(material.shininess, shininess)

    def test_material_default(self):
        """Test Material.default factory method"""
        material = Material.default()
        self.assertIsInstance(material, Material)
        # Default should be gray
        self.assertEqual(material.diffuse, [0.8, 0.8, 0.8, 1.0])

    def test_material_color_presets(self):
        """Test Material color preset factory methods"""
        # Test red
        red = Material.red()
        self.assertEqual(red.diffuse, [1.0, 0.0, 0.0, 1.0])

        # Test green
        green = Material.green()
        self.assertEqual(green.diffuse, [0.0, 1.0, 0.0, 1.0])

        # Test blue
        blue = Material.blue()
        self.assertEqual(blue.diffuse, [0.0, 0.0, 1.0, 1.0])

        # Test yellow
        yellow = Material.yellow()
        self.assertEqual(yellow.diffuse, [1.0, 1.0, 0.0, 1.0])

        # Test cyan
        cyan = Material.cyan()
        self.assertEqual(cyan.diffuse, [0.0, 1.0, 1.0, 1.0])

        # Test magenta
        magenta = Material.magenta()
        self.assertEqual(magenta.diffuse, [1.0, 0.0, 1.0, 1.0])

        # Test white
        white = Material.white()
        self.assertEqual(white.diffuse, [1.0, 1.0, 1.0, 1.0])

        # Test black
        black = Material.black()
        self.assertEqual(black.diffuse, [0.0, 0.0, 0.0, 1.0])

    def test_material_getters(self):
        """Test Material getter methods"""
        material = Material(
            ambient=[0.2, 0.0, 0.0, 1.0],
            diffuse=[1.0, 0.0, 0.0, 1.0],
            specular=[0.5, 0.5, 0.5, 1.0],
            shininess=32.0
        )

        self.assertEqual(material.get_ambient(), [0.2, 0.0, 0.0, 1.0])
        self.assertEqual(material.get_diffuse(), [1.0, 0.0, 0.0, 1.0])
        self.assertEqual(material.get_specular(), [0.5, 0.5, 0.5, 1.0])
        self.assertEqual(material.get_shininess(), 32.0)

    def test_material_setters(self):
        """Test Material setter methods"""
        material = Material.default()

        # Set ambient
        new_ambient = [0.3, 0.0, 0.0, 1.0]
        material.set_ambient(new_ambient)
        self.assertEqual(material.ambient, new_ambient)

        # Set diffuse
        new_diffuse = [1.0, 0.5, 0.0, 1.0]
        material.set_diffuse(new_diffuse)
        self.assertEqual(material.diffuse, new_diffuse)

        # Set specular
        new_specular = [0.8, 0.8, 0.8, 1.0]
        material.set_specular(new_specular)
        self.assertEqual(material.specular, new_specular)

        # Set shininess
        new_shininess = 64.0
        material.set_shininess(new_shininess)
        self.assertEqual(material.shininess, new_shininess)

    def test_material_set_color(self):
        """Test Material.set_color convenience method"""
        material = Material.default()

        new_color = [0.0, 1.0, 0.0, 1.0]
        material.set_color(new_color)

        # Check diffuse matches new color
        self.assertEqual(material.diffuse, new_color)

        # Check ambient is 20% of new color
        expected_ambient = [c * 0.2 for c in new_color]
        self.assertEqual(material.ambient, expected_ambient)

    def test_material_copy(self):
        """Test Material.copy method"""
        original = Material.red()
        copy = original.copy()

        # Should be equal but not the same object
        self.assertIsNot(copy, original)
        self.assertEqual(copy.ambient, original.ambient)
        self.assertEqual(copy.diffuse, original.diffuse)
        self.assertEqual(copy.specular, original.specular)
        self.assertEqual(copy.shininess, original.shininess)

        # Modifying copy should not affect original
        copy.set_color([0.0, 1.0, 0.0, 1.0])
        self.assertNotEqual(copy.diffuse, original.diffuse)

    def test_material_repr(self):
        """Test Material string representation"""
        material = Material.red()
        repr_str = repr(material)
        self.assertIn("Material", repr_str)
        self.assertIn("diffuse", repr_str)
        self.assertIn("ambient", repr_str)
        self.assertIn("specular", repr_str)
        self.assertIn("shininess", repr_str)


class TestMaterialPresets(unittest.TestCase):
    """Test MaterialPresets class"""

    def test_emerald_preset(self):
        """Test emerald material preset"""
        material = MaterialPresets.emerald()
        self.assertIsInstance(material, Material)
        self.assertEqual(material.shininess, 76.8)

    def test_jade_preset(self):
        """Test jade material preset"""
        material = MaterialPresets.jade()
        self.assertIsInstance(material, Material)
        self.assertEqual(material.shininess, 12.8)

    def test_ruby_preset(self):
        """Test ruby material preset"""
        material = MaterialPresets.ruby()
        self.assertIsInstance(material, Material)
        self.assertEqual(material.shininess, 76.8)

    def test_gold_preset(self):
        """Test gold material preset"""
        material = MaterialPresets.gold()
        self.assertIsInstance(material, Material)
        self.assertEqual(material.shininess, 51.2)

    def test_silver_preset(self):
        """Test silver material preset"""
        material = MaterialPresets.silver()
        self.assertIsInstance(material, Material)
        self.assertEqual(material.shininess, 51.2)

    def test_bronze_preset(self):
        """Test bronze material preset"""
        material = MaterialPresets.bronze()
        self.assertIsInstance(material, Material)
        self.assertEqual(material.shininess, 25.6)

    def test_chrome_preset(self):
        """Test chrome material preset"""
        material = MaterialPresets.chrome()
        self.assertIsInstance(material, Material)
        self.assertEqual(material.shininess, 76.8)

    def test_plastic_red_preset(self):
        """Test plastic red material preset"""
        material = MaterialPresets.plastic_red()
        self.assertIsInstance(material, Material)
        self.assertEqual(material.diffuse, [1.0, 0.0, 0.0, 1.0])

    def test_plastic_green_preset(self):
        """Test plastic green material preset"""
        material = MaterialPresets.plastic_green()
        self.assertIsInstance(material, Material)
        self.assertEqual(material.diffuse, [0.0, 1.0, 0.0, 1.0])

    def test_plastic_blue_preset(self):
        """Test plastic blue material preset"""
        material = MaterialPresets.plastic_blue()
        self.assertIsInstance(material, Material)
        self.assertEqual(material.diffuse, [0.0, 0.0, 1.0, 1.0])

    def test_rubber_black_preset(self):
        """Test rubber black material preset"""
        material = MaterialPresets.rubber_black()
        self.assertIsInstance(material, Material)
        self.assertEqual(material.shininess, 10.0)


class TestMaterialIntegration(unittest.TestCase):
    """Test Material integration with primitives"""

    def test_sphere_with_material(self):
        """Test creating a sphere with a material"""
        material = Material.red()
        pose = Pose(translation=np.array([[0.0], [0.0], [0.0]]),
                   rotation=np.array([[0.0], [0.0], [0.0]]))
        sphere = Sphere(pose=pose, radius=1.0, material=material)

        retrieved_material = sphere.get_material()
        self.assertEqual(retrieved_material.diffuse, material.diffuse)

    def test_sphere_with_color(self):
        """Test creating a sphere with color convenience parameter"""
        color = [0.0, 1.0, 0.0, 1.0]
        pose = Pose(translation=np.array([[0.0], [0.0], [0.0]]),
                   rotation=np.array([[0.0], [0.0], [0.0]]))
        sphere = Sphere(pose=pose, radius=1.0, color=color)

        material = sphere.get_material()
        self.assertEqual(material.diffuse, color)

        # Ambient should be 20% of color
        expected_ambient = [c * 0.2 for c in color]
        self.assertEqual(material.ambient, expected_ambient)

    def test_sphere_material_priority(self):
        """Test that material parameter takes priority over color parameter"""
        color = [0.0, 1.0, 0.0, 1.0]
        material = Material.red()
        pose = Pose(translation=np.array([[0.0], [0.0], [0.0]]),
                   rotation=np.array([[0.0], [0.0], [0.0]]))
        sphere = Sphere(pose=pose, radius=1.0, material=material, color=color)

        retrieved_material = sphere.get_material()
        # Should use material (red), not color (green)
        self.assertEqual(retrieved_material.diffuse, [1.0, 0.0, 0.0, 1.0])

    def test_sphere_default_material(self):
        """Test that sphere gets default material when none specified"""
        pose = Pose(translation=np.array([[0.0], [0.0], [0.0]]),
                   rotation=np.array([[0.0], [0.0], [0.0]]))
        sphere = Sphere(pose=pose, radius=1.0)

        material = sphere.get_material()
        self.assertIsInstance(material, Material)
        # Should be default gray
        self.assertEqual(material.diffuse, [0.8, 0.8, 0.8, 1.0])

    def test_sphere_set_material(self):
        """Test changing sphere material after creation"""
        pose = Pose(translation=np.array([[0.0], [0.0], [0.0]]),
                   rotation=np.array([[0.0], [0.0], [0.0]]))
        sphere = Sphere(pose=pose, radius=1.0, material=Material.red())

        # Change to blue
        blue_material = Material.blue()
        sphere.set_material(blue_material)

        retrieved_material = sphere.get_material()
        self.assertEqual(retrieved_material.diffuse, [0.0, 0.0, 1.0, 1.0])

    def test_sphere_set_color(self):
        """Test changing sphere color after creation"""
        pose = Pose(translation=np.array([[0.0], [0.0], [0.0]]),
                   rotation=np.array([[0.0], [0.0], [0.0]]))
        sphere = Sphere(pose=pose, radius=1.0, material=Material.red())

        # Change to green using convenience method
        new_color = [0.0, 1.0, 0.0, 1.0]
        sphere.set_color(new_color)

        material = sphere.get_material()
        self.assertEqual(material.diffuse, new_color)

        # Ambient should be updated too
        expected_ambient = [c * 0.2 for c in new_color]
        self.assertEqual(material.ambient, expected_ambient)

    def test_material_preset_integration(self):
        """Test using material presets with primitives"""
        pose = Pose(translation=np.array([[0.0], [0.0], [0.0]]),
                   rotation=np.array([[0.0], [0.0], [0.0]]))

        # Test with gold preset
        sphere = Sphere(pose=pose, radius=1.0, material=MaterialPresets.gold())
        material = sphere.get_material()
        self.assertEqual(material.shininess, 51.2)

        # Test changing to emerald preset
        sphere.set_material(MaterialPresets.emerald())
        material = sphere.get_material()
        self.assertEqual(material.shininess, 76.8)


if __name__ == '__main__':
    unittest.main()
