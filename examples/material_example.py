"""
Material System Example

This example demonstrates the material system for 3D objects:
- Using predefined color materials (red, green, blue, etc.)
- Using MaterialPresets for realistic appearances (gold, silver, emerald, etc.)
- Creating custom materials with specific Phong lighting properties
- Changing materials dynamically

The example displays 9 spheres in a 3x3 grid, each with a different material.
"""

import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *

from src.primitives.sphere import Sphere
from src.primitives.opengl_primitives_wrapper import OpenGLPrimitivesWrapper
from src.datatypes.pose import Pose
from src.datatypes.material import Material, MaterialPresets
from src.lights.spotlight import Spotlight
from src.lights.opengl_light_wrapper import OpenGLLightWrapper
from src.lights.base_light import LightPrimitive


def init_window(width, height, title):
    """Initialize GLFW window"""
    if not glfw.init():
        raise Exception("GLFW could not be initialized!")

    window = glfw.create_window(width, height, title, None, None)
    if not window:
        glfw.terminate()
        raise Exception("Window could not be created!")

    glfw.make_context_current(window)
    return window


def main():
    # Create window and OpenGL context
    window = init_window(1200, 900, "Material System Example")

    # Create spheres with different materials in a 3x3 grid
    spheres = []
    materials = [
        ("Red Plastic", Material.red()),
        ("Gold Metal", MaterialPresets.gold()),
        ("Blue Plastic", Material.blue()),
        ("Silver Metal", MaterialPresets.silver()),
        ("Emerald Gem", MaterialPresets.emerald()),
        ("Magenta Plastic", Material.magenta()),
        ("Bronze Metal", MaterialPresets.bronze()),
        ("Jade Gem", MaterialPresets.jade()),
        ("Ruby Gem", MaterialPresets.ruby()),
    ]

    # Create spheres in a 3x3 grid
    grid_spacing = 2.5
    for i, (name, material) in enumerate(materials):
        row = i // 3
        col = i % 3
        x = (col - 1) * grid_spacing
        y = (row - 1) * grid_spacing
        z = 0

        pose = Pose(
            translation=np.array([[x], [y], [z]]),
            rotation=np.array([[0.0], [0.0], [0.0]])
        )
        sphere = Sphere(pose=pose, radius=0.8, material=material, name=name)
        spheres.append((sphere, name))

    # Create OpenGL wrappers for all spheres
    sphere_wrappers = [OpenGLPrimitivesWrapper(sphere) for sphere, _ in spheres]

    # Create multiple lights for better illumination
    # Main spotlight from top-front - positioned to illuminate all spheres
    light1_pose = Pose(
        translation=np.array([[0], [8], [8]]),
        rotation=np.array([[0.0], [0.0], [0.0]])
    )
    spotlight1 = Spotlight(
        light_type=LightPrimitive.SPOT,
        diffuse=[1.2, 1.2, 1.2, 1.0],  # Brighter
        spot_cutoff=90.0,  # Wider coverage
        spot_exponent=1.0,  # Less focused, more spread
        pose=light1_pose,
        name="MainSpotlight"
    )

    # Secondary fill light from the side
    light2_pose = Pose(
        translation=np.array([[-8], [2], [5]]),
        rotation=np.array([[0.0], [0.0], [0.0]])
    )
    spotlight2 = Spotlight(
        light_type=LightPrimitive.SPOT,
        diffuse=[0.8, 0.8, 0.8, 1.0],
        spot_cutoff=90.0,  # Wider coverage
        spot_exponent=1.0,  # Less focused
        pose=light2_pose,
        name="FillLight"
    )

    lightWrapper1 = OpenGLLightWrapper(spotlight1, index=0)
    lightWrapper2 = OpenGLLightWrapper(spotlight2, index=1)

    # Enable OpenGL features
    glEnable(GL_LIGHTING)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)
    glEnable(GL_NORMALIZE)

    # Set background color (dark blue-gray for better contrast)
    glClearColor(0.1, 0.1, 0.15, 1.0)

    # Set global ambient light - this is crucial for visibility!
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.4, 0.4, 0.4, 1.0])

    # Setup lights
    lightWrapper1.setup_lighting()
    lightWrapper2.setup_lighting()

    # Camera rotation angle
    angle = 0.0

    print("\n" + "="*60)
    print("MATERIAL SYSTEM EXAMPLE")
    print("="*60)
    print("\nDisplaying 9 spheres with different materials:")
    for i, (_, name) in enumerate(spheres):
        print(f"  {i+1}. {name}")
    print("\nControls:")
    print("  - The camera automatically rotates around the scene")
    print("  - Press ESC or close window to exit")
    print("="*60 + "\n")

    # Main rendering loop
    while not glfw.window_should_close(window):
        glfw.poll_events()

        # Clear buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Setup projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, 1200/900, 0.1, 100.0)

        # Setup camera with automatic rotation
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Rotate camera around the scene
        angle += 0.3
        camera_distance = 12.0
        camera_x = camera_distance * np.sin(np.radians(angle))
        camera_z = camera_distance * np.cos(np.radians(angle))
        camera_y = 3.0

        gluLookAt(
            camera_x, camera_y, camera_z,  # Camera position
            0, 0, 0,  # Look at center
            0, 1, 0   # Up vector
        )

        # Draw all spheres
        for wrapper in sphere_wrappers:
            wrapper.draw()

        glfw.swap_buffers(window)

        # Check for ESC key
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            break

    glfw.terminate()
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
