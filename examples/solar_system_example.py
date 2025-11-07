"""
Solar System Example - Deep Hierarchy Demonstration

This example creates a solar system with a 3-level hierarchy:
- Sun (root, stationary)
  - Earth (orbits around Sun)
    - Moon (orbits around Earth)
  - Mars (orbits around Sun)

Demonstrates how transformations propagate through multiple levels.
The Moon's world position is affected by both Earth's and its own rotation.
"""

import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *

from src.primitives.sphere import Sphere
from src.primitives.opengl_primitives_wrapper import OpenGLPrimitivesWrapper
from src.datatypes.pose import Pose
from src.datatypes.scaling import Scaling
from src.lights.spotlight import Spotlight
from src.lights.opengl_light_wrapper import OpenGLLightWrapper
from src.lights.base_light import LightPrimitive


def init_window(width, height, title):
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
    window = init_window(1024, 768, "Solar System - Deep Hierarchy Example")

    # ===== Create Solar System Hierarchy =====

    # Sun (root - center of solar system)
    sun_pose = Pose(
        translation=np.array([[0], [0], [0]]),
        rotation=np.array([[0], [0], [0]])
    )
    sun = Sphere(
        pose=sun_pose,
        radius=2.0,  # Large sun
        subdivision=3,
        name="Sun"
    )
    sun_wrapper = OpenGLPrimitivesWrapper(sun)

    # Earth (orbits Sun at distance 8)
    earth_pose = Pose(
        translation=np.array([[8], [0], [0]]),  # 8 units from Sun
        rotation=np.array([[0], [0], [0]])
    )
    earth = Sphere(
        pose=earth_pose,
        radius=0.8,
        subdivision=2,
        name="Earth"
    )
    earth_wrapper = OpenGLPrimitivesWrapper(earth)

    # Moon (orbits Earth at distance 2)
    moon_pose = Pose(
        translation=np.array([[2], [0], [0]]),  # 2 units from Earth
        rotation=np.array([[0], [0], [0]])
    )
    moon = Sphere(
        pose=moon_pose,
        radius=0.3,
        subdivision=2,
        name="Moon"
    )
    moon_wrapper = OpenGLPrimitivesWrapper(moon)

    # Mars (orbits Sun at distance 12)
    mars_pose = Pose(
        translation=np.array([[12], [0], [0]]),  # 12 units from Sun
        rotation=np.array([[0], [0], [0]])
    )
    mars = Sphere(
        pose=mars_pose,
        radius=0.6,
        subdivision=2,
        name="Mars"
    )
    mars_wrapper = OpenGLPrimitivesWrapper(mars)

    # Establish hierarchy
    sun.add_child(earth)  # Earth is child of Sun
    earth.add_child(moon)  # Moon is child of Earth (grandchild of Sun!)
    sun.add_child(mars)   # Mars is child of Sun

    # Print hierarchy structure
    print("=== Solar System Hierarchy ===")
    print(f"Sun (root)")
    print(f"  |-- Earth")
    print(f"      |-- Moon")
    print(f"  |-- Mars")
    print()
    print(f"Verification:")
    print(f"  Sun's children: {[c.get_name() for c in sun.get_children()]}")
    print(f"  Earth's children: {[c.get_name() for c in earth.get_children()]}")
    print(f"  Moon's parent: {moon.get_parent().get_name()}")
    print(f"  Moon's grandparent: {moon.get_parent().get_parent().get_name()}")
    print()

    # Create lighting
    light_rotation = np.array([[0], [0], [0]])
    light_pose = Pose(translation=np.array([[10], [10], [10]]), rotation=light_rotation)
    spotlight = Spotlight(
        light_type=LightPrimitive.SPOT,
        diffuse=[1.0, 1.0, 1.0, 1.0],
        spot_cutoff=60.0,
        spot_exponent=3.0,
        pose=light_pose,
        name="SunLight"
    )
    lightWrapper = OpenGLLightWrapper(spotlight, index=0)

    # Enable OpenGL features
    glEnable(GL_LIGHTING)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)
    glEnable(GL_NORMALIZE)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    # Setup the light
    lightWrapper.setup_lighting()

    # Set material properties
    glMaterialfv(GL_FRONT, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
    glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
    glMaterialfv(GL_FRONT, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
    glMaterialf(GL_FRONT, GL_SHININESS, 50.0)

    # Animation speeds (radians per second)
    earth_orbit_speed = 0.3   # Earth orbits Sun
    moon_orbit_speed = 1.0    # Moon orbits Earth (faster)
    mars_orbit_speed = 0.2    # Mars orbits Sun (slower than Earth)
    sun_rotation_speed = 0.1  # Sun rotates on its axis

    print("Animation:")
    print("  - Sun: Yellow (rotates slowly)")
    print("  - Earth: Blue (orbits Sun)")
    print("  - Moon: White (orbits Earth - 3-level hierarchy!)")
    print("  - Mars: Red (orbits Sun slower than Earth)")
    print()
    print("Notice how Moon's motion combines:")
    print("  1. Its own orbit around Earth")
    print("  2. Earth's orbit around Sun")
    print("  = Complex path showing hierarchical transforms!")
    print()
    print("Close window to exit...")

    # Main render loop
    while not glfw.window_should_close(window):
        glfw.poll_events()

        current_time = glfw.get_time()

        # Update Sun rotation (rotates on its own axis)
        sun_angle = current_time * sun_rotation_speed
        sun.set_pose(Pose(
            translation=sun_pose.translation,
            rotation=np.array([[0], [sun_angle], [0]])  # Rotate around Y-axis
        ))

        # Update Earth orbit (rotates around Sun on Z-axis)
        earth_angle = current_time * earth_orbit_speed
        earth.set_pose(Pose(
            translation=earth_pose.translation,  # Fixed distance from Sun
            rotation=np.array([[0], [0], [earth_angle]])  # Orbit rotation
        ))

        # Update Moon orbit (rotates around Earth on Z-axis)
        moon_angle = current_time * moon_orbit_speed
        moon.set_pose(Pose(
            translation=moon_pose.translation,  # Fixed distance from Earth
            rotation=np.array([[0], [0], [moon_angle]])  # Orbit rotation
        ))

        # Update Mars orbit (rotates around Sun on Z-axis)
        mars_angle = current_time * mars_orbit_speed
        mars.set_pose(Pose(
            translation=mars_pose.translation,  # Fixed distance from Sun
            rotation=np.array([[0], [0], [mars_angle]])  # Orbit rotation
        ))

        # Clear buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Setup projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, 1024/768, 0.1, 100.0)

        # Setup camera (elevated view to see the system)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(15, 15, 15,  # Camera position
                  0, 0, 0,      # Look at origin (Sun)
                  0, 1, 0)      # Up vector

        # Draw Sun (yellow)
        glColor3f(1.0, 1.0, 0.0)  # Yellow
        sun_wrapper.draw()

        # Draw Earth (blue)
        glColor3f(0.2, 0.4, 1.0)  # Blue
        earth_wrapper.draw()

        # Draw Moon (white/gray)
        glColor3f(0.8, 0.8, 0.8)  # Light gray
        moon_wrapper.draw()

        # Draw Mars (red)
        glColor3f(1.0, 0.3, 0.2)  # Red
        mars_wrapper.draw()

        glfw.swap_buffers(window)

    glfw.terminate()
    print("\nProgram terminated.")


if __name__ == "__main__":
    main()
