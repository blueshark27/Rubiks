"""
Hierarchy Example - Demonstrates parent-child transformations

This example creates a parent sphere with two child spheres.
The parent rotates continuously, and the children orbit around it,
demonstrating hierarchical transformations.
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
    # Create window and OpenGL context FIRST
    window = init_window(800, 600, "Parent-Child Hierarchy Example")

    # Create parent sphere at origin
    parent_pose = Pose(
        translation=np.array([[0], [0], [0]]),
        rotation=np.array([[0], [0], [0]])
    )
    parent_sphere = Sphere(
        pose=parent_pose,
        radius=1.0,
        subdivision=2,
        name="Parent"
    )
    parent_wrapper = OpenGLPrimitivesWrapper(parent_sphere)

    # Create first child sphere (offset to the right)
    child1_pose = Pose(
        translation=np.array([[3], [0], [0]]),  # 3 units to the right
        rotation=np.array([[0], [0], [0]])
    )
    child1_sphere = Sphere(
        pose=child1_pose,
        radius=0.5,
        subdivision=2,
        scaling=Scaling(x=1.0, y=1.0, z=1.0),
        name="Child1"
    )
    child1_wrapper = OpenGLPrimitivesWrapper(child1_sphere)

    # Create second child sphere (offset to the left)
    child2_pose = Pose(
        translation=np.array([[-3], [0], [0]]),  # 3 units to the left
        rotation=np.array([[0], [0], [0]])
    )
    child2_sphere = Sphere(
        pose=child2_pose,
        radius=0.5,
        subdivision=2,
        scaling=Scaling(x=1.0, y=1.0, z=1.0),
        name="Child2"
    )
    child2_wrapper = OpenGLPrimitivesWrapper(child2_sphere)

    # Establish parent-child relationships
    parent_sphere.add_child(child1_sphere)
    parent_sphere.add_child(child2_sphere)

    print("Hierarchy established:")
    print(f"  Parent: {parent_sphere.get_name()}")
    print(f"  Children: {[child.get_name() for child in parent_sphere.get_children()]}")

    # Create and setup spotlight
    light_rotation = np.array([[np.pi/4], [0], [0]])  # Rotate 45° around X-axis
    light_pose = Pose(translation=np.array([[5], [5], [5]]), rotation=light_rotation)
    spotlight = Spotlight(
        light_type=LightPrimitive.SPOT,
        diffuse=[1.0, 1.0, 1.0, 1.0],
        spot_cutoff=45.0,
        spot_exponent=5.0,
        pose=light_pose,
        name="MainSpotlight"
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

    # Animation variables
    rotation_speed = 0.5  # radians per second

    print("\nAnimation started:")
    print("  - Parent sphere (large, center) rotates around Z-axis")
    print("  - Child spheres (small) orbit around parent")
    print("  - Close window to exit")

    while not glfw.window_should_close(window):
        glfw.poll_events()

        # Update parent rotation (rotate around Z-axis)
        current_time = glfw.get_time()
        angle = current_time * rotation_speed

        # Update parent pose
        new_parent_pose = Pose(
            translation=parent_pose.translation,
            rotation=np.array([[0], [0], [angle]])
        )
        parent_sphere.set_pose(new_parent_pose)

        # Clear buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Setup projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, 800/600, 0.1, 100.0)

        # Setup camera
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(8, 8, 8,   # Camera position (elevated to see hierarchy)
                  0, 0, 0,   # Look at origin
                  0, 1, 0)   # Up vector

        # Draw parent (white)
        glColor3f(1.0, 1.0, 1.0)
        parent_wrapper.draw()

        # Draw first child (red)
        glColor3f(1.0, 0.2, 0.2)
        child1_wrapper.draw()

        # Draw second child (blue)
        glColor3f(0.2, 0.2, 1.0)
        child2_wrapper.draw()

        glfw.swap_buffers(window)

    glfw.terminate()
    print("\nProgram terminated.")


if __name__ == "__main__":
    main()
