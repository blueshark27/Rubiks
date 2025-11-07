"""
Robot Arm Example - Hierarchical Transformations

This example creates a 3-segment robot arm with joints:
- Base (stationary)
  - Shoulder joint → Upper arm segment
    - Elbow joint → Forearm segment
      - Wrist joint → Hand

Each joint rotation affects all segments below it in the hierarchy.
"""

import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *

from src.primitives.sphere import Sphere
from src.primitives.cylinder import Cylinder
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
    window = init_window(1024, 768, "Robot Arm - Hierarchical Joint Control")

    # ===== Create Robot Arm Components =====

    # Base (stationary platform)
    base_pose = Pose(
        translation=np.array([[0], [0], [0]]),
        rotation=np.array([[0], [0], [0]])
    )
    base = Cylinder(
        pose=base_pose,
        radius=0.5,
        height=0.3,
        scaling=Scaling(x=1.0, y=1.0, z=1.0),
        name="Base"
    )
    base_wrapper = OpenGLPrimitivesWrapper(base)

    # Shoulder joint (rotates entire arm)
    shoulder_pose = Pose(
        translation=np.array([[0], [0.15], [0]]),  # On top of base
        rotation=np.array([[0], [0], [0]])
    )
    shoulder_joint = Sphere(
        pose=shoulder_pose,
        radius=0.3,
        subdivision=2,
        name="ShoulderJoint"
    )
    shoulder_wrapper = OpenGLPrimitivesWrapper(shoulder_joint)

    # Upper arm segment (from shoulder to elbow)
    upper_arm_pose = Pose(
        translation=np.array([[0], [1.5], [0]]),  # Extends upward from shoulder
        rotation=np.array([[np.pi/2], [0], [0]])  # Rotate to point upward
    )
    upper_arm = Cylinder(
        pose=upper_arm_pose,
        radius=0.2,
        height=3.0,
        name="UpperArm"
    )
    upper_arm_wrapper = OpenGLPrimitivesWrapper(upper_arm)

    # Elbow joint (rotates forearm)
    elbow_pose = Pose(
        translation=np.array([[0], [3.0], [0]]),  # At end of upper arm
        rotation=np.array([[0], [0], [0]])
    )
    elbow_joint = Sphere(
        pose=elbow_pose,
        radius=0.25,
        subdivision=2,
        name="ElbowJoint"
    )
    elbow_wrapper = OpenGLPrimitivesWrapper(elbow_joint)

    # Forearm segment (from elbow to wrist)
    forearm_pose = Pose(
        translation=np.array([[0], [1.2], [0]]),  # Extends from elbow
        rotation=np.array([[np.pi/2], [0], [0]])  # Rotate to point upward
    )
    forearm = Cylinder(
        pose=forearm_pose,
        radius=0.15,
        height=2.4,
        name="Forearm"
    )
    forearm_wrapper = OpenGLPrimitivesWrapper(forearm)

    # Wrist joint (rotates hand)
    wrist_pose = Pose(
        translation=np.array([[0], [2.4], [0]]),  # At end of forearm
        rotation=np.array([[0], [0], [0]])
    )
    wrist_joint = Sphere(
        pose=wrist_pose,
        radius=0.2,
        subdivision=2,
        name="WristJoint"
    )
    wrist_wrapper = OpenGLPrimitivesWrapper(wrist_joint)

    # Hand (end effector)
    hand_pose = Pose(
        translation=np.array([[0], [0.5], [0]]),  # Extends from wrist
        rotation=np.array([[0], [0], [0]])
    )
    hand = Sphere(
        pose=hand_pose,
        radius=0.3,
        subdivision=2,
        name="Hand"
    )
    hand_wrapper = OpenGLPrimitivesWrapper(hand)

    # ===== Establish Hierarchy =====
    # Base -> Shoulder -> UpperArm -> Elbow -> Forearm -> Wrist -> Hand

    base.add_child(shoulder_joint)
    shoulder_joint.add_child(upper_arm)
    shoulder_joint.add_child(elbow_joint)  # Elbow is child of shoulder
    elbow_joint.add_child(forearm)
    elbow_joint.add_child(wrist_joint)  # Wrist is child of elbow
    wrist_joint.add_child(hand)

    print("=== Robot Arm Hierarchy ===")
    print("Base (stationary)")
    print("  |-- ShoulderJoint (rotates around Z - entire arm swings)")
    print("      |-- UpperArm")
    print("      |-- ElbowJoint (rotates around Z - forearm bends)")
    print("          |-- Forearm")
    print("          |-- WristJoint (rotates around Y - hand twists)")
    print("              |-- Hand")
    print()
    print("Controls:")
    print("  - Shoulder rotates continuously")
    print("  - Elbow bends back and forth")
    print("  - Wrist twists")
    print()

    # Create lighting
    light_rotation = np.array([[np.pi/6], [0], [0]])
    light_pose = Pose(translation=np.array([[10], [10], [10]]), rotation=light_rotation)
    spotlight = Spotlight(
        light_type=LightPrimitive.SPOT,
        diffuse=[1.0, 1.0, 1.0, 1.0],
        spot_cutoff=60.0,
        spot_exponent=3.0,
        pose=light_pose,
        name="MainLight"
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

    # Animation speeds
    shoulder_speed = 0.3
    elbow_speed = 0.8
    wrist_speed = 1.5

    print("Animating... Close window to exit.")

    # Main render loop
    while not glfw.window_should_close(window):
        glfw.poll_events()

        current_time = glfw.get_time()

        # Animate shoulder (rotates entire arm around Z-axis)
        shoulder_angle = current_time * shoulder_speed
        shoulder_joint.set_pose(Pose(
            translation=shoulder_pose.translation,
            rotation=np.array([[0], [0], [shoulder_angle]])
        ))

        # Animate elbow (bends forearm back and forth)
        elbow_angle = np.sin(current_time * elbow_speed) * np.pi / 3  # ±60°
        elbow_joint.set_pose(Pose(
            translation=elbow_pose.translation,
            rotation=np.array([[0], [0], [elbow_angle]])
        ))

        # Animate wrist (twists hand around Y-axis)
        wrist_angle = np.sin(current_time * wrist_speed) * np.pi / 4  # ±45°
        wrist_joint.set_pose(Pose(
            translation=wrist_pose.translation,
            rotation=np.array([[0], [wrist_angle], [0]])
        ))

        # Clear buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Setup projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, 1024/768, 0.1, 100.0)

        # Setup camera
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(8, 4, 8,   # Camera position
                  0, 3, 0,   # Look at middle of arm
                  0, 1, 0)   # Up vector

        # Draw robot arm components

        # Base (dark gray)
        glColor3f(0.3, 0.3, 0.3)
        base_wrapper.draw()

        # Shoulder joint (red)
        glColor3f(1.0, 0.2, 0.2)
        shoulder_wrapper.draw()

        # Upper arm (silver)
        glColor3f(0.7, 0.7, 0.7)
        upper_arm_wrapper.draw()

        # Elbow joint (green)
        glColor3f(0.2, 1.0, 0.2)
        elbow_wrapper.draw()

        # Forearm (silver)
        glColor3f(0.7, 0.7, 0.7)
        forearm_wrapper.draw()

        # Wrist joint (blue)
        glColor3f(0.2, 0.2, 1.0)
        wrist_wrapper.draw()

        # Hand (yellow)
        glColor3f(1.0, 1.0, 0.2)
        hand_wrapper.draw()

        glfw.swap_buffers(window)

    glfw.terminate()
    print("\nProgram terminated.")


if __name__ == "__main__":
    main()
