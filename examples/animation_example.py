"""
Animation System Example

This example demonstrates the keyframe-based animation system:
1. Creating animations with keyframes
2. Smooth interpolation (SLERP) for rotations
3. AnimationPlayer for playback control
4. Animating multiple objects with AnimationClip

The scene shows:
- A sun that rotates in place
- A planet that orbits the sun while rotating
- A moon that orbits the planet

All objects are animated using keyframes with smooth interpolation.
"""

import numpy as np
import time
import math
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *

from src.primitives.sphere import Sphere
from src.primitives.opengl_primitives_wrapper import OpenGLPrimitivesWrapper
from src.datatypes.pose import Pose, PoseQuat
from src.datatypes.quaternion import Quaternion
from src.animation.animation import Animation, AnimationClip, AnimationPlayer
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
    print("=" * 70)
    print("ANIMATION SYSTEM EXAMPLE")
    print("=" * 70)
    print()
    print("This example demonstrates the keyframe animation system.")
    print("Watch as objects smoothly animate through their keyframes!")
    print()
    print("Features:")
    print("  - Keyframe-based animation")
    print("  - SLERP interpolation for smooth rotations")
    print("  - AnimationPlayer for playback control")
    print("  - Multiple objects animated in sync")
    print()
    print("=" * 70)
    print()

    # Create window and OpenGL context
    window = init_window(800, 600, "Animation System Demo")

    # Configure OpenGL
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glClearColor(0.1, 0.1, 0.15, 1.0)

    # Set up projection
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 800.0 / 600.0, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

    # Set up lighting
    spotlight_pose = Pose(
        translation=np.array([[5], [5], [10]]),
        rotation=np.array([[0], [0], [0]])
    )
    spotlight = Spotlight(
        light_type=LightPrimitive.SPOT,
        pose=spotlight_pose,
        name="Spotlight",
        diffuse=[0.8, 0.8, 0.8, 1.0],
        spot_cutoff=45.0,
        spot_exponent=2.0
    )
    light_wrapper = OpenGLLightWrapper(spotlight, index=0)
    light_wrapper.setup_lighting()

    # Create initial poses
    sun_init_pose = Pose(
        translation=np.array([[0], [0], [0]]),
        rotation=np.array([[0], [0], [0]])
    )

    planet_init_pose = Pose(
        translation=np.array([[3], [0], [0]]),
        rotation=np.array([[0], [0], [0]])
    )

    moon_init_pose = Pose(
        translation=np.array([[0.8], [0], [0]]),
        rotation=np.array([[0], [0], [0]])
    )

    # Create objects
    sun = Sphere(
        pose=sun_init_pose,
        name="Sun",
        radius=1.0,
        subdivision=0
    )
    sun_wrapper = OpenGLPrimitivesWrapper(sun, color=[1.0, 1.0, 0.0, 1.0])

    planet = Sphere(
        pose=planet_init_pose,
        name="Planet",
        radius=0.3,
        subdivision=0,
        parent=sun
    )
    planet_wrapper = OpenGLPrimitivesWrapper(planet, color=[0.0, 0.5, 1.0, 1.0])

    moon = Sphere(
        pose=moon_init_pose,
        name="Moon",
        radius=0.1,
        subdivision=0,
        parent=planet
    )
    moon_wrapper = OpenGLPrimitivesWrapper(moon, color=[0.8, 0.8, 0.8, 1.0])

    # ========================================
    # Create animations with keyframes
    # ========================================

    print("[Creating animations...]")
    print()

    # Animation for Sun - rotating in place
    sun_anim = Animation('sun_rotation', interpolation='linear')

    # 4 keyframes for full 360° rotation over 4 seconds
    for i in range(5):  # 0, 1, 2, 3, 4
        angle = (i / 4.0) * 2 * np.pi
        pose = PoseQuat.from_translation_axis_angle(
            translation=np.array([[0], [0], [0]]),
            axis=np.array([[0], [0], [1]]),
            angle=angle
        )
        sun_anim.add_keyframe(time=float(i), pose=pose)

    # Animation for Planet - orbiting sun
    planet_anim = Animation('planet_orbit', interpolation='smooth')

    orbit_radius = 3.0
    # 8 keyframes for orbit over 8 seconds
    for i in range(9):  # 0-8
        angle = (i / 8.0) * 2 * np.pi
        x = orbit_radius * math.cos(angle)
        y = orbit_radius * math.sin(angle)

        # Also rotate the planet as it orbits
        planet_rotation = angle * 2  # Rotate faster than orbit

        pose = PoseQuat.from_translation_axis_angle(
            translation=np.array([[x], [y], [0]]),
            axis=np.array([[0], [0], [1]]),
            angle=planet_rotation
        )
        planet_anim.add_keyframe(time=float(i), pose=pose)

    # Animation for Moon - orbiting planet
    moon_anim = Animation('moon_orbit', interpolation='smooth')

    moon_orbit = 0.8
    # 12 keyframes for fast orbit over 4 seconds (3 orbits during planet's 1 orbit)
    for i in range(13):  # 0-12
        t = i / 3.0  # Map to 0-4 seconds
        angle = (i / 12.0) * 2 * np.pi * 3  # 3 full orbits
        x = moon_orbit * math.cos(angle)
        y = moon_orbit * math.sin(angle)

        pose = PoseQuat.from_translation_axis_angle(
            translation=np.array([[x], [y], [0]]),
            axis=np.array([[1], [0], [0]]),
            angle=angle
        )
        moon_anim.add_keyframe(time=t, pose=pose)

    # Create animation clip
    clip = AnimationClip('solar_system_animation')
    clip.add_animation('Sun', sun_anim)
    clip.add_animation('Planet', planet_anim)
    clip.add_animation('Moon', moon_anim)

    print(f"Animation clip created:")
    print(f"  Duration: {clip.get_duration():.1f} seconds")
    print(f"  Objects: {', '.join(clip.get_object_names())}")
    print(f"  Sun keyframes: {sun_anim.get_keyframe_count()}")
    print(f"  Planet keyframes: {planet_anim.get_keyframe_count()}")
    print(f"  Moon keyframes: {moon_anim.get_keyframe_count()}")
    print()

    # Create animation player
    player = AnimationPlayer(clip, loop=True)
    player.play()

    print("[Starting animation playback]")
    print()
    print("Controls:")
    print("  - Animation loops automatically")
    print("  - Close window to exit")
    print()
    print("-" * 70)
    print()

    # Animation loop
    last_time = time.time()
    frame_count = 0
    last_report_time = last_time

    try:
        while not glfw.window_should_close(window):
            # Calculate delta time
            current_time = time.time()
            delta_time = current_time - last_time
            last_time = current_time

            # Update animation
            player.update(delta_time)

            # Get current poses and apply to objects
            poses = player.get_current_poses()

            if 'Sun' in poses:
                sun.set_pose(poses['Sun'].to_pose())
            if 'Planet' in poses:
                planet.set_pose(poses['Planet'].to_pose())
            if 'Moon' in poses:
                moon.set_pose(poses['Moon'].to_pose())

            # Render
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()

            # Camera position
            gluLookAt(
                8, 8, 8,  # Camera position
                0, 0, 0,  # Look at origin
                0, 0, 1   # Up vector
            )

            # Draw objects
            sun_wrapper.draw()
            planet_wrapper.draw()
            moon_wrapper.draw()

            glfw.swap_buffers(window)
            glfw.poll_events()

            # Progress report every 2 seconds
            frame_count += 1
            if current_time - last_report_time >= 2.0:
                anim_time = player.get_time()
                duration = clip.get_duration()
                progress = (anim_time / duration) * 100 if duration > 0 else 0
                fps = frame_count / (current_time - last_report_time)

                print(f"Time: {anim_time:.2f}s / {duration:.1f}s ({progress:.0f}%) | FPS: {fps:.1f}")

                frame_count = 0
                last_report_time = current_time

    except KeyboardInterrupt:
        print("\n[Interrupted by user]")
    finally:
        glfw.terminate()

    print()
    print("-" * 70)
    print()
    print("[Animation Statistics]")
    print(f"  Final time: {player.get_time():.2f}s")
    print(f"  Clip duration: {clip.get_duration():.1f}s")
    print(f"  Is playing: {player.is_playing()}")
    print()

    # Demonstrate different interpolation modes
    print("=" * 70)
    print("INTERPOLATION MODE COMPARISON")
    print("=" * 70)
    print()
    print("The animation system supports three interpolation modes:")
    print()

    # Create test animation with 2 keyframes
    test_start = PoseQuat.from_translation_axis_angle(
        np.array([[0], [0], [0]]),
        np.array([[0], [0], [1]]),
        0.0
    )
    test_end = PoseQuat.from_translation_axis_angle(
        np.array([[10], [0], [0]]),
        np.array([[0], [0], [1]]),
        np.pi
    )

    # Test each mode
    for mode in ['linear', 'step', 'smooth']:
        test_anim = Animation('test', interpolation=mode)
        test_anim.add_keyframe(0.0, test_start)
        test_anim.add_keyframe(2.0, test_end)

        print(f"{mode.upper()} interpolation:")
        for t in [0.0, 0.5, 1.0, 1.5, 2.0]:
            pose = test_anim.evaluate(t)
            x = pose.translation[0, 0]
            angle_axis, angle = pose.quaternion.to_axis_angle()
            angle_deg = math.degrees(angle)
            print(f"  t={t:.1f}s: x={x:.2f}, rotation={angle_deg:.1f}deg")
        print()

    print("=" * 70)
    print("ANIMATION EXAMPLE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
