"""
Advanced Rendering API Example

Demonstrates advanced features:
- Multiple objects with hierarchy
- Multiple lights
- Dynamic camera control
- Scene management
"""
import numpy as np

from src.rendering import Renderer, Scene, Camera
from src.primitives.sphere import Sphere
from src.primitives.cylinder import Cylinder
from src.datatypes.pose import Pose
from src.datatypes.material import MaterialPresets
from src.lights.spotlight import Spotlight
from src.lights.base_light import LightPrimitive


def main():
    # Create renderer
    renderer = Renderer(
        width=1024,
        height=768,
        title="Advanced Rendering Example",
        background_color=(0.05, 0.05, 0.1, 1.0)
    )

    # Create scene and camera
    scene = Scene(name="Solar System")
    camera = Camera.perspective(
        fov=45.0,
        position=[5, 5, 5],
        look_at=[0, 0, 0]
    )

    # Create a simple solar system hierarchy
    # Sun (parent)
    sun_pose = Pose(
        translation=np.array([[0], [0], [0]]),
        rotation=np.array([[0], [0], [0]])
    )
    sun = Sphere(
        pose=sun_pose,
        radius=1.0,
        material=MaterialPresets.gold(),
        name="Sun"
    )

    # Earth (child of Sun)
    earth_pose = Pose(
        translation=np.array([[3], [0], [0]]),  # 3 units from Sun
        rotation=np.array([[0], [0], [0]])
    )
    earth = Sphere(
        pose=earth_pose,
        radius=0.4,
        material=MaterialPresets.emerald(),
        name="Earth",
        parent=sun  # Earth orbits Sun
    )

    # Moon (child of Earth)
    moon_pose = Pose(
        translation=np.array([[0.8], [0], [0]]),  # 0.8 units from Earth
        rotation=np.array([[0], [0], [0]])
    )
    moon = Sphere(
        pose=moon_pose,
        radius=0.15,
        material=MaterialPresets.silver(),
        name="Moon",
        parent=earth  # Moon orbits Earth
    )

    # Add all objects to scene
    scene.add(sun)
    scene.add(earth)
    scene.add(moon)

    # Add multiple lights for better illumination
    # Main light from above
    light1_pose = Pose(
        translation=np.array([[0], [5], [0]]),
        rotation=np.array([[np.pi/2], [0], [0]])
    )
    light1 = Spotlight(
        light_type=LightPrimitive.SPOT,
        diffuse=[1.0, 1.0, 0.9, 1.0],
        spot_cutoff=45.0,
        spot_exponent=5.0,
        pose=light1_pose,
        name="TopLight"
    )

    # Secondary light from the side
    light2_pose = Pose(
        translation=np.array([[5], [2], [5]]),
        rotation=np.array([[0], [0], [0]])
    )
    light2 = Spotlight(
        light_type=LightPrimitive.SPOT,
        diffuse=[0.5, 0.5, 0.6, 1.0],
        spot_cutoff=60.0,
        spot_exponent=3.0,
        pose=light2_pose,
        name="SideLight"
    )

    scene.add_light(light1)
    scene.add_light(light2)

    # Animation state
    orbit_speed = 0.3
    rotation_speed = 1.0
    camera_angle = 0.0

    def update(delta_time):
        """Update function for animations"""
        nonlocal camera_angle

        # Rotate Sun on its axis
        sun_pose = sun.get_pose()
        sun.set_pose(Pose(
            translation=sun_pose.translation,
            rotation=sun_pose.rotation + np.array([[0], [rotation_speed * delta_time], [0]])
        ))

        # Rotate Earth around Sun (handled by hierarchy when we rotate Sun)
        # But also rotate Earth on its own axis
        earth_pose = earth.get_pose()
        earth.set_pose(Pose(
            translation=earth_pose.translation,
            rotation=earth_pose.rotation + np.array([[0], [rotation_speed * 2 * delta_time], [0]])
        ))

        # Moon automatically orbits Earth due to hierarchy
        # Just rotate it on its axis
        moon_pose = moon.get_pose()
        moon.set_pose(Pose(
            translation=moon_pose.translation,
            rotation=moon_pose.rotation + np.array([[0], [rotation_speed * delta_time], [0]])
        ))

        # Optional: Orbit camera around scene
        camera_angle += orbit_speed * delta_time * 0.2
        radius = 7.0
        camera.set_position([
            radius * np.cos(camera_angle),
            4.0,
            radius * np.sin(camera_angle)
        ])
        camera.set_look_at([0, 0, 0])

    # Run render loop
    print(f"Rendering {scene}")
    print(f"Objects: {scene.get_object_count()}")
    print(f"Lights: {scene.get_light_count()}")
    print("Watch the hierarchy in action!")
    print("Press ESC or close window to exit")

    renderer.run(scene, camera, update_callback=update, fps_limit=60)

    # Cleanup
    renderer.close()
    print("Done!")


if __name__ == "__main__":
    main()
