"""
High-Level API Example - Clean and Simple

Demonstrates the new rendering API that hides OpenGL details.
Compare this to sphere_example.py to see the improvement!
"""
import numpy as np

from src.rendering import Renderer, Scene, Camera
from src.primitives.sphere import Sphere
from src.datatypes.pose import Pose
from src.datatypes.material import MaterialPresets
from src.lights.spotlight import Spotlight
from src.lights.base_light import LightPrimitive


def main():
    # Create renderer with desired settings
    renderer = Renderer(
        width=800,
        height=600,
        title="High-Level API Example",
        background_color=(0.1, 0.1, 0.15, 1.0)
    )

    # Create scene and camera
    scene = Scene(name="Simple Scene")
    camera = Camera.perspective(
        fov=45.0,
        position=[2, 2, 2],
        look_at=[0, 0, 0]
    )

    # Create spheres with different materials
    sphere1_pose = Pose(
        translation=np.array([[-1.5], [0], [0]]),
        rotation=np.array([[0], [0], [0]])
    )
    sphere1 = Sphere(pose=sphere1_pose, radius=0.8, material=MaterialPresets.gold())

    sphere2_pose = Pose(
        translation=np.array([[0], [0], [0]]),
        rotation=np.array([[0], [0], [0]])
    )
    sphere2 = Sphere(pose=sphere2_pose, radius=0.8, material=MaterialPresets.emerald())

    sphere3_pose = Pose(
        translation=np.array([[1.5], [0], [0]]),
        rotation=np.array([[0], [0], [0]])
    )
    sphere3 = Sphere(pose=sphere3_pose, radius=0.8, material=MaterialPresets.chrome())

    # Add objects to scene
    scene.add(sphere1)
    scene.add(sphere2)
    scene.add(sphere3)

    # Add lighting
    light_pose = Pose(
        translation=np.array([[3], [3], [3]]),
        rotation=np.array([[np.pi/4], [0], [0]])
    )
    spotlight = Spotlight(
        light_type=LightPrimitive.SPOT,
        diffuse=[1.0, 1.0, 1.0, 1.0],
        spot_cutoff=30.0,
        spot_exponent=10.0,
        pose=light_pose,
        name="MainSpotlight"
    )
    scene.add_light(spotlight)

    # Optional: Define update callback for animations
    rotation_speed = 0.5

    def update(delta_time):
        """Update function called each frame"""
        # Rotate spheres
        angle = rotation_speed * delta_time

        # Update sphere1 rotation
        current_pose1 = sphere1.get_pose()
        new_rotation1 = current_pose1.rotation + np.array([[0], [angle], [0]])
        sphere1.set_pose(Pose(translation=current_pose1.translation, rotation=new_rotation1))

        # Update sphere2 rotation (different axis)
        current_pose2 = sphere2.get_pose()
        new_rotation2 = current_pose2.rotation + np.array([[angle], [0], [0]])
        sphere2.set_pose(Pose(translation=current_pose2.translation, rotation=new_rotation2))

        # Update sphere3 rotation (combined)
        current_pose3 = sphere3.get_pose()
        new_rotation3 = current_pose3.rotation + np.array([[angle], [angle], [0]])
        sphere3.set_pose(Pose(translation=current_pose3.translation, rotation=new_rotation3))

    # Run render loop
    print(f"Rendering {scene}")
    print("Press ESC or close window to exit")

    renderer.run(scene, camera, update_callback=update, fps_limit=60)

    # Cleanup
    renderer.close()
    print("Done!")


if __name__ == "__main__":
    main()
