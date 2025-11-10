"""
High-level rendering module for the Rubiks framework.

Provides clean abstractions for OpenGL rendering without exposing low-level details.

Example usage:
    >>> from src.rendering import Renderer, Scene, Camera
    >>> from src.primitives.sphere import Sphere
    >>> from src.datatypes.pose import Pose
    >>> from src.datatypes.material import MaterialPresets
    >>>
    >>> # Create renderer and scene
    >>> renderer = Renderer(width=800, height=600, title="My App")
    >>> scene = Scene()
    >>> camera = Camera.perspective(fov=45, position=[2, 2, 2], look_at=[0, 0, 0])
    >>>
    >>> # Add objects
    >>> sphere = Sphere(pose=Pose(...), material=MaterialPresets.gold())
    >>> scene.add(sphere)
    >>>
    >>> # Render
    >>> renderer.run(scene, camera)
"""

from src.rendering.renderer import Renderer
from src.rendering.scene import Scene
from src.rendering.camera import Camera

__all__ = ['Renderer', 'Scene', 'Camera']
