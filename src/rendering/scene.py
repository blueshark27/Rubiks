"""
Scene management for organizing objects and lights.

Provides a high-level container for scene graph elements without exposing OpenGL details.
"""
from typing import List, Optional
from src.primitives.base_scene_object import BaseSceneObject
from src.primitives.opengl_primitives_wrapper import OpenGLPrimitivesWrapper
from src.lights.base_light import BaseLight
from src.lights.opengl_light_wrapper import OpenGLLightWrapper


class Scene:
    """
    Scene container for managing objects and lights.

    Handles automatic wrapper creation and provides high-level add/remove operations.
    """

    def __init__(self, name: str = "Scene"):
        """
        Initialize scene.

        Args:
            name: Scene name for identification
        """
        self.name = name
        self._objects: List[BaseSceneObject] = []
        self._object_wrappers: List[OpenGLPrimitivesWrapper] = []
        self._lights: List[BaseLight] = []
        self._light_wrappers: List[OpenGLLightWrapper] = []
        self._next_light_index = 0

    def add(self, obj: BaseSceneObject, color: Optional[List[float]] = None):
        """
        Add an object to the scene.

        Args:
            obj: BaseSceneObject to add (Sphere, Cylinder, Cone, etc.)
            color: Optional color override [r, g, b, a]
        """
        if obj not in self._objects:
            self._objects.append(obj)
            wrapper = OpenGLPrimitivesWrapper(obj, color=color)
            self._object_wrappers.append(wrapper)

    def remove(self, obj: BaseSceneObject):
        """
        Remove an object from the scene.

        Args:
            obj: BaseSceneObject to remove
        """
        if obj in self._objects:
            idx = self._objects.index(obj)
            self._objects.pop(idx)
            self._object_wrappers.pop(idx)

    def add_light(self, light: BaseLight):
        """
        Add a light to the scene.

        Automatically assigns light index (GL_LIGHT0, GL_LIGHT1, etc.).

        Args:
            light: BaseLight to add (Spotlight, etc.)

        Raises:
            ValueError: If maximum number of lights (8) is reached
        """
        if self._next_light_index >= 8:
            raise ValueError("Maximum number of lights (8) reached")

        if light not in self._lights:
            self._lights.append(light)
            wrapper = OpenGLLightWrapper(light, index=self._next_light_index)
            self._light_wrappers.append(wrapper)
            self._next_light_index += 1

    def remove_light(self, light: BaseLight):
        """
        Remove a light from the scene.

        Args:
            light: BaseLight to remove
        """
        if light in self._lights:
            idx = self._lights.index(light)
            self._lights.pop(idx)
            self._light_wrappers.pop(idx)
            # Note: Light indices don't shift, so _next_light_index stays the same

    def get_objects(self) -> List[BaseSceneObject]:
        """Get list of all objects in the scene."""
        return self._objects.copy()

    def get_lights(self) -> List[BaseLight]:
        """Get list of all lights in the scene."""
        return self._lights.copy()

    def clear_objects(self):
        """Remove all objects from the scene."""
        self._objects.clear()
        self._object_wrappers.clear()

    def clear_lights(self):
        """Remove all lights from the scene."""
        self._lights.clear()
        self._light_wrappers.clear()
        self._next_light_index = 0

    def clear(self):
        """Remove all objects and lights from the scene."""
        self.clear_objects()
        self.clear_lights()

    def render(self):
        """
        Render all objects in the scene.

        This is called internally by the Renderer.
        """
        for wrapper in self._object_wrappers:
            wrapper.draw()

    def setup_lights(self):
        """
        Setup all lights in the scene.

        This is called internally by the Renderer during initialization.
        """
        for wrapper in self._light_wrappers:
            wrapper.setup_lighting()

    def get_object_count(self) -> int:
        """Get number of objects in the scene."""
        return len(self._objects)

    def get_light_count(self) -> int:
        """Get number of lights in the scene."""
        return len(self._lights)

    def __str__(self) -> str:
        return f"Scene('{self.name}', objects={self.get_object_count()}, lights={self.get_light_count()})"

    def __repr__(self) -> str:
        return self.__str__()
