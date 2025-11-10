"""
High-level renderer for managing OpenGL rendering pipeline.

Handles window creation, OpenGL state, and render loop without exposing low-level details.
"""
import glfw
from OpenGL.GL import *
from typing import Optional, Callable
from src.rendering.scene import Scene
from src.rendering.camera import Camera


class Renderer:
    """
    High-level renderer managing window and rendering pipeline.

    Encapsulates all OpenGL initialization and state management.
    """

    def __init__(self,
                 width: int = 800,
                 height: int = 600,
                 title: str = "Rubiks Framework",
                 enable_depth_test: bool = True,
                 enable_lighting: bool = True,
                 background_color: tuple = (0.1, 0.1, 0.1, 1.0)):
        """
        Initialize renderer and create window.

        Args:
            width: Window width in pixels
            height: Window height in pixels
            title: Window title
            enable_depth_test: Enable depth testing (default: True)
            enable_lighting: Enable lighting (default: True)
            background_color: RGBA background color (default: dark gray)

        Raises:
            RuntimeError: If GLFW or window initialization fails
        """
        self.width = width
        self.height = height
        self.title = title
        self.background_color = background_color

        # Initialize GLFW
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        # Create window
        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create window")

        glfw.make_context_current(self.window)

        # Setup OpenGL state
        if enable_depth_test:
            glEnable(GL_DEPTH_TEST)

        if enable_lighting:
            glEnable(GL_LIGHTING)
            glEnable(GL_NORMALIZE)
            glShadeModel(GL_SMOOTH)

        # Set background color
        glClearColor(*self.background_color)

        # Window resize callback
        glfw.set_window_size_callback(self.window, self._on_resize)

        self._is_running = False

    def _on_resize(self, window, width: int, height: int):
        """Handle window resize events."""
        self.width = width
        self.height = height
        glViewport(0, 0, width, height)

    def should_close(self) -> bool:
        """Check if window should close."""
        return glfw.window_should_close(self.window)

    def clear(self, color_buffer: bool = True, depth_buffer: bool = True):
        """
        Clear buffers.

        Args:
            color_buffer: Clear color buffer
            depth_buffer: Clear depth buffer
        """
        mask = 0
        if color_buffer:
            mask |= GL_COLOR_BUFFER_BIT
        if depth_buffer:
            mask |= GL_DEPTH_BUFFER_BIT

        if mask:
            glClear(mask)

    def render_frame(self, scene: Scene, camera: Camera):
        """
        Render a single frame.

        Args:
            scene: Scene to render
            camera: Camera for view/projection
        """
        # Clear buffers
        self.clear()

        # Update camera aspect ratio
        camera.set_aspect_ratio(self.width, self.height)

        # Apply camera transformations
        camera.apply()

        # Render scene
        scene.render()

        # Swap buffers
        glfw.swap_buffers(self.window)

    def run(self,
            scene: Scene,
            camera: Camera,
            update_callback: Optional[Callable[[float], None]] = None,
            fps_limit: int = 60):
        """
        Run the main render loop.

        Args:
            scene: Scene to render
            camera: Camera for view/projection
            update_callback: Optional callback called each frame with delta_time
            fps_limit: Maximum FPS (0 for unlimited)

        Example:
            >>> def update(dt):
            ...     sphere.rotate([0, 0, dt])
            >>> renderer.run(scene, camera, update_callback=update)
        """
        self._is_running = True

        # Setup lights once at start
        scene.setup_lights()

        last_time = glfw.get_time()
        frame_time = 1.0 / fps_limit if fps_limit > 0 else 0.0

        while not self.should_close() and self._is_running:
            # Calculate delta time
            current_time = glfw.get_time()
            delta_time = current_time - last_time

            # FPS limiting
            if fps_limit > 0 and delta_time < frame_time:
                continue

            last_time = current_time

            # Process events
            glfw.poll_events()

            # Call update callback
            if update_callback:
                update_callback(delta_time)

            # Render frame
            self.render_frame(scene, camera)

    def stop(self):
        """Stop the render loop."""
        self._is_running = False

    def close(self):
        """Close window and terminate GLFW."""
        glfw.terminate()

    def get_window_size(self) -> tuple:
        """Get current window size."""
        return (self.width, self.height)

    def set_background_color(self, r: float, g: float, b: float, a: float = 1.0):
        """
        Set background clear color.

        Args:
            r: Red component (0.0-1.0)
            g: Green component (0.0-1.0)
            b: Blue component (0.0-1.0)
            a: Alpha component (0.0-1.0)
        """
        self.background_color = (r, g, b, a)
        glClearColor(r, g, b, a)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
