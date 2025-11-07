import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *

from src.primitives.sphere import Sphere
from src.primitives.opengl_primitives_wrapper import OpenGLPrimitivesWrapper
from src.datatypes.pose import Pose
from src.lights.spotlight import Spotlight
from src.lights.opengl_light_wrapper import OpenGLLightWrapper
from src.lights.base_light import LightPrimitive

def init_window(width, height, title):
    if not glfw.init():
        raise Exception("GLFW konnte nicht initialisiert werden!")

    window = glfw.create_window(width, height, title, None, None)
    if not window:
        glfw.terminate()
        raise Exception("Fenster konnte nicht erstellt werden!")

    glfw.make_context_current(window)
    return window

def main():
    # Create window and OpenGL context FIRST
    window = init_window(800, 600, "Sphere example")

    # Now we can create the sphere and configure OpenGL
    pose = Pose(translation=np.array([[0], [0], [0]]), rotation=np.array([[0], [0], [1]]))
    glWrapper = OpenGLPrimitivesWrapper(Sphere(pose=pose, radius=1.0))

    # Create and setup spotlight
    # Rotation uses axis-angle representation: axis * angle (in radians)
    # Example: [1, 0, 0] with magnitude π/4 means rotate π/4 radians around X-axis
    # Zero rotation means spotlight points down (0, -1, 0)
    light_rotation = np.array([[np.pi/4], [0], [0]])  # Rotate 45° around X-axis
    light_pose = Pose(translation=np.array([[3], [3], [3]]), rotation=light_rotation)
    spotlight = Spotlight(
        light_type=LightPrimitive.SPOT,
        diffuse=[1.0, 1.0, 1.0, 1.0],
        spot_cutoff=30.0,
        spot_exponent=10.0,
        pose=light_pose,
        name="MainSpotlight"
    )
    lightWrapper = OpenGLLightWrapper(spotlight, index=0)

    # Enable lighting
    glEnable(GL_LIGHTING)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)
    glEnable(GL_NORMALIZE)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    # Setup the light
    lightWrapper.setup_lighting()

    # Set material properties for the sphere
    glMaterialfv(GL_FRONT, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
    glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
    glMaterialfv(GL_FRONT, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
    glMaterialf(GL_FRONT, GL_SHININESS, 50.0)


    while not glfw.window_should_close(window):
        glfw.poll_events()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, 800/600, 0.1, 100.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(2,2,2, 0,0,0, 0,1,0)

        glWrapper.draw()

        glfw.swap_buffers(window)

    glfw.terminate()


if __name__ == "__main__":
    main()
