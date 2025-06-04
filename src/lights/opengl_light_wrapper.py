from OpenGL.GL import *

from src.lights.base_light import BaseLight


class OpenGLLightWrapper:
    def __init__(self, light: BaseLight, index: int = 0):
        self.light = light
        self.index = index

    def setup_lighting(self):
        glEnable(eval(f"GL_LIGHT{self.index}"))
        for l in self.light.get_options():
            if type(self.light.get_options()[l]) is list:
                glLightfv(eval(f"GL_LIGHT{self.index}"), eval(f"GL_{l.upper()}"), self.light.get_options()[l])
            else:
                glLightf(eval(f"GL_LIGHT{self.index}"), eval(f"GL_{l.upper()}"), self.light.get_options()[l])

        pose = self.light.get_pose()
        glLightfv(eval(f"GL_LIGHT{self.index}"), GL_POSITION, self.light.pose.get_translation_as_homogeneous())
        if self.light.get_light_type() == "SPOTLIGHT":
            glLightfv(GL_LIGHT3, GL_SPOT_DIRECTION, [0.0, -1.0, 0.0])
