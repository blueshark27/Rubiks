from enum import Enum
from typing import Dict, List

from src.scene_object import SceneObject
from src.datatypes.pose import Pose

class LightPrimitive(Enum):
    AMBIENT = 1
    DIRECTIONAL = 2
    SPOT = 3
    POINT = 4


class BaseLight(SceneObject):

    def __init__(self, light_type: LightPrimitive, pose: Pose, name: str = "Light", **options):
        super().__init__(name=name, pose=pose)
        self.__options = {}
        self.__options_v = {}
        for o_name in options:
            self.__options[o_name] = options[o_name]
        self.light_type = light_type

    def get_options(self) -> Dict:
        return self.__options

    def get_pose(self) -> Pose:
        return self.pose

    def set_pose(self, pose: Pose):
        self.pose = pose

    def get_name(self) -> str:
        pass

    def get_light_type(self) -> LightPrimitive:
        return self.light_type

    def get_parent(self):
        pass

    def set_parent(self, parent):
        pass

    def get_children(self) -> List:
        pass

    def add_child(self, child):
        pass

    def remove_child(self, child):
        pass
