import numpy as np
from typing import List

from src.datatypes.pose import Pose
from src.lights.base_light import BaseLight, LightPrimitive

class Spotlight(BaseLight):

    def __init__(self,
                 light_type: LightPrimitive,
                 diffuse: List[float] = None,
                 spot_cutoff: float = 45.0,
                 spot_exponent: float = 2.0,
                 pose: Pose = Pose(translation=np.array([[0], [0], [0]]), rotation=np.array([[0], [0], [1]])),
                 name: str = "Spotlight"):
        if diffuse is None:
            diffuse = [1.0, 1.0, 1.0, 1.0]
        assert len(diffuse) == 4, "Diffuse color must be a list of 4 floats"
        self.diffuse = diffuse
        self.spot_cutoff = spot_cutoff
        self.spot_exponent = spot_exponent
        self.pose = pose
        self.name = name

        super().__init__(light_type = light_type,
                         pose = pose,
                         diffuse = self.diffuse,
                         spot_cutoff = self.spot_cutoff,
                         spot_exponent = self.spot_exponent)

    def get_pose(self) -> Pose:
        pass

    def set_pose(self, pose: Pose):
        pass

    def get_name(self) -> str:
        pass

    def get_type(self) -> str:
        pass

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
