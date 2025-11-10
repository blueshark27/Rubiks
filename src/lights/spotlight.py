import numpy as np
from typing import List

from src.datatypes.pose import Pose
from src.lights.base_light import BaseLight, LightPrimitive

class Spotlight(BaseLight):

    def __init__(self,
                 light_type: LightPrimitive,
                 diffuse: List[float] = None,
                 ambient: List[float] = None,
                 specular: List[float] = None,
                 spot_cutoff: float = 45.0,
                 spot_exponent: float = 2.0,
                 pose: Pose = Pose(translation=np.array([[0], [0], [0]]), rotation=np.array([[0], [0], [1]])),
                 name: str = "Spotlight"):
        # Set default values for lighting components
        if diffuse is None:
            diffuse = [1.0, 1.0, 1.0, 1.0]
        if ambient is None:
            ambient = [0.2, 0.2, 0.2, 1.0]  # Soft background illumination
        if specular is None:
            specular = [1.0, 1.0, 1.0, 1.0]  # Bright highlights

        # Validate lighting components
        assert len(diffuse) == 4, "Diffuse color must be a list of 4 floats"
        assert len(ambient) == 4, "Ambient color must be a list of 4 floats"
        assert len(specular) == 4, "Specular color must be a list of 4 floats"

        self.diffuse = diffuse
        self.ambient = ambient
        self.specular = specular
        self.spot_cutoff = spot_cutoff
        self.spot_exponent = spot_exponent

        super().__init__(light_type = light_type,
                         pose = pose,
                         name = name,
                         diffuse = self.diffuse,
                         ambient = self.ambient,
                         specular = self.specular,
                         spot_cutoff = self.spot_cutoff,
                         spot_exponent = self.spot_exponent)

    def get_type(self) -> str:
        return "SPOTLIGHT"

    def get_parent(self):
        return None

    def set_parent(self, parent):
        pass

    def get_children(self) -> List:
        return []

    def add_child(self, child):
        pass

    def remove_child(self, child):
        pass
