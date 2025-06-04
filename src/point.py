from typing import Optional

from src.datatypes.pose import Pose
from scene_object import SceneObject


class Point(SceneObject):

    def __init__(self, pose: Pose, name: str = "Point", parent: Optional[SceneObject] = None):
        self.pose = pose

    def get_pose(self):
        return self.pose

